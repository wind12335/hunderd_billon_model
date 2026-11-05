#!/usr/bin/env python3
"""验证 qkv 布局假说: 把 linear_qkv.weight 重排为按头交错 [q0;k0;v0; q1;k1;v1; ...]。

若等价测试 cosine 从 0.6 跳到 ~1.0,即确认转换器布局 bug。
"""
from __future__ import annotations

import sys
from pathlib import Path

_SELF_DIR = Path(__file__).resolve().parent
if str(_SELF_DIR) not in sys.path:
    sys.path.insert(0, str(_SELF_DIR))

from megatron_musa_bootstrap import bootstrap  # noqa: E402

bootstrap(verbose=False)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torch.distributed as dist  # noqa: E402

from megatron.core import parallel_state  # noqa: E402
from megatron.core.models.gpt import GPTModel  # noqa: E402
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec  # noqa: E402
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed  # noqa: E402
from megatron.core.transformer import TransformerConfig  # noqa: E402

HF_DIR = "/ws/model"
CKPT = "/ws/musa_framework/generated/ckpt_13b/iter_0000001/mp_rank_00/model_optim_rng.pt"
SEQ = 8
VOCAB = 69104
HIDDEN = 5120
NLAYERS = 40
NHEADS = 40
HEAD_DIM = HIDDEN // NHEADS


def interleave_qkv(sd: dict) -> dict:
    """把 [q_all;k_all;v_all] 重排为按头交错 [q0;k0;v0; ...]。"""
    out = dict(sd)
    for i in range(NLAYERS):
        k = f"decoder.layers.{i}.self_attention.linear_qkv.weight"
        t = out[k]  # [15360, 5120]
        q, kk, v = t[0:HIDDEN], t[HIDDEN:2 * HIDDEN], t[2 * HIDDEN:3 * HIDDEN]
        # 每头 128 行,交错成 [q_g;k_g;v_g]
        q = q.view(NHEADS, HEAD_DIM, -1)
        kk = kk.view(NHEADS, HEAD_DIM, -1)
        v = v.view(NHEADS, HEAD_DIM, -1)
        out[k] = torch.cat([q, kk, v], dim=1).reshape(3 * HIDDEN, -1)
    return out


def main() -> int:
    from transformers import LlamaForCausalLM

    hf = LlamaForCausalLM.from_pretrained(HF_DIR, torch_dtype=torch.bfloat16).eval()

    dist.init_process_group("gloo", rank=0, world_size=1, init_method="tcp://127.0.0.1:29991")
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(0)

    config = TransformerConfig(
        num_layers=NLAYERS, hidden_size=HIDDEN, ffn_hidden_size=13824,
        num_attention_heads=NHEADS, num_query_groups=NHEADS,
        normalization="RMSNorm", layernorm_epsilon=1e-6,
        gated_linear_unit=True, activation_func=torch.nn.functional.silu,
        add_bias_linear=False, params_dtype=torch.bfloat16,
    )
    if not hasattr(config, "offload_moe_fc1_input"):
        config.offload_moe_fc1_input = False
    model = GPTModel(
        config, transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=VOCAB, max_sequence_length=4096,
        position_embedding_type="rope", rotary_base=10000,
    )
    model.cuda().to(torch.bfloat16).eval()

    raw = torch.load(CKPT, map_location="cpu", weights_only=True)["model"]
    interleaved = interleave_qkv(raw)
    ret = model.load_state_dict(interleaved, strict=True)
    assert not ret.missing_keys and not ret.unexpected_keys, ret
    print("[verify] interleaved qkv strict 加载通过")

    torch.manual_seed(1234)
    input_ids = torch.randint(0, VOCAB, (1, SEQ), dtype=torch.long)

    with torch.no_grad():
        hf_logits = hf(input_ids=input_ids).logits.float()

        ids = input_ids.to("musa:0")
        pos = torch.arange(SEQ, dtype=torch.long, device="musa:0").unsqueeze(0)
        mask = torch.tril(torch.ones(SEQ, SEQ, dtype=torch.bool, device="musa:0")).unsqueeze(0).unsqueeze(0)
        mc_out = model(input_ids=ids, position_ids=pos, attention_mask=mask)
        mc_logits = mc_out.float().cpu()

    hf_l, mc_l = hf_logits[0], mc_logits[0]
    cos_all = F.cosine_similarity(hf_l.double(), mc_l.double(), dim=1)
    print(f"mean cosine over all positions: {cos_all.mean().item():.6f}")
    for p in [0, 1, 3, 7]:
        a, b = hf_l[p].double(), mc_l[p].double()
        c = F.cosine_similarity(a, b, dim=0).item()
        print(f"  pos {p}: cosine={c:.6f}  hf_norm={a.norm().item():.2f}  mc_norm={b.norm().item():.2f}")
    ok = cos_all.mean().item() > 0.995
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
