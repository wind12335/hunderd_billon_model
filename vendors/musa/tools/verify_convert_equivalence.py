#!/usr/bin/env python3
"""数值等价验证：转换后的 mcore 模型 vs HF 原模型，同一输入对比 logits。

一次校验：权重融合顺序（qkv=[q;k;v]、fc1=[gate;up]）、RoPE 参数、
RMSNorm eps、embedding/lm_head 绑定关系等。若任一环节映射错误，
logits 会明显偏差。
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
import torch.distributed as dist  # noqa: E402

from megatron.core import parallel_state  # noqa: E402
from megatron.core.models.gpt import GPTModel  # noqa: E402
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec  # noqa: E402
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed  # noqa: E402
from megatron.core.transformer import TransformerConfig  # noqa: E402

import os

HF_DIR = "/ws/model"
CKPT = "/ws/musa_framework/generated/ckpt_13b/iter_0000001/mp_rank_00/model_optim_rng.pt"
SEQ = int(os.environ.get("SEQ", "8"))
BATCH = 1
VOCAB = 69104
HIDDEN = 5120
FFN = 13824
NLAYERS = 40
NHEADS = 40


def build_mcore() -> GPTModel:
    dist.init_process_group("gloo", rank=0, world_size=1, init_method="tcp://127.0.0.1:29994")
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(0)

    config = TransformerConfig(
        num_layers=NLAYERS,
        hidden_size=HIDDEN,
        ffn_hidden_size=FFN,
        num_attention_heads=NHEADS,
        num_query_groups=NHEADS,
        normalization="RMSNorm",
        layernorm_epsilon=1e-6,
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        add_bias_linear=False,
        params_dtype=torch.bfloat16,
    )
    if not hasattr(config, "offload_moe_fc1_input"):
        config.offload_moe_fc1_input = False
    model = GPTModel(
        config,
        transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=VOCAB,
        max_sequence_length=4096,
        position_embedding_type="rope",
        rotary_base=10000,
    )
    model.cuda().to(torch.bfloat16)
    model.eval()

    ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
    ret = model.load_state_dict(ckpt["model"], strict=True)
    assert not ret.missing_keys and not ret.unexpected_keys, ret
    print("[verify] mcore 权重 strict 加载通过")
    return model


def main() -> int:
    from transformers import LlamaForCausalLM

    hf = LlamaForCausalLM.from_pretrained(HF_DIR, torch_dtype=torch.bfloat16)
    hf.eval()

    mcore = build_mcore()

    torch.manual_seed(1234)
    input_ids = torch.randint(0, VOCAB, (BATCH, SEQ), dtype=torch.long)

    with torch.no_grad():
        hf_logits = hf(input_ids=input_ids).logits.float()  # (1, s, V) fp32 CPU

        device = torch.device("musa:0")
        ids = input_ids.to(device)
        position_ids = torch.arange(SEQ, dtype=torch.long, device=device).unsqueeze(0)
        attn_mask = torch.tril(
            torch.ones(SEQ, SEQ, dtype=torch.bool, device=device)
        ).unsqueeze(0).unsqueeze(0)
        mc_out = mcore(input_ids=ids, position_ids=position_ids, attention_mask=attn_mask)
        mc_logits = mc_out.float().cpu()  # (1, s, V) fp32

    hf_l, mc_l = hf_logits[0], mc_logits[0]  # (s, V)
    print(f"{'pos':>4} {'cosine':>10} {'max_abs_diff':>14} {'hf_norm':>10} {'mc_norm':>10}")
    ok = True
    for p in [0, 1, 3, 7]:
        a, b = hf_l[p].double(), mc_l[p].double()
        cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
        mad = (a - b).abs().max().item()
        print(f"{p:>4} {cos:>10.6f} {mad:>14.6f} {a.norm().item():>10.2f} {b.norm().item():>10.2f}")
        if cos < 0.99:
            ok = False
    # 均值余弦（更稳）
    cos_all = torch.nn.functional.cosine_similarity(hf_l.double(), mc_l.double(), dim=1)
    print(f"mean cosine over all positions: {cos_all.mean().item():.6f}")
    ok = ok and cos_all.mean().item() > 0.995
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
