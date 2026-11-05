#!/usr/bin/env python3
"""遍历 fc1 / qkv 块顺序组合，定位等价性偏差来源。

对每个候选顺序：加载转换权重（按顺序重排块），前向，与 HF logits 比 cosine。
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

HF_DIR = "/ws/model"
CKPT = "/ws/musa_framework/generated/ckpt_13b/iter_0000001/mp_rank_00/model_optim_rng.pt"
SEQ = 8
VOCAB = 69104
HIDDEN = 5120
FFN = 13824
NLAYERS = 40
NHEADS = 40


def build_mcore() -> GPTModel:
    config = TransformerConfig(
        num_layers=NLAYERS, hidden_size=HIDDEN, ffn_hidden_size=FFN,
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
    model.cuda().to(torch.bfloat16)
    model.eval()
    return model


def reorder(fc1_blocks: tuple[int, int], qkv_blocks: tuple[int, int, int]):
    """返回一个转换函数：把 ckpt 的 fc1/qkv 块按 (a,b,c) 顺序重排。"""
    def apply(ck: dict):
        out = dict(ck)
        for i in range(NLAYERS):
            fk = f"decoder.layers.{i}.mlp.linear_fc1.weight"
            t = ck[fk]
            ch = t.shape[0] // 2
            halves = [t[0:ch], t[ch:2 * ch]]
            out[fk] = torch.cat([halves[fc1_blocks[0]], halves[fc1_blocks[1]]], dim=0)
            qk = f"decoder.layers.{i}.self_attention.linear_qkv.weight"
            t = ck[qk]
            ch = t.shape[0] // 3
            q, k, v = t[0:ch], t[ch:2 * ch], t[2 * ch:3 * ch]
            blk = {"q": q, "k": k, "v": v}
            out[qk] = torch.cat([blk[b] for b in qkv_blocks], dim=0)
        return out
    return apply


def main() -> int:
    from transformers import LlamaForCausalLM

    hf = LlamaForCausalLM.from_pretrained(HF_DIR, torch_dtype=torch.bfloat16)
    hf.eval()

    dist.init_process_group("gloo", rank=0, world_size=1, init_method="tcp://127.0.0.1:29995")
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(0)
    model = build_mcore()

    base_ck = torch.load(CKPT, map_location="cpu", weights_only=True)["model"]

    torch.manual_seed(1234)
    input_ids = torch.randint(0, VOCAB, (1, SEQ), dtype=torch.long)

    with torch.no_grad():
        hf_logits = hf(input_ids=input_ids).logits.float()

        device = torch.device("musa:0")
        ids = input_ids.to(device)
        position_ids = torch.arange(SEQ, dtype=torch.long, device=device).unsqueeze(0)
        attn_mask = torch.tril(torch.ones(SEQ, SEQ, dtype=torch.bool, device=device)).unsqueeze(0).unsqueeze(0)

        def cos_for(transform):
            model.load_state_dict(transform(base_ck), strict=False)
            with torch.no_grad():
                out = model(input_ids=ids, position_ids=position_ids, attention_mask=attn_mask)
            a = hf_logits[0].double()
            b = out.float().cpu()[0].double()
            return torch.nn.functional.cosine_similarity(a, b, dim=1).mean().item()

    cases = [
        ("fc1[g;u] qkv[q;k;v]", (0, 1), ("q", "k", "v")),
        ("fc1[u;g] qkv[q;k;v]", (1, 0), ("q", "k", "v")),
        ("fc1[g;u] qkv[k;q;v]", (0, 1), ("k", "q", "v")),
        ("fc1[g;u] qkv[v;k;q]", (0, 1), ("v", "k", "q")),
        ("fc1[g;u] qkv[q;v;k]", (0, 1), ("q", "v", "k")),
    ]
    for name, fcb, qkvb in cases:
        c = cos_for(reorder(fcb, qkvb))
        print(f"{name:28s} mean cosine = {c:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
