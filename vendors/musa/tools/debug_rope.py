#!/usr/bin/env python3
"""对比 post-RoPE 的 q/k/v: mcore(MUSA) vs HF。定位 RoPE 应用是否正确。

RoPE 本应 norm-preserving。若 post-rope q/k/v 的 norm 或方向偏离 HF,就是 RoPE 的 bug。
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
NHEADS = 40
HEAD_DIM = 128


def cos(a, b):
    a, b = a.cpu().float(), b.cpu().float()
    return F.cosine_similarity(a.double(), b.double(), dim=-1).mean().item()


def main() -> int:
    from transformers import LlamaForCausalLM
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    hf = LlamaForCausalLM.from_pretrained(HF_DIR, torch_dtype=torch.bfloat16).eval()

    dist.init_process_group("gloo", rank=0, world_size=1, init_method="tcp://127.0.0.1:29992")
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(0)

    config = TransformerConfig(
        num_layers=40, hidden_size=HIDDEN, ffn_hidden_size=13824,
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
    model.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True)["model"], strict=True)

    torch.manual_seed(1234)
    input_ids = torch.randint(0, VOCAB, (1, SEQ), dtype=torch.long)

    cap: dict = {}

    def cap_ca(_m, inp, _out):
        cap["q"], cap["k"], cap["v"] = inp[0].detach(), inp[1].detach(), inp[2].detach()

    mc0 = model.decoder.layers[0]
    mc0.self_attention.core_attention.register_forward_hook(cap_ca)

    with torch.no_grad():
        ids = input_ids.to("musa:0")
        pos = torch.arange(SEQ, dtype=torch.long, device="musa:0").unsqueeze(0)
        mask = torch.tril(torch.ones(SEQ, SEQ, dtype=torch.bool, device="musa:0")).unsqueeze(0).unsqueeze(0)
        model(input_ids=ids, position_ids=pos, attention_mask=mask)

    # --- HF 的 post-rope q/k/v ---
    hf0 = hf.model.layers[0]
    with torch.no_grad():
        hf_out = hf(input_ids=input_ids, output_hidden_states=True)
        ln = hf0.input_layernorm(hf_out.hidden_states[0])
        hf_q = hf0.self_attn.q_proj(ln).view(1, SEQ, NHEADS, HEAD_DIM).transpose(1, 2)   # [b,n,s,d]
        hf_k = hf0.self_attn.k_proj(ln).view(1, SEQ, NHEADS, HEAD_DIM).transpose(1, 2)
        hf_v = hf0.self_attn.v_proj(ln).view(1, SEQ, NHEADS, HEAD_DIM).transpose(1, 2)
        pos_ids = torch.arange(SEQ).unsqueeze(0)
        cos_, sin_ = hf.model.rotary_emb(hf_q, pos_ids)
        hf_qr, hf_kr = apply_rotary_pos_emb(hf_q, hf_k, cos_, sin_, position_ids=pos_ids)  # [b,n,s,d]

    # --- mc 的 post-rope q/k/v: [sq, b, n, d] -> [b, n, s, d] ---
    mc_q = cap["q"].transpose(0, 1).transpose(1, 2)  # [b, n, s, d]
    mc_k = cap["k"].transpose(0, 1).transpose(1, 2)
    mc_v = cap["v"].transpose(0, 1).transpose(1, 2)

    print("== post-RoPE q/k/v: HF vs MC(MUSA) ==")
    for name, a, b in [("q", hf_qr, mc_q), ("k", hf_kr, mc_k), ("v", hf_v, mc_v)]:
        ra = a.float().norm().item() / (a.numel() ** 0.5) * (128 ** 0.5)
        print(f"  {name}: cosine={cos(a, b):9.6f}  hf_norm={a.float().norm().item():9.2f}  mc_norm={b.float().norm().item():9.2f}")

    # 若 q/k 方向对但 norm 不对: RoPE 应用了非酉变换
    # 若 direction 对: 说明 RoPE 本身 OK
    # pre-rope 对照
    hf_q_flat = hf_q.transpose(1, 2).reshape(1, SEQ, HIDDEN)  # [b,s,h]
    mc_q_flat = cap["q"].transpose(0, 1).reshape(1, SEQ, HIDDEN)  # [b,s,h]
    print(f"  [ref] pre-rope q: hf_norm={hf_q_flat.float().norm().item():9.2f} mc_norm={mc_q_flat.float().norm().item():9.2f}")
    # 看 mc q 是否等于 hf 未旋转的 q (即 rope 根本没生效?)
    print(f"  [ref] mc post-rope q vs hf PRE-rope q: cosine={cos(mc_q, hf_q):9.6f}")
    print(f"  [ref] hf post-rope q vs hf PRE-rope q: cosine={cos(hf_qr, hf_q):9.6f}")

    # RoPE 是否 norm-preserving? 打印每个 token 的 q 行 norm 前后
    print("  per-token q row-norm (mc post-rope vs hf post-rope):")
    for t in range(SEQ):
        a = hf_qr[0, 0, t].float().norm().item()
        b = mc_q[0, 0, t].float().norm().item()
        print(f"    t{t}: hf={a:8.3f}  mc={b:8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
