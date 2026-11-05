#!/usr/bin/env python3
"""定位 core_attention 内部发散: 截取真实 q/k/v/mask,逐步重算对比 CPU 参考。

步骤: scores=bmm(Q,K^T)/√d → scale_mask_softmax → context=bmm(probs,V)
"""
from __future__ import annotations

import sys
from pathlib import Path

_SELF_DIR = Path(__file__).resolve().parent
if str(_SELF_DIR) not in sys.path:
    sys.path.insert(0, str(_SELF_DIR))

from megatron_musa_bootstrap import bootstrap  # noqa: E402

bootstrap(verbose=False)

import math  # noqa: E402
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
FFN = 13824
NLAYERS = 40
NHEADS = 40


def cos(a, b):
    a, b = a.cpu().float(), b.cpu().float()
    return F.cosine_similarity(a.double(), b.double(), dim=-1).mean().item()


def main() -> int:
    from transformers import LlamaForCausalLM

    hf = LlamaForCausalLM.from_pretrained(HF_DIR, torch_dtype=torch.bfloat16).eval()

    dist.init_process_group("gloo", rank=0, world_size=1, init_method="tcp://127.0.0.1:29993")
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(0)

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
    model.cuda().to(torch.bfloat16).eval()
    model.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True)["model"], strict=True)

    torch.manual_seed(1234)
    input_ids = torch.randint(0, VOCAB, (1, SEQ), dtype=torch.long)

    cap: dict = {}

    def cap_ca(_m, inp, out):
        cap["q"], cap["k"], cap["v"] = inp[0].detach(), inp[1].detach(), inp[2].detach()
        cap["mask"] = inp[3].detach() if len(inp) > 3 and inp[3] is not None else None
        cap["out"] = (out[0] if isinstance(out, tuple) else out).detach()

    mc0 = model.decoder.layers[0]
    mc0.self_attention.core_attention.register_forward_hook(cap_ca)

    with torch.no_grad():
        ids = input_ids.to("musa:0")
        pos = torch.arange(SEQ, dtype=torch.long, device="musa:0").unsqueeze(0)
        mask = torch.tril(torch.ones(SEQ, SEQ, dtype=torch.bool, device="musa:0")).unsqueeze(0).unsqueeze(0)
        model(input_ids=ids, position_ids=pos, attention_mask=mask)

    q, k, v = cap["q"], cap["k"], cap["v"]
    print(f"  q shape={tuple(q.shape)}  k={tuple(k.shape)}  v={tuple(v.shape)}  mask={None if cap['mask'] is None else tuple(cap['mask'].shape)}")
    # core_attention input 是 [sq, b, np, hn]
    sq, b, np_, hn = q.shape
    d = hn
    scale = 1.0 / math.sqrt(d)

    # --- MUSA 逐步重算 ---
    qf = q.reshape(sq, b * np_, -1)          # [sq, b*np, hn]
    kf = k.reshape(sq, b * np_, -1)          # [sk, b*np, hn]
    scores_mc = torch.bmm(qf.transpose(0, 1), kf.transpose(0, 1).transpose(1, 2)) * scale  # [b*np, sq, sk]
    scores_mc = scores_mc.view(b, np_, sq, sq)

    # 应用 mask (bool -> -inf)
    m = cap["mask"]  # [1,1,s,s] bool
    if m is not None:
        mf = m.float()
        neg = mf.where(mf.bool(), float("-inf"))
        scores_masked = scores_mc + neg
    else:
        scores_masked = scores_mc
    probs_mc = torch.softmax(scores_masked, dim=-1)
    print(f"  dtype trace: q={q.dtype} k={k.dtype} v={v.dtype} scores={scores_mc.dtype} scores_masked={scores_masked.dtype} probs={probs_mc.dtype}")
    context_mc = torch.bmm(
        probs_mc.to(torch.bfloat16).reshape(b * np_, sq, sq),
        v.reshape(sq, b * np_, -1).transpose(0, 1),
    ).view(sq, b, np_, hn)

    # --- 模型实际输出 ---
    print(f"  model core_attn out: norm={cap['out'].float().norm().item():.3f}")
    print(f"  our recompute     : norm={context_mc.float().norm().item():.3f}")
    print(f"  recompute vs model: cosine={cos(context_mc.reshape(sq, b, -1), cap['out']):.6f}")

    # --- CPU fp32 参考 ---
    qc, kc, vc = q.cpu().double(), k.cpu().double(), v.cpu().double()
    scores_cpu = torch.bmm(qc.reshape(sq, b * np_, -1).transpose(0, 1), kc.reshape(sq, b * np_, -1).transpose(0, 1).transpose(1, 2)) * scale
    scores_cpu = scores_cpu.view(b, np_, sq, sq)
    mask_cpu = m.cpu().float().double()
    scores_masked_cpu = scores_cpu + mask_cpu.where(mask_cpu.bool(), float("-inf"))
    probs_cpu = torch.softmax(scores_masked_cpu, dim=-1)
    context_cpu = torch.bmm(
        probs_cpu.reshape(b * np_, sq, sq),
        vc.reshape(sq, b * np_, -1).transpose(0, 1),
    ).view(sq, b, np_, hn)

    print(f"  [MUSA vs CPU] scores   : cosine={cos(scores_mc, scores_cpu):.6f}  max_abs={(scores_mc.cpu().double() - scores_cpu).abs().max().item():.6f}")
    print(f"  [MUSA vs CPU] probs    : cosine={cos(probs_mc, probs_cpu):.6f}  rowsum={probs_mc.cpu().float().sum(-1)[0,0,0].item():.4f}")
    print(f"  [MUSA vs CPU] context  : cosine={cos(context_mc, context_cpu):.6f}  norm_mc={context_mc.float().norm().item():.3f} norm_cpu={context_cpu.float().norm().item():.3f}")

    # --- 直接测 FusedScaleMaskSoftmax ---
    sms = mc0.self_attention.core_attention.scale_mask_softmax
    print("  scale_mask_softmax class:", type(sms).__module__ + "." + type(sms).__name__)
    probs_sms = sms(scores_mc.double().float(), cap["mask"]).double()
    print(f"  [SMS vs CPU probs] cosine={cos(probs_sms, probs_cpu):.6f}  norm={probs_sms.float().norm().item():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
