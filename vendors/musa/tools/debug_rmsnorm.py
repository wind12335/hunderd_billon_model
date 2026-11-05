#!/usr/bin/env python3
"""用真实权重+真实输入，判定 MUSA torch.rms_norm 是否忠实于数学公式。

关键区别：此前用 weight=ones 测过(单位元掩盖了权重处理 bug)，这里用真实 LLaMA 权重。
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
FFN = 13824
NLAYERS = 40
NHEADS = 40
EPS = 1e-6


def manual_rms(x: torch.Tensor, w: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """数学公式参考（fp32）。"""
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    out = xf * torch.rsqrt(var + eps)
    return out * w.float()


def cos(a, b):
    return F.cosine_similarity(a.float().double(), b.float().double(), dim=-1).mean().item()


def main() -> int:
    from transformers import LlamaForCausalLM

    hf = LlamaForCausalLM.from_pretrained(HF_DIR, torch_dtype=torch.bfloat16).eval()

    dist.init_process_group("gloo", rank=0, world_size=1, init_method="tcp://127.0.0.1:29998")
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(0)

    config = TransformerConfig(
        num_layers=NLAYERS, hidden_size=HIDDEN, ffn_hidden_size=FFN,
        num_attention_heads=NHEADS, num_query_groups=NHEADS,
        normalization="RMSNorm", layernorm_epsilon=EPS,
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

    mc0 = model.decoder.layers[0]
    captured: dict = {}

    def cap_ln(name):
        def hook(_m, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            captured[name + "_in"] = inp[0].detach()
            captured[name + "_out"] = o.detach()
        return hook

    mc0.input_layernorm.register_forward_hook(cap_ln("mc_ln"))
    mc0.self_attention.linear_qkv.register_forward_hook(lambda m, i, o: None)  # noqa

    with torch.no_grad():
        ids = input_ids.to("musa:0")
        pos = torch.arange(SEQ, dtype=torch.long, device="musa:0").unsqueeze(0)
        mask = torch.tril(torch.ones(SEQ, SEQ, dtype=torch.bool, device="musa:0")).unsqueeze(0).unsqueeze(0)
        model(input_ids=ids, position_ids=pos, attention_mask=mask)

    # ---- module types ----
    print("== module types ==")
    print("  mc0.input_layernorm:", type(mc0.input_layernorm).__module__ + "." + type(mc0.input_layernorm).__name__)
    lnm = getattr(mc0.input_layernorm, "norm_impl", None)
    print("  mc0.input_layernorm.norm_impl:", lnm)

    # ---- faithful test: MUSA module vs manual formula ----
    x_mc = captured["mc_ln_in"].cpu()
    w_mc = mc0.input_layernorm.weight.detach().cpu()
    y_mc = captured["mc_ln_out"].cpu()
    ref_mc = manual_rms(x_mc, w_mc)
    print("== MUSA input_layernorm faithful? ==")
    print(f"  module vs manual  : cosine={cos(y_mc, ref_mc):.6f}  norm_ratio={ref_mc.float().norm()/y_mc.float().norm():.6f}")
    print(f"  weight range      : min={w_mc.float().min():.6f} max={w_mc.float().max():.6f} mean={w_mc.float().mean():.6f}")
    print(f"  weight==1 frac    : {(w_mc.float()==1.0).float().mean().item()*100:.1f}%")

    # ---- same test for HF module ----
    hf0 = hf.model.layers[0]
    with torch.no_grad():
        hf_out = hf(input_ids=input_ids, output_hidden_states=True)
    hf_in = hf_out.hidden_states[0].detach()
    hf_w = hf0.input_layernorm.weight.detach()
    hf_y = hf0.input_layernorm(hf_in).detach()
    ref_hf = manual_rms(hf_in, hf_w)
    print("== HF input_layernorm faithful? ==")
    print(f"  module vs manual  : cosine={cos(hf_y, ref_hf):.6f}")

    # ---- embedding: direct lookup vs what layer0 actually receives ----
    mc_word = model.embedding.word_embeddings(ids).cpu()
    print("== embedding direct vs layer-0 input ==")
    print(f"  word_emb (HF h0 vs MC direct): cosine={cos(hf_in, mc_word):.6f}")
    print(f"  word_emb (MC direct vs hook-in): cosine={cos(x_mc, mc_word):.6f}")
    # permutation check: 排序后坐标值是否一致（一致 → 纯坐标重排）
    a = hf_in[0, 0].float().sort()[0]
    b = x_mc[0, 0].float().sort()[0]
    print(f"  sorted-value cosine (permutation?): cosine={cos(a.unsqueeze(0), b.unsqueeze(0)):.6f}")
    print(f"  hf_in[0,0,:8]={hf_in[0,0,:8].tolist()}")
    print(f"  x_mc[0,0,:8]={x_mc[0,0,:8].tolist()}")

    # ---- do the two models' inputs/weights match? ----
    print("== cross-model: inputs / weights ==")
    print(f"  ln input   (HF vs MC) : cosine={cos(hf_in, x_mc):.6f}  norm_ratio={x_mc.float().norm()/hf_in.float().norm():.6f}")
    print(f"  ln weight  (HF vs MC) : cosine={cos(hf_w, w_mc):.6f}  max_abs_diff={(hf_w.float()-w_mc.float()).abs().max().item():.8f}")
    print(f"  ln output  (HF vs MC) : cosine={cos(hf_y, y_mc):.6f}")

    # ---- softmax sanity ----
    print("== MUSA softmax sanity ==")
    z = torch.randn(64, 64)
    ref_sm = torch.softmax(z, -1)
    mc_sm = torch.softmax(z.to("musa:0"), -1).cpu()
    print(f"  softmax cosine={cos(ref_sm, mc_sm):.6f}  rowsum(hf)={ref_sm.sum(-1)[:3].mean().item():.4f}  rowsum(mc)={mc_sm.sum(-1)[:3].mean().item():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
