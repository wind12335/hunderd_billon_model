#!/usr/bin/env python3
"""Layer-0 stepwise comparison: HF vs mcore (corrected for [s,b,h] vs [b,s,h]).

Megatron 内部用 [s b h],HF 用 [b s h]。所有 mc 输出先 transpose(0,1) 成 [b s h] 再比。
norm_ratio 是置换不变的,反映真实幅度;cosine 只在形状对齐后才有意义。
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


def cos(a, b):
    a, b = a.cpu(), b.cpu()
    return F.cosine_similarity(a.float().double(), b.float().double(), dim=-1).mean().item()


def norm(a):
    return a.cpu().float().norm().item()


def fmt(name, a, b):
    # a,b both [b,s,h]
    print(f"  {name:>18s} cosine={cos(a, b):9.6f}  norm_ratio(mc/hf)={norm(b) / norm(a):10.4f}  hf_norm={norm(a):11.3f} mc_norm={norm(b):11.3f}")


def main() -> int:
    from transformers import LlamaForCausalLM

    hf = LlamaForCausalLM.from_pretrained(HF_DIR, torch_dtype=torch.bfloat16).eval()

    dist.init_process_group("gloo", rank=0, world_size=1, init_method="tcp://127.0.0.1:29999")
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

    # ---- hooks: HF layer 0 ([b,s,h]) ----
    hf0 = hf.model.layers[0]
    hf_outs: dict = {}

    def hf_hook(name):
        def hook(_m, _inp, out):
            o = out[0] if isinstance(out, tuple) else out
            hf_outs[name] = o.detach()
        return hook

    hf0.input_layernorm.register_forward_hook(hf_hook("ln"))
    hf0.self_attn.q_proj.register_forward_hook(hf_hook("q"))
    hf0.self_attn.k_proj.register_forward_hook(hf_hook("k"))
    hf0.self_attn.v_proj.register_forward_hook(hf_hook("v"))
    hf0.self_attn.register_forward_hook(hf_hook("attn"))
    hf0.post_attention_layernorm.register_forward_hook(hf_hook("palm"))
    hf0.mlp.register_forward_hook(hf_hook("mlp"))
    hf0.register_forward_hook(hf_hook("layer"))

    # ---- hooks: mcore layer 0 (internally [s,b,h]) ----
    mc0 = model.decoder.layers[0]
    mc_outs: dict = {}

    def mc_hook(name):
        def hook(_m, _inp, out):
            o = out[0] if isinstance(out, tuple) else out
            mc_outs[name] = o.detach()
        return hook

    def cap_ln(_m, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        mc_outs["ln_in"] = inp[0].detach()
        mc_outs["ln_out"] = o.detach()

    mc0.input_layernorm.register_forward_hook(cap_ln)
    mc0.self_attention.linear_qkv.register_forward_hook(mc_hook("qkv"))
    mc0.self_attention.core_attention.register_forward_hook(mc_hook("core_attn"))
    mc0.self_attention.register_forward_hook(mc_hook("attn"))
    mc0.pre_mlp_layernorm.register_forward_hook(mc_hook("palm"))
    mc0.mlp.register_forward_hook(mc_hook("mlp"))
    mc0.register_forward_hook(mc_hook("layer"))

    with torch.no_grad():
        hf(input_ids=input_ids, output_hidden_states=True)
        ids = input_ids.to("musa:0")
        pos = torch.arange(SEQ, dtype=torch.long, device="musa:0").unsqueeze(0)
        mask = torch.tril(torch.ones(SEQ, SEQ, dtype=torch.bool, device="musa:0")).unsqueeze(0).unsqueeze(0)
        model(input_ids=ids, position_ids=pos, attention_mask=mask)

    # 转置 helper: mc 的 [s,b,h] -> [b,s,h]
    def mc(name):
        return mc_outs[name].transpose(0, 1).contiguous()

    print("== layer-0 stepwise (corrected [b,s,h] alignment) ==")
    hf_emb = hf.model.embed_tokens(input_ids).detach()
    mc_ln_in = mc_outs["ln_in"].transpose(0, 1).contiguous().cpu()
    print(f"  [check] HF embed vs MC ln-input : cosine={cos(hf_emb, mc_ln_in):.6f}  max_abs_diff={(hf_emb.float() - mc_ln_in.float()).abs().max().item():.6f}")
    fmt("layernorm_out", hf_outs["ln"], mc_outs["ln_out"].transpose(0, 1))
    # qkv: mc 的 [s,b,3h] -> [b,s,3h]
    qkv_mc = mc_outs["qkv"].transpose(0, 1).contiguous()
    qkv_hf = torch.cat([hf_outs["q"], hf_outs["k"], hf_outs["v"]], dim=-1)
    fmt("qkv(cat)", qkv_hf, qkv_mc)
    fmt("q block", hf_outs["q"], qkv_mc[:, :, 0:HIDDEN])
    fmt("k block", hf_outs["k"], qkv_mc[:, :, HIDDEN:2 * HIDDEN])
    fmt("v block", hf_outs["v"], qkv_mc[:, :, 2 * HIDDEN:3 * HIDDEN])
    print("  (mc qkv shape: %s, hf cat shape: %s)" % (tuple(qkv_mc.shape), tuple(qkv_hf.shape)))
    fmt("core_attn_out", hf_outs["attn"], mc("core_attn"))
    fmt("attn_out(o_proj)", hf_outs["attn"], mc("attn"))
    fmt("post_attn_ln", hf_outs["palm"], mc("palm"))
    fmt("mlp_out", hf_outs["mlp"], mc("mlp"))
    fmt("layer_out", hf_outs["layer"], mc("layer"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
