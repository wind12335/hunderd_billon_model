#!/usr/bin/env python3
"""HF colossal-llama2-13B vs mcore(转换后) 前向逐层等价测试。

定位权重转换/前向配置 bug 的决定性工具：同一句输入跑两边 forward，
逐层对比 hidden states（embedding → layer0/1/20/39 → final norm → logits）。
哪一层开始 cos 崩塌，bug 就在哪一层。

用法（单卡，需 ~26GB 空闲）：
  python3 tools/verify_forward_equiv.py [--model-dir /private/model/llama2-13b]
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

_SELF_DIR = Path(__file__).resolve().parent
if str(_SELF_DIR) not in sys.path:
    sys.path.insert(0, str(_SELF_DIR))

from megatron_corex_bootstrap import bootstrap  # noqa: E402

bootstrap(verbose=False)

import glob  # noqa: E402
import os  # noqa: E402

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from convert_hf_to_mcore import add_extra_state, build_full_state_dict  # noqa: E402

PROBE_LAYERS = [0, 1, 20, 39]
TEXT = "Human: 你好，请简单介绍一下大语言模型。Assistant: 大语言模型是一种"


def capture_hf(model_dir: str, input_ids: list[int]) -> dict[str, torch.Tensor]:
    """HF 原模型 forward，抓 embedding/各层输出/logits。"""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, device_map="cuda:0"
    )
    model.eval()
    caps: dict[str, torch.Tensor] = {}
    hooks = []

    embs = model.get_input_embeddings()
    hooks.append(embs.register_forward_hook(
        lambda m, i, o: caps.__setitem__("embedding", o.detach().float().cpu())))

    for i in PROBE_LAYERS:
        layer = model.model.layers[i]
        hooks.append(layer.register_forward_hook(
            (lambda idx: lambda m, inp, o: caps.__setitem__(
                f"layer{idx}", o[0].detach().float().squeeze(0).cpu()))(i)))

    # layer0 层内子模块探针
    l0 = model.model.layers[0]
    hooks.append(l0.input_layernorm.register_forward_hook(
        lambda m, i_, o: caps.__setitem__("L0.norm1", o.detach().float().cpu())))
    hooks.append(l0.self_attn.register_forward_hook(
        lambda m, i_, o: caps.__setitem__("L0.attn", o[0].detach().float().cpu())))
    hooks.append(l0.post_attention_layernorm.register_forward_hook(
        lambda m, i_, o: caps.__setitem__("L0.norm2", o.detach().float().cpu())))
    hooks.append(l0.mlp.register_forward_hook(
        lambda m, i_, o: caps.__setitem__("L0.mlp", o.detach().float().cpu())))

    norm = model.model.norm
    hooks.append(norm.register_forward_hook(
        lambda m, i, o: caps.__setitem__("final_norm", o.detach().float().cpu())))

    ids = torch.tensor([input_ids], device="cuda:0")
    with torch.no_grad():
        logits = model(ids).logits
    caps["logits"] = logits.detach().float().cpu()
    for h in hooks:
        h.remove()
    del model, logits
    gc.collect()
    torch.cuda.empty_cache()
    return caps


def capture_mcore(model_dir: str, input_ids: list[int]) -> dict[str, torch.Tensor]:
    """转换权重加载进 mcore GPTModel，抓同样的阶段。"""
    from megatron.core import parallel_state
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer import TransformerConfig

    dist.init_process_group("gloo", rank=0, world_size=1,
                            init_method="tcp://127.0.0.1:29997")
    parallel_state.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(0)

    config = TransformerConfig(
        num_layers=40, hidden_size=5120, ffn_hidden_size=13824,
        num_attention_heads=40, num_query_groups=40,
        normalization="RMSNorm", layernorm_epsilon=1e-6,
        gated_linear_unit=True, activation_func=torch.nn.functional.silu,
        add_bias_linear=False, params_dtype=torch.bfloat16,
    )
    if not hasattr(config, "offload_moe_fc1_input"):
        config.offload_moe_fc1_input = False
    model = GPTModel(
        config, transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=69104, max_sequence_length=4096,
        position_embedding_type="rope", rotary_base=10000,
    )

    # 转换权重（与 convert_hf_to_mcore.py 主流程一致）
    hf: dict[str, torch.Tensor] = {}
    for f in sorted(glob.glob(os.path.join(model_dir, "pytorch_model-*.bin"))):
        hf.update(torch.load(f, map_location="cpu", weights_only=True))
    full = build_full_state_dict(hf, 40, 40)
    del hf
    full = {k: v.to(torch.bfloat16) for k, v in full.items()}
    add_extra_state(full, 40)
    missing, unexpected = model.load_state_dict(full, strict=False)
    print(f"[mcore] load: missing={list(missing)[:4]} unexpected={list(unexpected)[:4]}")
    del full
    gc.collect()

    model.cuda().to(torch.bfloat16).eval()
    caps: dict[str, torch.Tensor] = {}
    hooks = []
    hooks.append(model.embedding.word_embeddings.register_forward_hook(
        lambda m, i, o: caps.__setitem__("embedding", o.detach().float().cpu())))
    # layer0 层内子模块探针（mcore 命名）
    ml0 = model.decoder.layers[0]
    hooks.append(ml0.input_layernorm.register_forward_hook(
        lambda m, i_, o: caps.__setitem__("L0.norm1", o.detach().float().cpu())))
    hooks.append(ml0.self_attention.register_forward_hook(
        lambda m, i_, o: caps.__setitem__("L0.attn", o[0].detach().float().cpu())))
    hooks.append(ml0.pre_mlp_layernorm.register_forward_hook(
        lambda m, i_, o: caps.__setitem__("L0.norm2", o.detach().float().cpu())))
    hooks.append(ml0.mlp.register_forward_hook(
        lambda m, i_, o: caps.__setitem__(
            "L0.mlp", (o[0] if isinstance(o, tuple) else o).detach().float().cpu())))
    for i in PROBE_LAYERS:
        layer = model.decoder.layers[i]
        hooks.append(layer.register_forward_hook(
            (lambda idx: lambda m, inp, o: caps.__setitem__(
                f"layer{idx}", o[0].detach().float().cpu()))(i)))
    hooks.append(model.decoder.final_layernorm.register_forward_hook(
        lambda m, i, o: caps.__setitem__("final_norm", o.detach().float().cpu())))

    device = torch.device("cuda:0")
    s = len(input_ids)
    ids = torch.tensor([input_ids], device=device)
    pos = torch.arange(s, device=device).unsqueeze(0)
    # mcore 语义: attention_mask 为 True 的位置被 masked_fill -10000（True=屏蔽）。
    # 因果 mask ⇒ 上三角（j>i）为 True，与直觉的 tril 可attend 相反。
    mask = torch.triu(
        torch.ones(s, s, dtype=torch.bool, device=device), diagonal=1
    ).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        logits = model(input_ids=ids, position_ids=pos, attention_mask=mask)
    caps["logits"] = logits.detach().float().cpu()
    for h in hooks:
        h.remove()
    return caps


def compare(a: dict, b: dict) -> None:
    print(f"\n{'stage':<12} {'shape':<20} {'maxdiff':>10} {'cos':>10} 判定")
    for k in ["embedding", "L0.norm1", "L0.attn", "L0.norm2", "L0.mlp", "layer0"] + [f"layer{i}" for i in PROBE_LAYERS[1:]] + ["final_norm", "logits"]:
        if k not in a or k not in b:
            print(f"{k:<12} 缺失")
            continue
        x, y = a[k], b[k]
        # HF [b,s,h] / mcore [s,b,h] → 统一收平成 [token, feature]
        x = x.reshape(-1, x.shape[-1])
        y = y.reshape(-1, y.shape[-1])
        if x.shape != y.shape:
            print(f"{k:<12} shape 不一致 {tuple(x.shape)} vs {tuple(y.shape)}")
            continue
        cos = torch.nn.functional.cosine_similarity(
            x.flatten(), y.flatten(), dim=0).item()
        md = (x - y).abs().max().item()
        ok = "✓" if cos > 0.98 else "✗ 崩塌"
        print(f"{k:<12} {str(tuple(x.shape)):<20} {md:>10.4f} {cos:>10.6f} {ok}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/private/model/llama2-13b")
    ap.add_argument("--text", default=TEXT)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    input_ids = tok(args.text)["input_ids"]
    print(f"[test] tokens={len(input_ids)} ids={input_ids[:12]}")

    hf_caps = capture_hf(args.model_dir, input_ids)
    print("[test] HF 侧捕获完成，卸载后加载 mcore 侧 ...")
    mc_caps = capture_mcore(args.model_dir, input_ids)
    compare(hf_caps, mc_caps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
