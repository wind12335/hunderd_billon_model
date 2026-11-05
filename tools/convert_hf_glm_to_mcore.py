#!/usr/bin/env python3
"""HF GLM-4.5-Air -> mcore torch_dist 权重转换（开发机 CPU 单进程，无 GPU）。

路线（最高保真，零手写分片）:
  1. 在 CPU 上按集群同一条构建链（core_transformer_config_from_args +
     get_gpt_decoder_block_spec(local) + GPTModel）建全尺寸模型（bf16）。
  2. 逐层增量 load_state_dict(assign=True) 灌入 HF 权重（控峰值内存 ~215G）。
  3. 用 mcore 自己的 sharded_state_dict() + dist_checkpointing.save 落盘——
     SwiGLU fc1 工厂、专家维分片、extra_state 全由 mcore 原生生成，
     训练侧 --load 后自动 reshard 到 TP8/PP2/EP8。
  4. 用 tools/checkpoint/dist_checkpoint_io.load_dist_checkpoint_full 回读
     全量张量，做覆盖清单 + 逐位抽查（QKV 交错切片回拼 vs HF 原始）。

结构决策（与 render 旗标一一对应，改前先想清楚）:
  * --disable-bias-linear + --add-qkv-bias: QKV 真偏置、其余层无偏置，
    与 HF 逐位一致（弃用了早前的"零偏置"方案）。
  * --moe-router-score-function sigmoid + --moe-router-enable-expert-bias:
    DeepSeek-V3 式路由，expert_bias 存 fp32（mcore 惯例）。
  * --rotary-percent 0.5、--transformer-impl local（键名确定，不依赖 TE）。
  * MTP 层 46 整层跳过（HF 的 model.layers.46.*）。
  * QKV 交错: 每组 [q(hpg 头), k(1 头), v(1 头)] 顺序（镜像
    3rdparty tools/checkpoint/loader_mixtral_hf.py:87-99 的官方 reshape+cat）。
  * fc1 = cat([gate_proj, up_proj], dim=0)（稠密/共享/路由专家同律）。
  * 尺寸从 HF 张量形状推导（单一事实来源），mcore 旗标随之生成。

用法:
  冒烟(2 层真数据全链路): python3 -I tools/convert_hf_glm_to_mcore.py --smoke
  正式(46 层):  nohup python3 -I tools/convert_hf_glm_to_mcore.py \
      --hf-dir models/glm45_air_base --out-dir models/glm45_air_base_mcore &
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(1, str(REPO / "Megatron-LM-das" / "3rdparty" / "Megatron-LM"))

import sitecustomize  # noqa: F401  typing.override shim (py3.11)
import warnings

warnings.filterwarnings("ignore")

import os

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29574")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")

import torch
from safetensors import safe_open

from mcore_cpu_build import build_glm_model


def hf_prefix(k: str) -> str:
    """HF safetensors 键去掉 shard 前缀（有的导出带 'model.'，lm_head 除外）。"""
    return k


class HfStore:
    """按需打开 shard 的懒加载句柄（避免 206G 一次性进内存）。"""

    def __init__(self, hf_dir: Path):
        index = json.loads((hf_dir / "model.safetensors.index.json").read_text())
        self.weight_map: dict[str, str] = index["weight_map"]
        self.hf_dir = hf_dir
        self._handles: dict[str, object] = {}
        # 全量键清单（覆盖校验用）
        self.all_keys: set[str] = set(self.weight_map)

    def get(self, key: str) -> torch.Tensor:
        shard = self.weight_map[key]
        h = self._handles.get(shard)
        if h is None:
            h = safe_open(self.hf_dir / shard, framework="pt")
            self._handles[shard] = h
        return h.get_tensor(key)


def interleave_qkv(wq, wk, wv, groups: int, hpg: int, dim: int, is_bias: bool):
    """官方 GQA 交错: (G, hpg*dim, H)+(G, dim, H)*2 -> 每组 [q,k,v] 拼接。"""
    if is_bias:
        q, k, v = wq.view(groups, hpg * dim), wk.view(groups, dim), wv.view(groups, dim)
        return torch.cat([q, k, v], dim=1).reshape(-1)
    q, k, v = wq.view(groups, hpg * dim, -1), wk.view(groups, dim, -1), wv.view(groups, dim, -1)
    return torch.cat([q, k, v], dim=1).reshape(groups * (hpg + 2) * dim, wq.shape[1])


def derive_dims(store: HfStore, layers: int) -> dict:
    """从 HF 张量形状推导全部结构尺寸（不依赖 config.json 字段名猜测）。"""
    q = store.get("model.layers.3.self_attn.q_proj.weight")
    k = store.get("model.layers.3.self_attn.k_proj.weight")
    hidden = q.shape[1]
    kv_dim = k.shape[0]
    groups = kv_dim // 128 if kv_dim % 128 == 0 else 1
    # head_dim=128、G=kv 头数 是 GLM-4.5 的事实；这里再做一致性断言
    assert kv_dim % groups == 0
    dim = kv_dim // groups
    hpg = q.shape[0] // (groups * dim)
    moe_first = "model.layers.0.mlp.gate.weight" in store.all_keys
    dense_idx = [0] if not moe_first else []
    n_experts = 1 + max(
        int(k_.split(".")[5]) for k_ in store.all_keys
        if k_.startswith("model.layers.3.mlp.experts.") and k_.endswith("gate_proj.weight")
    )
    ffn_dense = store.get("model.layers.0.mlp.gate_proj.weight").shape[0]
    ffn_moe = store.get("model.layers.3.mlp.experts.0.gate_proj.weight").shape[0]
    vocab = store.get("model.embed_tokens.weight").shape[0]
    return dict(
        hidden=hidden, groups=groups, dim=dim, hpg=hpg,
        n_experts=n_experts, ffn_dense=ffn_dense, ffn_moe=ffn_moe,
        vocab=vocab, moe_first=moe_first,
        moe_layer_freq=f"[{0 if moe_first else 1}]*{len(dense_idx)}+"
                       f"[{1 if moe_first else 0}]*{layers - len(dense_idx)}",
    )


def build_chunk(store: HfStore, d: dict, i: int, hpg: int):
    """第 i 层的 {mcore键: HF张量} 映射。"""
    g = d["groups"]
    p = f"model.layers.{i}."
    mp = f"decoder.layers.{i}."
    out: dict[str, torch.Tensor] = {}
    out[mp + "input_layernorm.weight"] = store.get(p + "input_layernorm.weight")
    out[mp + "pre_mlp_layernorm.weight"] = store.get(p + "post_attention_layernorm.weight")
    wq = store.get(p + "self_attn.q_proj.weight")
    wk = store.get(p + "self_attn.k_proj.weight")
    wv = store.get(p + "self_attn.v_proj.weight")
    out[mp + "self_attention.linear_qkv.weight"] = interleave_qkv(wq, wk, wv, g, hpg, d["dim"], False)
    bq = store.get(p + "self_attn.q_proj.bias")
    bk = store.get(p + "self_attn.k_proj.bias")
    bv = store.get(p + "self_attn.v_proj.bias")
    out[mp + "self_attention.linear_qkv.bias"] = interleave_qkv(bq, bk, bv, g, hpg, d["dim"], True)
    out[mp + "self_attention.linear_proj.weight"] = store.get(p + "self_attn.o_proj.weight")
    is_moe = (i > 0) if not d["moe_first"] else True
    if not is_moe:
        out[mp + "mlp.linear_fc1.weight"] = torch.cat(
            [store.get(p + "mlp.gate_proj.weight"), store.get(p + "mlp.up_proj.weight")], dim=0)
        out[mp + "mlp.linear_fc2.weight"] = store.get(p + "mlp.down_proj.weight")
        return out
    out[mp + "mlp.router.weight"] = store.get(p + "mlp.gate.weight")
    out[mp + "mlp.router.expert_bias"] = store.get(p + "mlp.gate.e_score_correction_bias").float()
    for e in range(d["n_experts"]):
        ep = f"{p}mlp.experts.{e}."
        out[f"{mp}mlp.experts.local_experts.{e}.linear_fc1.weight"] = torch.cat(
            [store.get(ep + "gate_proj.weight"), store.get(ep + "up_proj.weight")], dim=0)
        out[f"{mp}mlp.experts.local_experts.{e}.linear_fc2.weight"] = store.get(ep + "down_proj.weight")
    sp = f"{p}mlp.shared_expert."
    out[mp + "mlp.shared_experts.linear_fc1.weight"] = torch.cat(
        [store.get(sp + "gate_proj.weight"), store.get(sp + "up_proj.weight")], dim=0)
    out[mp + "mlp.shared_experts.linear_fc2.weight"] = store.get(sp + "down_proj.weight")
    return out


def assign_chunk(model, chunk: dict, consumed: set[str], hf_note: dict) -> None:
    """assign=True 增量灌入；键必须逐个命中模型 state_dict（除 extra_state 外）。"""
    sd = model.state_dict()
    for k, t in chunk.items():
        assert k in sd, f"模型无此键: {k}"
        target = sd[k]
        assert torch.is_tensor(target), f"目标非张量: {k}"
        assert tuple(target.shape) == tuple(t.shape), \
            f"{k} 形状不符: 模型{tuple(target.shape)} vs 转换{tuple(t.shape)}"
        if t.dtype != target.dtype:
            t = t.to(target.dtype)
        chunk[k] = t
        hf_note[k] = str(tuple(t.shape))
    missing, unexpected = model.load_state_dict(chunk, strict=False, assign=True)
    assert not unexpected, f"意外键: {unexpected}"
    assert all(m.endswith("_extra_state") for m in missing), f"缺失键: {missing}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hf-dir", default="models/glm45_air_base")
    ap.add_argument("--out-dir", default="models/glm45_air_base_mcore")
    ap.add_argument("--layers", type=int, default=46)
    ap.add_argument("--smoke", action="store_true", help="2 层真数据全链路（含保存/校验）")
    ap.add_argument("--verify-count", type=int, default=12)
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.layers, args.out_dir = 2, "generated/conv_smoke"

    hf_dir = REPO / args.hf_dir
    out_dir = REPO / args.out_dir
    t0 = time.time()
    store = HfStore(hf_dir)
    d = derive_dims(store, args.layers)
    print(f"[conv] 结构推导: {d}  ({time.time()-t0:.0f}s)")

    # 1) 建模型（集群同链路，bf16）
    model, margs = build_glm_model(
        num_layers=args.layers, hidden=d["hidden"], heads=d["groups"] * d["hpg"],
        groups=d["groups"], kv_channels=d["dim"], ffn_dense=d["ffn_dense"],
        moe_ffn=d["ffn_moe"], shared_ffn=d["ffn_moe"], experts=d["n_experts"],
        vocab=d["vocab"], moe_layer_freq=d["moe_layer_freq"], bf16=True,
    )
    print(f"[conv] 模型构建完成 ({time.time()-t0:.0f}s)")

    # 2) 逐层灌入（含 embedding/output/final_norm 首块）
    consumed: set[str] = set()
    hf_note: dict[str, str] = {}
    head = OrderedDict()
    head["embedding.word_embeddings.weight"] = store.get("model.embed_tokens.weight")
    head["decoder.final_layernorm.weight"] = store.get("model.norm.weight")
    head["output_layer.weight"] = store.get("lm_head.weight")
    assign_chunk(model, head, consumed, hf_note)
    consumed |= {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
    for i in range(args.layers):
        assign_chunk(model, build_chunk(store, d, i, d["hpg"]), consumed, hf_note)
        if i % 8 == 0 or i == args.layers - 1:
            print(f"[conv] 层 {i} 完成 ({time.time()-t0:.0f}s)")
        gc.collect()

    # 3) HF 键覆盖清单：全部键 = 已消费 + MTP 层46（跳过）
    consumed |= {
        k for k in store.all_keys
        for _ in [0]
        if (k.startswith("model.layers.") and int(k.split(".")[2]) < args.layers
            and not k.startswith("model.norm") and k != "lm_head.weight"
            and k != "model.embed_tokens.weight")
    }
    # 精确重构：按 build_chunk 的取用清单逐一登记
    consumed = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
    for i in range(args.layers):
        p = f"model.layers.{i}."
        consumed |= {p + "input_layernorm.weight", p + "post_attention_layernorm.weight",
                     p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight",
                     p + "self_attn.v_proj.weight",
                     p + "self_attn.q_proj.bias", p + "self_attn.k_proj.bias",
                     p + "self_attn.v_proj.bias", p + "self_attn.o_proj.weight"}
        is_moe = (i > 0) if not d["moe_first"] else True
        if not is_moe:
            consumed |= {p + k for k in ("mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")}
        else:
            consumed |= {p + "mlp.gate.weight", p + "mlp.gate.e_score_correction_bias"}
            for e in range(d["n_experts"]):
                consumed |= {f"{p}mlp.experts.{e}.{n}" for n in ("gate_proj.weight", "up_proj.weight", "down_proj.weight")}
            consumed |= {f"{p}mlp.shared_expert.{n}" for n in ("gate_proj.weight", "up_proj.weight", "down_proj.weight")}
    leftover = store.all_keys - consumed
    skipped_mtp = {k for k in leftover if k.startswith("model.layers.46.") or ".shared_head" in k or "mtp" in k.lower()}
    unexpected_left = leftover - skipped_mtp
    assert not unexpected_left, f"未消费的 HF 键: {sorted(unexpected_left)[:10]} 共{len(unexpected_left)}"
    print(f"[conv] 覆盖校验: 消费 {len(consumed)}，跳过 MTP {len(skipped_mtp)}，无遗漏")

    # 4) mcore 原生保存
    from megatron.core import dist_checkpointing
    from megatron.core.dist_checkpointing.core import CheckpointingConfig

    iter_dir = out_dir / "iter_0000000"
    dist_checkpointing.save(
        model.sharded_state_dict(),
        str(iter_dir),
        sharded_strategy=None,  # 默认 torch_dist
        validate_access_integrity=True,
    )
    (out_dir / "latest_checkpointed_iteration.txt").write_text("0")
    manifest = {
        "source": str(hf_dir), "layers": args.layers, "derived": {k: str(v) for k, v in d.items()},
        "hf_keys_consumed": len(consumed), "hf_keys_skipped_mtp": sorted(skipped_mtp)[:8],
        "elapsed_build_assign_save_s": round(time.time() - t0, 1),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "conversion_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"[conv] 保存完成: {iter_dir} ({time.time()-t0:.0f}s)")

    if args.no_verify:
        return
    del model, head
    gc.collect()
    verify(store, d, iter_dir, args.layers, args.verify_count)


def verify(store: HfStore, d: dict, iter_dir: Path, layers: int, count: int) -> None:
    """回读全量张量，逐位抽查 + 交错切片回拼。"""
    sys.path.insert(0, str(REPO / "Megatron-LM-das" / "3rdparty" / "Megatron-LM" / "tools" / "checkpoint"))
    from dist_checkpoint_io import load_dist_checkpoint_full

    full, _, _, backend, _ = load_dist_checkpoint_full(str(iter_dir))
    print(f"[verify] 回读 {len(full)} 个张量, backend={backend}")

    def eq(name, a, b):
        assert torch.equal(a.contiguous(), b.contiguous()), f"不一致: {name}\n{a.flatten()[:8]}\n{b.flatten()[:8]}"
        print(f"[verify] OK {name}")

    eq("embedding[0:256]", full["embedding.word_embeddings.weight"][:256],
       store.get("model.embed_tokens.weight")[:256])
    eq("final_layernorm", full["decoder.final_layernorm.weight"].float(),
       store.get("model.norm.weight").float())
    eq("output_layer[:256]", full["output_layer.weight"][:256], store.get("lm_head.weight")[:256])
    # 第 1 层(MoE)与第 0 层(稠密)都抽
    for i in sorted({0, min(1, layers - 1), layers - 1}):
        p = f"model.layers.{i}."
        mp = f"decoder.layers.{i}."
        w = full[mp + "self_attention.linear_qkv.weight"].float()
        g, hpg, dim = d["groups"], d["hpg"], d["dim"]
        wq = w.view(g, (hpg + 2) * dim, -1)[:, : hpg * dim].reshape(-1, w.shape[1])
        wk = w.view(g, (hpg + 2) * dim, -1)[:, hpg * dim : hpg * dim + dim].reshape(-1, w.shape[1])
        wv = w.view(g, (hpg + 2) * dim, -1)[:, hpg * dim + dim :].reshape(-1, w.shape[1])
        eq(f"L{i}.qkv.wq", wq, store.get(p + "self_attn.q_proj.weight").float())
        eq(f"L{i}.qkv.wk", wk, store.get(p + "self_attn.k_proj.weight").float())
        eq(f"L{i}.qkv.wv", wv, store.get(p + "self_attn.v_proj.weight").float())
        b = full[mp + "self_attention.linear_qkv.bias"].float()
        bq = b.view(g, (hpg + 2) * dim)[:, : hpg * dim].reshape(-1)
        bk = b.view(g, (hpg + 2) * dim)[:, hpg * dim : hpg * dim + dim].reshape(-1)
        bv = b.view(g, (hpg + 2) * dim)[:, hpg * dim + dim :].reshape(-1)
        eq(f"L{i}.qkv.bq", bq, store.get(p + "self_attn.q_proj.bias").float())
        eq(f"L{i}.qkv.bk", bk, store.get(p + "self_attn.k_proj.bias").float())
        eq(f"L{i}.qkv.bv", bv, store.get(p + "self_attn.v_proj.bias").float())
        eq(f"L{i}.o_proj", full[mp + "self_attention.linear_proj.weight"],
           store.get(p + "self_attn.o_proj.weight"))
        eq(f"L{i}.in_ln", full[mp + "input_layernorm.weight"].float(),
           store.get(p + "input_layernorm.weight").float())
    # MoE 层: 路由 + 专家 + 共享
    if layers > 1:
        p, mp = "model.layers.1.", "decoder.layers.1."
        eq("L1.router", full[mp + "mlp.router.weight"].float(), store.get(p + "mlp.gate.weight").float())
        eb = full[mp + "mlp.router.expert_bias"]
        assert eb.dtype == torch.float32, f"expert_bias 应为 fp32, 实际 {eb.dtype}"
        eq("L1.expert_bias", eb, store.get(p + "mlp.gate.e_score_correction_bias").float())
        for e in (0, 5, d["n_experts"] - 1):
            # 回读键可能是全专家 3D 或逐专家 2D，两种都兼容
            f1 = full.get(f"{mp}mlp.experts.local_experts.{e}.linear_fc1.weight")
            if f1 is None or f1.dim() == 3:
                f1 = full[f"{mp}mlp.experts.linear_fc1.weight"][e] if f1 is None else f1[e]
            eq(f"L1.e{e}.fc1", f1, torch.cat([store.get(f"{p}mlp.experts.{e}.gate_proj.weight"),
                                              store.get(f"{p}mlp.experts.{e}.up_proj.weight")], dim=0))
        eq("L1.shared.fc2", full[mp + "mlp.shared_experts.linear_fc2.weight"],
           store.get(p + "mlp.shared_expert.down_proj.weight"))
    print("[verify] 全部抽查通过")


if __name__ == "__main__":
    main()
