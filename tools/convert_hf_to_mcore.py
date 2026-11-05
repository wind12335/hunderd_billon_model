#!/usr/bin/env python3
"""HF（colossal-llama2-13B）→ mcore 本地 spec 权重转换器。

读取 HF pytorch_model-*.bin 全量权重，融合成 mcore GPTModel(local spec) 的
state_dict，并按 Megatron legacy checkpoint 目录格式写出，供
pretrain_gpt.py --load 直接使用（TP=1 时为单一 mp_rank_00 全量文件）。

mcore 本地 spec 的 key 结构已用 tools/dump_mcore_state_dict.py 枚举验证：
  embedding.word_embeddings.weight                 (V, H)    ← model.embed_tokens.weight
  decoder.layers.{i}.input_layernorm.weight        (H,)      ← model.layers.{i}.input_layernorm.weight
  decoder.layers.{i}.self_attention.linear_qkv.weight  (3H, H)  ← 按头交错 [q0;k0;v0; q1;k1;v1; ...]（见 attention.py view/split）
  decoder.layers.{i}.self_attention.linear_proj.weight (H, H)  ← model.layers.{i}.self_attn.o_proj.weight
  decoder.layers.{i}.pre_mlp_layernorm.weight      (H,)      ← post_attention_layernorm.weight
  decoder.layers.{i}.mlp.linear_fc1.weight         (2F, H)   ← cat([gate,up],0)，行序 [gate;up]（见 mlp.py glu 拆法）
  decoder.layers.{i}.mlp.linear_fc2.weight         (H, F)    ← down_proj.weight
  decoder.final_layernorm.weight                   (H,)      ← model.norm.weight
  output_layer.weight                              (V, H)    ← lm_head.weight（untied）

TP 分片规则（仅 --tensor-model-parallel-size>1 时启用）：
  ColumnParallelLinear / VocabParallelEmbedding：按 dim0 均匀切片
  RowParallelLinear：                            按 dim1 均匀切片
  LayerNorm：                                   各 rank 全量复制

用法:
  python3 tools/convert_hf_to_mcore.py \
      --model-dir /ws/model \
      --output-dir /ws/iluvatar_framework/generated/ckpt_13b \
      --tensor-model-parallel-size 1
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# 分片工具
# ---------------------------------------------------------------------------

def shard_column(full: torch.Tensor, rank: int, tp: int) -> torch.Tensor:
    """ColumnParallelLinear / VocabParallelEmbedding：按 dim0 切片。"""
    n = full.shape[0]
    assert n % tp == 0, f"dim0 {n} 不能被 TP={tp} 整除"
    chunk = n // tp
    # .contiguous(): 切片是 view，torch.save 会连背后整个 storage 一起存（TP>1 时
    # 每文件都是全量大小）。clone 成独立存储，每 rank 只落盘自己的分片。
    return full[rank * chunk : (rank + 1) * chunk].detach().clone()


def shard_row(full: torch.Tensor, rank: int, tp: int) -> torch.Tensor:
    """RowParallelLinear：按 dim1 切片。"""
    n = full.shape[1]
    assert n % tp == 0, f"dim1 {n} 不能被 TP={tp} 整除"
    chunk = n // tp
    return full[:, rank * chunk : (rank + 1) * chunk].detach().clone()


def shard_qkv_interleaved(
    full: torch.Tensor, rank: int, tp: int, num_heads: int
) -> torch.Tensor:
    """linear_qkv（按头交错布局）的 TP 分片：按 head-group 切片。

    每个 head-group 含 3*hd 行（该头的 q/k/v）。TP 分片 = 均分 head-group，
    保证每个 rank 拿到整头，避免把 q/k/v 拆散。
    """
    per_head = full.shape[0] // (3 * num_heads)
    group = 3 * per_head
    assert num_heads % tp == 0, f"heads {num_heads} 不能被 TP={tp} 整除"
    heads_per_rank = num_heads // tp
    return full[rank * heads_per_rank * group : (rank + 1) * heads_per_rank * group].detach().clone()


def shard_fc1_gate_up(full: torch.Tensor, rank: int, tp: int) -> torch.Tensor:
    """linear_fc1（[gate; up] 行拼接, [2F, H]）的 TP 分片。

    mcore 每 rank 的 swiglu forward: fc1_r(x) → chunk(2,-1) → silu(gate_r)*up_r，
    要求 rank r 的行布局 = [gate 的第 r 块; up 的第 r 块] 配对拼接。
    若直接对 [2F,H] 连续切块（shard_column），rank0/1 会拿到纯 gate、
    rank2/3 拿到纯 up —— swiglu 配对错乱，MLP 输出垃圾
    （实测 3 步冒烟 loss 12.5 = 随机初始化水平的根因）。
    """
    two_f, _ = full.shape
    f = two_f // 2
    assert two_f == 2 * f and f % tp == 0, f"fc1 行数 {two_f} 不是 2×可整除布局"
    chunk = f // tp
    gate, up = full[:f], full[f:]
    return (
        torch.cat(
            [gate[rank * chunk : (rank + 1) * chunk], up[rank * chunk : (rank + 1) * chunk]],
            dim=0,
        )
        .detach()
        .clone()
    )


def build_full_state_dict(
    hf: dict[str, torch.Tensor], num_layers: int, num_heads: int
) -> dict[str, torch.Tensor]:
    """把 HF 全量权重融合成 mcore GPTModel(local spec) 的完整 state_dict。"""
    sd: dict[str, torch.Tensor] = {}

    def put(key: str, t: torch.Tensor) -> None:
        sd[key] = t.detach().clone()

    put("embedding.word_embeddings.weight", hf["model.embed_tokens.weight"])

    for i in range(num_layers):
        L = f"model.layers.{i}"
        q = hf[f"{L}.self_attn.q_proj.weight"]
        k = hf[f"{L}.self_attn.k_proj.weight"]
        v = hf[f"{L}.self_attn.v_proj.weight"]
        # mcore attention.py 布局：mixed_qkv 先 view 成 [sq,b,ng,(np/ng+2)*hd]，
        # 再沿 dim3 按 [query|key|value] split —— 即每个头 g 的 q/k/v 各 hd 行交错：
        #   [q0;k0;v0; q1;k1;v1; ...; q_{n-1};k_{n-1};v_{n-1}]
        # HF 是分离的 q_proj/k_proj/v_proj（行序 [q_all;k_all;v_all]），必须重排成
        # 按头交错，否则 view/split 会把 q/k/v 张冠李戴、attention 输出爆炸。
        hd = q.shape[0] // num_heads
        qv = q.view(num_heads, hd, -1)
        kv = k.view(num_heads, hd, -1)
        vv = v.view(num_heads, hd, -1)
        put(f"decoder.layers.{i}.self_attention.linear_qkv.weight",
            torch.cat([qv, kv, vv], dim=1).reshape(3 * q.shape[0], -1))
        put(f"decoder.layers.{i}.self_attention.linear_proj.weight", hf[f"{L}.self_attn.o_proj.weight"])
        put(f"decoder.layers.{i}.input_layernorm.weight", hf[f"{L}.input_layernorm.weight"])
        put(f"decoder.layers.{i}.pre_mlp_layernorm.weight", hf[f"{L}.post_attention_layernorm.weight"])
        # mcore mlp.py：glu(x)=activation(x[0])*x[1]，前半是 gate → 行序 [gate;up]
        put(f"decoder.layers.{i}.mlp.linear_fc1.weight",
            torch.cat([hf[f"{L}.mlp.gate_proj.weight"], hf[f"{L}.mlp.up_proj.weight"]], dim=0))
        put(f"decoder.layers.{i}.mlp.linear_fc2.weight", hf[f"{L}.mlp.down_proj.weight"])

    put("decoder.final_layernorm.weight", hf["model.norm.weight"])
    put("output_layer.weight", hf["lm_head.weight"])
    return sd


# mcore 中带 _extra_state（None）的模块 —— 加载时 strict=True 需完全匹配
EXTRA_STATE_SUFFIXES = [
    "self_attention.linear_qkv",
    "self_attention.linear_proj",
    "mlp.linear_fc1",
    "mlp.linear_fc2",
]
COLUMN_KEYS = {"embedding.word_embeddings.weight", "output_layer.weight"}
ROW_KEYS = set()


def add_extra_state(sd: dict[str, object], num_layers: int) -> None:
    for i in range(num_layers):
        for suf in EXTRA_STATE_SUFFIXES:
            sd[f"decoder.layers.{i}.{suf}._extra_state"] = None
    sd["output_layer._extra_state"] = None


def build_rank_state_dict(
    full: dict[str, torch.Tensor], rank: int, tp: int, num_layers: int, num_heads: int
) -> dict[str, object]:
    """对给定 TP rank 分片。TP=1 时即为全量。"""
    sd: dict[str, object] = {}
    for key, t in full.items():
        if "layernorm" in key:
            sd[key] = t  # 全量复制
        elif ".linear_qkv." in key:
            sd[key] = shard_qkv_interleaved(t, rank, tp, num_heads)
        elif ".linear_fc1." in key:
            sd[key] = shard_fc1_gate_up(t, rank, tp)
        elif key in COLUMN_KEYS:
            sd[key] = shard_column(t, rank, tp)
        elif key in ROW_KEYS or ".linear_proj." in key or ".linear_fc2." in key:
            sd[key] = shard_row(t, rank, tp)
        else:
            raise ValueError(f"未知 key 的分片类型: {key}")
    add_extra_state(sd, num_layers)
    return sd


def build_rank_pp_state_dict(
    full: dict[str, torch.Tensor],
    tp_rank: int,
    pp_rank: int,
    tp: int,
    pp: int,
    num_layers: int,
    num_heads: int,
) -> dict[str, object]:
    """对给定 (TP rank, PP stage) 分片 —— CoreX 32GB 卡 8 卡 13B 主路径。

    PP 切层规则（与 megatron TransformerBlock._get_layer_offset 一致，无 VPP）：
      stage s 持全局层 [s*L/pp, (s+1)*L/pp)，checkpoint 内重编号为本地
      0..L/pp-1（Megatron state_dict 用 stage 内本地索引）。
    embedding 只进 stage 0（pre_process），final_layernorm/output_layer 只进
    最后 stage（post_process）。
    """
    layers_per_stage = num_layers // pp
    assert num_layers % pp == 0, f"layers {num_layers} 不能被 PP={pp} 整除"
    start = pp_rank * layers_per_stage

    sd: dict[str, object] = {}
    for key, t in full.items():
        if key == "embedding.word_embeddings.weight":
            if pp_rank != 0:
                continue
            sd[key] = shard_column(t, tp_rank, tp)
        elif key in ("decoder.final_layernorm.weight", "output_layer.weight"):
            if pp_rank != pp - 1:
                continue
            sd[key] = t if "layernorm" in key else shard_column(t, tp_rank, tp)
        elif key.startswith("decoder.layers."):
            global_idx = int(key.split(".")[2])
            if not (start <= global_idx < start + layers_per_stage):
                continue
            local_idx = global_idx - start
            local_key = f"decoder.layers.{local_idx}." + key.split(".", 3)[3]
            if "layernorm" in local_key:
                sd[local_key] = t  # 全量复制
            elif ".linear_qkv." in local_key:
                sd[local_key] = shard_qkv_interleaved(t, tp_rank, tp, num_heads)
            elif ".linear_fc1." in local_key:
                sd[local_key] = shard_fc1_gate_up(t, tp_rank, tp)
            elif ".linear_proj." in local_key or ".linear_fc2." in local_key:
                sd[local_key] = shard_row(t, tp_rank, tp)
            else:
                raise ValueError(f"未知 key 的分片类型: {local_key}")
        else:
            raise ValueError(f"未知 key 的分片类型: {key}")
    add_extra_state(sd, layers_per_stage)
    return sd


def main() -> int:
    parser = argparse.ArgumentParser(description="HF → mcore 权重转换")
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--tensor-model-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-model-parallel-size", type=int, default=1,
                        help="PP 切层（32GB 卡 8 卡 13B 建议 TP=4/PP=2）")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--num-layers", type=int, default=40)
    parser.add_argument("--num-heads", type=int, default=40)
    args = parser.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    shards = sorted(glob.glob(os.path.join(args.model_dir, "pytorch_model-*.bin")))
    if not shards:
        print(f"[convert] 未找到 HF 分片: {args.model_dir}", file=__import__("sys").stderr)
        return 2

    hf: dict[str, torch.Tensor] = {}
    for f in shards:
        sd = torch.load(f, map_location="cpu", weights_only=True)
        hf.update(sd)
        del sd
    print(f"[convert] 合并 {len(shards)} 个分片，共 {len(hf)} 个 key")

    full = build_full_state_dict(hf, args.num_layers, args.num_heads)
    del hf
    for k, v in full.items():
        full[k] = v.to(dtype)

    total_params = sum(v.numel() for v in full.values())
    print(f"[convert] 全量参数量: {total_params/1e9:.3f}B")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    iteration_dir = out_dir / "iter_0000001"
    iteration_dir.mkdir(parents=True, exist_ok=True)

    for tp_rank in range(args.tensor_model_parallel_size):
        for pp_rank in range(args.pipeline_model_parallel_size):
            if args.pipeline_model_parallel_size > 1:
                rank_sd = build_rank_pp_state_dict(
                    full, tp_rank, pp_rank,
                    args.tensor_model_parallel_size, args.pipeline_model_parallel_size,
                    args.num_layers, args.num_heads,
                )
                # Megatron 命名约定：PP>1 时 mp_rank_{tp:02d}_{pp:03d}
                ckpt_path = iteration_dir / f"mp_rank_{tp_rank:02d}_{pp_rank:03d}" / "model_optim_rng.pt"
            else:
                rank_sd = build_rank_state_dict(
                    full, tp_rank, args.tensor_model_parallel_size,
                    args.num_layers, args.num_heads,
                )
                ckpt_path = iteration_dir / f"mp_rank_{tp_rank:02d}" / "model_optim_rng.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "checkpoint_version": 3.0,
                "iteration": 1,
                "model": rank_sd,
                "num_floating_point_operations_so_far": 0,
            }
            torch.save(payload, ckpt_path)
            del rank_sd
            print(f"[convert] 写入 {ckpt_path}")

    (out_dir / "latest_checkpointed_iteration.txt").write_text("1\n")
    print(f"[convert] 完成 → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
