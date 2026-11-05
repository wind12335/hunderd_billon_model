#!/usr/bin/env python3
"""Infinity-Instruct-7M-Core 打包为 Megatron mmap 数据集（GLM tokenizer）。

流水线: 15 个 parquet 分片 -> GLM-4 对话格式文本 -> 固定种子全局洗牌取半
-> tokenize -> 连续切块(seq_len) -> MMapIndexedDataset(.bin/.idx, int32)。

用法（正式，半量）:
  python3 tools/pack_instruct_data.py \
      --input datasets/infinity_instruct_7m_core/7M_core \
      --tokenizer models/glm45_air_base \
      --out datasets/packed_glm_half_2048/instruct_text \
      --seq-len 2048 --half --seed 42
冒烟（前 N 条全链路验证，不洗牌不取半）:
  ... --limit 500
训练侧接线（render 层负责）:
  --data-path <out 前缀> --tokenizer-type NullTokenizer --vocab-size 151552

决策记录（有意为之，改动前先想清楚）:
  * 对话格式: Base 版 tokenizer_config 无 chat_template，按 GLM-4 系列官方
    格式手写——序列头 [gMASK]<sop>，角色块 <|system|>/<|user|>/<|assistant|>，
    对话末尾 <|endoftext|>（eos，也是 pad）。
  * 损失: 全序列参与损失（不 mask prompt）。mcore 经典 mmap 路线没有逐
    token loss_mask；PDF 量级的指令微调常接受此简化。若要严格 mask，
    需要自定义数据集返回 loss_mask，工程量另计。
  * 打包: 文档流连续切块（Megatron 预训练经典做法），块边界不回避文档
    边界；不足一块的尾部丢弃（约 <2048 token，占比可忽略）。
  * dtype int32: vocab 151552 超过 uint16 上限(65535)，不能按 8B 时的
    uint16 惯例走。
  * .idx 写入器: 本文件自带最小实现，字节级镜像 mcore 0.18.2
    indexed_dataset._IndexWriter 的布局（魔数 MMIDIDX\\x00\\x00、版本 1、
    dtype 码 4、长度 int32/指针 int64/文档索引 int64）——不 import
    megatron（那条链会拖 triton，开发机没有）。
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

_INDEX_MAGIC = b"MMIDIDX\x00\x00"
_DTYPE_CODE_INT32 = 4  # mcore DType: int32=4
_INT32_ITEMSIZE = 4


def write_index(idx_path: str, sequence_lengths: list[int]) -> None:
    """按 mcore 0.18.2 布局写 .idx；每块即一个文档，文档索引=[0..N]。"""
    n = len(sequence_lengths)
    pointers = np.zeros(n, dtype=np.int64)
    if n:
        # 字节偏移 = 前缀 token 数 × 4
        pointers[1:] = np.cumsum(sequence_lengths, dtype=np.int64)[:-1] * _INT32_ITEMSIZE
    document_indices = np.arange(n + 1, dtype=np.int64)
    with open(idx_path, "wb") as f:
        f.write(_INDEX_MAGIC)
        f.write(struct.pack("<Q", 1))                      # 版本（固定 1）
        f.write(struct.pack("<B", _DTYPE_CODE_INT32))      # dtype 码
        f.write(struct.pack("<Q", n))                      # 序列数
        f.write(struct.pack("<Q", n + 1))                  # 文档数
        f.write(np.array(sequence_lengths, dtype=np.int32).tobytes())
        f.write(pointers.tobytes())
        f.write(document_indices.tobytes())

ROLE_USER = {"human", "user"}
ROLE_ASSISTANT = {"gpt", "assistant"}

# GLM-4 系列对话格式特殊 token（Base tokenizer 的 added_tokens 已含全部）
SP_PREFIX = "[gMASK]<sop>"
SP_SYSTEM = "<|system|>\n"
SP_USER = "<|user|>\n"
SP_ASSISTANT = "<|assistant|>\n"


def conversation_to_text(conversation: list[dict]) -> str | None:
    """把一条 sharegpt 式对话渲染为 GLM-4 格式纯文本。

    无法识别的角色（tool/observation 等罕见角色）返回 None，调用方跳过并计数。
    """
    parts: list[str] = []
    for turn in conversation:
        role = turn.get("from", "")
        value = (turn.get("value") or "").strip()
        if not value:
            return None
        if role in ROLE_USER:
            parts.append(SP_USER + value)
        elif role in ROLE_ASSISTANT:
            parts.append(SP_ASSISTANT + value)
        elif role == "system":
            parts.append(SP_SYSTEM + value)
        else:
            return None
    if not any(p.startswith(SP_ASSISTANT) for p in parts):
        return None  # 没有助手回合的样本对 SFT 无意义，跳过
    return SP_PREFIX + "".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="7M_core parquet 目录")
    parser.add_argument("--tokenizer", required=True, help="GLM tokenizer 目录")
    parser.add_argument("--out", required=True, help="输出前缀（生成 .bin/.idx）")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--half", action="store_true", help="固定种子洗牌后取半量")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0, help="冒烟模式：只取前 N 条")
    parser.add_argument("--batch-tokens", type=int, default=512, help="分批 tokenize 的对话数")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    eos_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    assert eos_id is not None and eos_id >= 0, "tokenizer 缺 <|endoftext|>"

    shards = sorted(Path(args.input).glob("*.parquet"))
    assert shards, f"目录里没有 parquet: {args.input}"
    total_rows = sum(pq.read_metadata(f).num_rows for f in shards)
    print(f"[pack] 分片 {len(shards)} 个，总行数 {total_rows}")

    selected = np.ones(total_rows, dtype=bool)
    if args.limit:
        selected[:] = False
        selected[: args.limit] = True
        print(f"[pack] 冒烟模式：只取前 {args.limit} 条")
    elif args.half:
        perm = np.random.RandomState(args.seed).permutation(total_rows)
        selected[:] = False
        selected[perm[: total_rows // 2]] = True
        print(f"[pack] 种子 {args.seed} 洗牌取半：{int(selected.sum())} 条")

    out_prefix = Path(args.out)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    data_file = open(str(out_prefix) + ".bin", "wb")
    sequence_lengths: list[int] = []

    stats = {
        "docs_seen": 0,
        "docs_used": 0,
        "docs_skipped_role": 0,
        "docs_skipped_no_assistant": 0,
        "tokens": 0,
        "blocks": 0,
        "tail_dropped_tokens": 0,
    }
    buffer: list[int] = []
    t0 = time.time()

    def flush_blocks() -> None:
        while len(buffer) >= args.seq_len:
            block = buffer[: args.seq_len]
            del buffer[: args.seq_len]
            data_file.write(np.array(block, dtype=np.int32).tobytes())
            sequence_lengths.append(args.seq_len)
            stats["blocks"] += 1
            stats["tokens"] += len(block)

    global_idx = 0
    pending_texts: list[str] = []

    def drain_pending() -> None:
        nonlocal pending_texts
        if not pending_texts:
            return
        enc = tokenizer(pending_texts, add_special_tokens=False)["input_ids"]
        for ids in enc:
            buffer.extend(ids)
            buffer.append(eos_id)  # 文档分隔符（对话末尾 eos）
        flush_blocks()
        pending_texts = []

    for shard in shards:
        if args.limit and global_idx >= args.limit:
            break
        reader = pq.ParquetFile(shard)
        for batch in reader.iter_batches(batch_size=1024, columns=["conversations"]):
            if args.limit and global_idx >= args.limit:
                break
            for row in batch.column("conversations").to_pylist():
                if selected[global_idx]:
                    text = conversation_to_text(row)
                    stats["docs_seen"] += 1
                    if text is None:
                        had_asst = any(
                            (t.get("from") in ROLE_ASSISTANT) for t in row
                        )
                        key = (
                            "docs_skipped_no_assistant"
                            if had_asst
                            else "docs_skipped_role"
                        )
                        stats[key] += 1
                    else:
                        pending_texts.append(text)
                        stats["docs_used"] += 1
                        if len(pending_texts) >= args.batch_tokens:
                            drain_pending()
                global_idx += 1
            if global_idx % 100000 < 1024:
                drain_pending()
                rate = stats["docs_used"] / max(time.time() - t0, 1e-6)
                print(
                    f"[pack] 进度 {global_idx}/{total_rows} 行，已用 "
                    f"{stats['docs_used']} 条，{stats['blocks']} 块，"
                    f"{stats['tokens']/1e6:.1f}M token，{rate:.0f} 条/s",
                    flush=True,
                )
    drain_pending()
    stats["tail_dropped_tokens"] = len(buffer)
    data_file.close()
    write_index(str(out_prefix) + ".idx", sequence_lengths)

    meta = {
        "input": args.input,
        "tokenizer": args.tokenizer,
        "seed": args.seed,
        "half": bool(args.half),
        "limit": args.limit,
        "seq_len": args.seq_len,
        "vocab": tokenizer.vocab_size,
        "eos_id": eos_id,
        "elapsed_seconds": round(time.time() - t0, 1),
        **stats,
    }
    Path(str(out_prefix) + ".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2)
    )
    print(f"[pack] 完成: {stats['blocks']} 块 × {args.seq_len} = "
          f"{stats['tokens']/1e6:.1f}M token；尾丢 {stats['tail_dropped_tokens']} token")
    print(f"[pack] 产物: {out_prefix}.bin/.idx；元数据: {out_prefix}.meta.json")
    print(f"[pack] 步数估算(gbs 256): {stats['blocks'] * args.seq_len / (256 * args.seq_len):.0f} 步")


if __name__ == "__main__":
    main()
