#!/usr/bin/env python3
"""把 BAAI/Infinity-Instruct 7M_core parquet 转成 Megatron preprocess_data 需要的 jsonl。

每条样本格式化为 Colossal-LLaMA 风格的完整对话文档：
    <s>Human: {h1}\nAssistant: {a1}\nHuman: {h2}\nAssistant: {a2}</s>

训练口径：base 模型全序列 next-token loss（与 Colossal-LLaMA-2 自身训练方式一致，
不对 prompt 做 loss 掩码）。EOD/BOS 由 preprocess_data.py --append-eod 负责。

用法:
  python3 prepare_infinity_sft.py --input-dir /ws/dataset/7M_core \
      --output /ws/dataset/infinity_7M_core.jsonl [--limit N]

依赖: pyarrow（容器内已装）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def format_conversation(conversations: list[dict]) -> str:
    """把 sharegpt 多轮对话格式化成 Colossal-LLaMA 风格文本。"""
    parts: list[str] = []
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("from", "")).strip().lower()
        value = str(turn.get("value", "")).strip()
        if not value:
            continue
        if role == "human":
            parts.append(f"Human: {value}")
        elif role == "assistant":
            parts.append(f"Assistant: {value}")
        elif role == "system":
            parts.append(f"System: {value}")
        else:
            parts.append(f"{role}: {value}")
    if not parts:
        return ""
    return "<s>" + "\n".join(parts) + "</s>"


def main() -> int:
    parser = argparse.ArgumentParser(description="Infinity-Instruct 7M_core → Megatron jsonl")
    parser.add_argument("--input-dir", type=str, required=True,
                        help="7M_core parquet 目录")
    parser.add_argument("--output", type=str, required=True,
                        help="输出 jsonl 路径")
    parser.add_argument("--limit", type=int, default=None,
                        help="仅处理前 N 条（子集验证用）")
    args = parser.parse_args()

    import pyarrow.parquet as pq

    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob("*.parquet"))
    if not files:
        print(f"[prepare] 未找到 parquet: {input_dir}", file=sys.stderr)
        return 2

    total_written = 0
    text_total_chars = 0
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as fout:
        for fi, path in enumerate(files):
            table = pq.read_table(path)
            rows = table.to_pylist()
            for row in rows:
                if args.limit is not None and total_written >= args.limit:
                    break
                conv = row.get("conversations")
                if not conv:
                    continue
                text = format_conversation(conv)
                if not text:
                    continue
                fout.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                total_written += 1
                text_total_chars += len(text)
            print(f"[prepare] shard {fi + 1}/{len(files)} done, cumulative {total_written}",
                  flush=True)
            if args.limit is not None and total_written >= args.limit:
                break

    print(f"[prepare] 完成: {total_written} 条 → {out_path} ({text_total_chars/1e6:.1f}MB 文本)",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
