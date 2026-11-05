#!/usr/bin/env python3
"""把 Megatron 训练日志解析成 CSV（iteration / loss / lr / 吞吐等）。

用法:
  python3 tools/log_to_csv.py --log <worker0日志> [更多日志...] \
      --out generated/glm128_formal/train_metrics.csv
匹配 mcore 0.18.2 的训练日志行（兼容新旧两种 loss/lr 写法）；同一 iteration
出现多次时保留最后一条（重放/重跑场景）；输出按 iteration 升序。
画曲线: 见 tools/plot_loss_curve.py（训完再跑）。
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

FIELDS = [
    "iteration",
    "loss",
    "lr",
    "consumed_samples",
    "consumed_tokens",
    "tokens_per_sec_per_accelerator",
]

RE_ITER = re.compile(r"iteration\s+(\d+)(?:\s*/\s*\d+)?")
RE_LOSS = re.compile(r"lm\s*loss\s*:?\s*([0-9]+(?:\.\d+)?(?:[eE][+-]?\d+)?)")
RE_LOSS_FALLBACK = re.compile(
    r"(?<![a-zA-Z])loss\s*:?\s*([0-9]+\.[0-9]+(?:[eE][+-]?\d+)?)"
)
RE_LR = re.compile(
    r"(?:learning\s*rate|\blr\b)\s*:?\s*([0-9]+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
)
RE_CONSUMED_TOKENS = re.compile(r"consumed\s*tokens:?\s*([\d,]+)")
RE_CONSUMED_SAMPLES = re.compile(r"consumed\s*samples:?\s*([\d,]+)")
RE_TOKENS_PER_SEC = re.compile(r"tokens\s*per\s*second[^:]*:?\s*([\d.,]+)")


def parse_line(line: str) -> dict | None:
    if "iteration" not in line:
        return None
    row: dict = {}
    m = RE_ITER.search(line)
    if not m:
        return None
    row["iteration"] = int(m.group(1))
    m = RE_LOSS.search(line) or RE_LOSS_FALLBACK.search(line)
    if m:
        row["loss"] = float(m.group(1))
    m = RE_LR.search(line)
    if m:
        row["lr"] = float(m.group(1))
    m = RE_CONSUMED_TOKENS.search(line)
    if m:
        row["consumed_tokens"] = int(m.group(1).replace(",", ""))
    m = RE_CONSUMED_SAMPLES.search(line)
    if m:
        row["consumed_samples"] = int(m.group(1).replace(",", ""))
    m = RE_TOKENS_PER_SEC.search(line)
    if m:
        row["tokens_per_sec_per_accelerator"] = float(m.group(1).replace(",", ""))
    # 至少要有 loss 或 lr 之一，否则多半是进度行而非指标行
    if "loss" not in row and "lr" not in row:
        return None
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", nargs="+", required=True, help="训练日志文件（可多个）")
    parser.add_argument("--out", required=True, help="输出 CSV 路径")
    args = parser.parse_args()

    rows: dict[int, dict] = {}
    n_lines = 0
    for path in args.log:
        with open(path, errors="replace") as f:
            for line in f:
                n_lines += 1
                row = parse_line(line)
                if row:
                    rows[row["iteration"]] = row  # 同 iteration 保留最后一条

    if not rows:
        sys.exit(f"未解析到任何指标行（共扫 {n_lines} 行）。请贴几行真实训练日志给我校准正则。")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for it in sorted(rows):
            writer.writerow(rows[it])

    losses = [r["loss"] for r in rows.values() if "loss" in r]
    print(f"[log2csv] {len(rows)} 个 iteration -> {out}")
    if losses:
        print(f"[log2csv] loss: 首值 {losses[0]:.4f} -> 末值 {losses[-1]:.4f}"
              f"（最小 {min(losses):.4f}）")


if __name__ == "__main__":
    main()
