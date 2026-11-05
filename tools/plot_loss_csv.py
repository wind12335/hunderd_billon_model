#!/usr/bin/env python3
"""读 loss/lr CSV，画 iteration-loss 与 iteration-lr 曲线 PNG（双子图）。

由摩尔线程版 plot_loss_csv.py 扩展：天数版微调要求同时记录 loss 与
learning rate 曲线。输出单张 PNG，上图 loss、下图 lr，方便对照
cosine 退火节奏与 loss 下降的对应关系。

用法:
  python3 plot_loss_csv.py --csv /workspace/ft13b/logs/ft8_loss.csv --out /workspace/ft13b/logs/loss_lr_curve.png

CSV 三列: iteration, lm_loss, learning_rate
（可由 tools/log_to_csv.py 生成；learning_rate 列缺失时只画 loss。）
纯 matplotlib，无 torch 依赖，无头环境直接输出 PNG。
"""

from __future__ import annotations

import argparse
import csv

import matplotlib

matplotlib.use("Agg")  # 无头环境，只输出 PNG
import matplotlib.pyplot as plt  # noqa: E402

TITLE = "colossal-llama2-13B Finetune on Iluvatar BI-V150 x8 (TP=4/PP=2, gbs=256)"


def main() -> int:
    ap = argparse.ArgumentParser(description="loss/lr CSV → 曲线 PNG")
    ap.add_argument("--csv", default="/workspace/ft13b/logs/ft8_loss.csv", help="CSV 路径")
    ap.add_argument("--out", default="/workspace/ft13b/logs/loss_lr_curve.png", help="输出 PNG 路径")
    args = ap.parse_args()

    iters: list[int] = []
    losses: list[float] = []
    lrs: list[float | None] = []
    with open(args.csv, newline="") as f:
        for row in csv.DictReader(f):
            iters.append(int(row["iteration"]))
            losses.append(float(row["lm_loss"]))
            lr_raw = row.get("learning_rate")
            lrs.append(float(lr_raw) if lr_raw not in (None, "") else None)

    if not iters:
        print(f"CSV 为空或没有数据行: {args.csv}", flush=True)
        return 1

    lr_valid = [lr for lr in lrs if lr is not None]
    has_lr = len(lr_valid) > 0

    if has_lr:
        fig, (ax_loss, ax_lr) = plt.subplots(
            2, 1, figsize=(10, 9), dpi=120, sharex=True,
            gridspec_kw={"height_ratios": [3, 2]},
        )
    else:
        fig, ax_loss = plt.subplots(figsize=(10, 6.5), dpi=120)
        ax_lr = None

    # ---- loss 曲线 ----
    ax_loss.plot(iters, losses, marker=".", linestyle="-", color="#d62728", lw=1.2, ms=4)
    ax_loss.set_ylabel("lm loss")
    ax_loss.set_title(TITLE)
    ax_loss.grid(True, alpha=0.3)
    ax_loss.set_xlim(left=0)
    lo, hi = min(losses), max(losses)
    pad = (hi - lo) * 0.08 or 0.5
    ax_loss.set_ylim(lo - pad, hi + pad)
    ax_loss.annotate(
        f"{losses[-1]:.3f}",
        (iters[-1], losses[-1]),
        textcoords="offset points",
        xytext=(8, 6),
        fontsize=9,
        color="#d62728",
    )

    # ---- learning rate 曲线 ----
    if ax_lr is not None and has_lr:
        lr_x = [i for i, lr in zip(iters, lrs) if lr is not None]
        lr_y = [lr for lr in lrs if lr is not None]
        ax_lr.plot(lr_x, lr_y, marker=".", linestyle="-", color="#1f77b4", lw=1.2, ms=4)
        ax_lr.set_xlabel("iteration")
        ax_lr.set_ylabel("learning rate")
        ax_lr.grid(True, alpha=0.3)
        from matplotlib.ticker import ScalarFormatter

        fmt = ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((-2, 2))
        ax_lr.yaxis.set_major_formatter(fmt)
        ax_lr.annotate(
            f"{lr_y[-1]:.2e}",
            (lr_x[-1], lr_y[-1]),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=9,
            color="#1f77b4",
        )
    else:
        ax_loss.set_xlabel("iteration")

    fig.tight_layout()
    fig.savefig(args.out)
    plt.close(fig)

    lr_note = f" + lr({len(lr_valid)} 点)" if has_lr else "（无 lr 列，仅 loss）"
    print(f"已绘制 {len(iters)} 个点{lr_note} -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
