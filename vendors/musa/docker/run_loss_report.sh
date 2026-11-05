#!/usr/bin/env bash
set -euo pipefail
# 一步生成 loss CSV + 曲线图（容器内跑，共享 /ws 卷）。
#
# 用法:
#   bash /root/workspace/run_loss_report.sh
#   # 可选覆盖默认路径:
#   LOG=/path/log CSV=/path.csv OUT=/path.png bash /root/workspace/run_loss_report.sh
#
# 产物:
#   /root/workspace/ft8_loss.csv      # iteration, lm_loss 两列，可直接 Excel 打开
#   /root/workspace/loss_curve.png    # 曲线图
# 分步执行（自行控制）:
#   docker run --rm -v /root/workspace:/ws <image> python3 \
#     /ws/musa_framework/tools/log_to_csv.py --log /ws/ft8_full.log --csv /ws/ft8_loss.csv
#   docker run --rm -v /root/workspace:/ws <image> python3 \
#     /ws/musa_framework/tools/plot_loss_csv.py --csv /ws/ft8_loss.csv --out /ws/loss_curve.png

IMAGE="sh-harbor.mthreads.com/sets/sqt-vllm-train-musa-bench:musa-sdk-4.3.5_torch2.7.1-intel"
LOG="${LOG:-/root/workspace/ft8_full.log}"
CSV="${CSV:-/root/workspace/ft8_loss.csv}"
OUT="${OUT:-/root/workspace/loss_curve.png}"

# 宿主机路径 → 容器内 /ws 路径
cpt() { echo "/ws${1#/root/workspace}"; }

docker run --rm -v /root/workspace:/ws "$IMAGE" bash -c "
python3 /ws/musa_framework/tools/log_to_csv.py  --log $(cpt "$LOG") --csv $(cpt "$CSV") &&
python3 /ws/musa_framework/tools/plot_loss_csv.py --csv $(cpt "$CSV") --out $(cpt "$OUT")
"
