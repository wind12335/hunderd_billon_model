#!/usr/bin/env bash
set -euo pipefail

# 调起声明式双卡小规模训练验证入口。
# 框架真实入口是 tools/run_autoparallel_e2e_report.py：
# bootstrap → 环境检测 → 声明式规划(@parallelize) → 编译搜索 → 下放(lower) → 流式启动训练。
# 本脚本只是它的薄封装：先确认容器里至少透传了 2 张 MUSA GPU，再以 mock 数据跑 2 卡链路。

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

VISIBLE_GPU_COUNT="$(python3 - <<'PY'
import torch
import torch_musa  # noqa: F401  # 注册 MUSA 为 privateuse1 backend，torch.musa.device_count 才可用
print(torch.musa.device_count())
PY
)"

if [ "${VISIBLE_GPU_COUNT}" -lt 2 ]; then
    echo "[autoparallel-test] 当前会话只看到 ${VISIBLE_GPU_COUNT} 张 GPU，不能做真实双卡训练验证。" >&2
    exit 1
fi

python3 "${ROOT_DIR}/tools/run_autoparallel_e2e_report.py" \
    --backend MEGATRON \
    --data-mode mock \
    --num-devices 2
