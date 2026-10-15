#!/usr/bin/env bash
set -euo pipefail

# 这个脚本只做一件事：
# 调起声明式双卡小规模训练验证入口，验证当前会话是否真的可以跑通双卡 TP 训练。

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

source "${ROOT_DIR}/scripts/activate_megatron_hygon.sh"

VISIBLE_GPU_COUNT="$(python3 - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"

if [ "${VISIBLE_GPU_COUNT}" -lt 2 ]; then
    echo "[autoparallel-test] 当前会话只看到 ${VISIBLE_GPU_COUNT} 张 GPU，不能做真实双卡训练验证。" >&2
    exit 1
fi

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29541}"
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"

python3 "${ROOT_DIR}/examples/autoparallel_dual_gpu_small_train.py"
