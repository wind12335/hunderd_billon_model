#!/usr/bin/env bash
set -euo pipefail

# 该脚本由 autoparallel 生成：通过 SSH 按 hostfile 在所有节点拉起训练。
# 用法（在主节点执行）:
#   bash launch_cluster.sh [--dry-run]
#
# 前置条件:
#   1. 所有节点可通过 ssh 免密登录（端口可用 hostfile 的 host:port 指定）。
#   2. 所有节点在相同路径部署了本仓库（NFS 或手动同步）。
#   3. hostfile 已把 localhost 替换成真实主机名。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

python3 "${PROJECT_ROOT}/tools/launch_cluster.py" \
    --hostfile "${SCRIPT_DIR}/hostfile" \
    --script "${launch_script_path}" \
    --gpus-per-node "${GPUS_PER_NODE:-8}" \
    "$@"
