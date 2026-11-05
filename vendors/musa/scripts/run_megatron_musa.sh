#!/usr/bin/env bash
set -euo pipefail

# MUSA(摩尔线程)版 Megatron 统一运行入口。
# 示例：
#   scripts/run_megatron_musa.sh Megatron-LM/pretrain_gpt.py --help

_megatron_musa_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_megatron_musa_root="$(cd "${_megatron_musa_script_dir}/.." && pwd)"

# 这里直接 source 环境脚本，确保当前进程内的 Python 路径和 MUSA 路径一致。
source "${_megatron_musa_root}/scripts/activate_megatron_musa.sh"

exec python3 "${_megatron_musa_root}/tools/run_megatron_musa.py" "$@"
