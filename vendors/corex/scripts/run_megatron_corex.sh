#!/usr/bin/env bash
set -euo pipefail

# CoreX(天数智芯)版 Megatron 统一运行入口。
# 示例：
#   scripts/run_megatron_corex.sh Megatron-LM/pretrain_gpt.py --help

_megatron_corex_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_megatron_corex_root="$(cd "${_megatron_corex_script_dir}/.." && pwd)"

# 这里直接 source 环境脚本，确保当前进程内的 Python 路径和 CoreX 路径一致。
source "${_megatron_corex_root}/scripts/activate_megatron_corex.sh"

exec python3 "${_megatron_corex_root}/tools/run_megatron_corex.py" "$@"
