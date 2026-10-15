#!/usr/bin/env bash
set -euo pipefail

# 海光版 Megatron 统一运行入口。
# 示例：
#   scripts/run_megatron_hygon.sh Megatron-LM/pretrain_gpt.py --help

_megatron_hygon_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_megatron_hygon_root="$(cd "${_megatron_hygon_script_dir}/.." && pwd)"

# 这里直接 source 环境脚本，确保当前进程内的 Python 路径和 DTK 路径一致。
source "${_megatron_hygon_root}/scripts/activate_megatron_hygon.sh"

exec python3 "${_megatron_hygon_root}/tools/run_megatron_hygon.py" "$@"
