#!/usr/bin/env bash

# 海光版 Megatron 环境激活脚本（多机/多布局兼容版）。
# 用法：
#   source scripts/activate_megatron_hygon.sh
#
# 上游 Megatron-LM 源码目录按以下优先级解析：
#   1. 环境变量 MEGATRON_UPSTREAM_ROOT
#   2. <repo>/Megatron-LM（旧布局）
#   3. <repo>/Megatron-LM-das/3rdparty/Megatron-LM（hcu-megatron submodule 布局）
# 找不到时不阻断 source（PYTORCH backend 不需要上游源码），
# 只打印警告；MEGATRON backend 会在 bootstrap 阶段给出明确报错。

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "请使用 source 激活环境: source scripts/activate_megatron_hygon.sh"
  exit 1
fi

_megatron_hygon_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_megatron_hygon_root="$(cd "${_megatron_hygon_script_dir}/.." && pwd)"
_megatron_hygon_tools="${_megatron_hygon_root}/tools"

# 解析上游 Megatron-LM 源码目录。
_megatron_hygon_upstream=""
_upstream_candidates=(
  "${MEGATRON_UPSTREAM_ROOT:-}"
  "${_megatron_hygon_root}/Megatron-LM"
  "${_megatron_hygon_root}/Megatron-LM-das/3rdparty/Megatron-LM"
)
for _candidate in "${_upstream_candidates[@]}"; do
  if [[ -n "${_candidate}" && -f "${_candidate}/megatron/training/arguments.py" ]]; then
    _megatron_hygon_upstream="${_candidate}"
    break
  fi
done

if [[ -z "${_megatron_hygon_upstream}" ]]; then
  echo "[megatron-hygon] 警告: 未找到上游 Megatron-LM 源码目录（已尝试 Megatron-LM/ 与 Megatron-LM-das/3rdparty/Megatron-LM）。"
  echo "[megatron-hygon] MEGATRON backend 需要先执行: cd Megatron-LM-das && git submodule update --init --recursive"
  echo "[megatron-hygon] PYTORCH backend 不受影响，可继续运行。"
fi

if [[ -d "/opt/dtk" ]]; then
  export DTK_HOME="/opt/dtk"
  export ROCM_HOME="${ROCM_HOME:-/opt/dtk}"
  export HIP_PATH="${HIP_PATH:-/opt/dtk}"
  export PATH="/opt/dtk/bin:${PATH}"
  export LD_LIBRARY_PATH="/opt/dtk/lib:/opt/dtk/lib64:${LD_LIBRARY_PATH:-}"
fi

export MEGATRON_PROJECT_ROOT="${_megatron_hygon_root}"
export MEGATRON_HYGON_ENABLED="1"

if [[ -n "${_megatron_hygon_upstream}" ]]; then
  export MEGATRON_UPSTREAM_ROOT="${_megatron_hygon_upstream}"
  export PYTHONPATH="${_megatron_hygon_tools}:${_megatron_hygon_upstream}:${PYTHONPATH:-}"
else
  export PYTHONPATH="${_megatron_hygon_tools}:${PYTHONPATH:-}"
fi

echo "[megatron-hygon] MEGATRON_PROJECT_ROOT=${MEGATRON_PROJECT_ROOT}"
echo "[megatron-hygon] MEGATRON_UPSTREAM_ROOT=${MEGATRON_UPSTREAM_ROOT:-<未找到，仅 PYTORCH backend 可用>}"
echo "[megatron-hygon] PYTHONPATH=${PYTHONPATH}"
echo "[megatron-hygon] 如需自检，请执行: python3 tools/check_megatron_hygon_env.py"
