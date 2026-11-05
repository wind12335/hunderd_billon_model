#!/usr/bin/env bash

# 沐曦版 Megatron 环境激活脚本。
# 用法：
#   source scripts/activate_megatron_maca.sh
#
# 由海光版 activate_megatron_hygon.sh 适配而来：
#   - DTK(海光) 环境变量 → MACA(沐曦) 环境变量
#   - 环境变量取值参考沐曦官方 megatron 示例脚本
#     maca-megatron-lm-*/Megatron-LM-0.12.0/examples/qwen3/pretrain_qwen3_14B_8gpu.sh

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "请使用 source 激活环境: source scripts/activate_megatron_maca.sh"
  exit 1
fi

_megatron_maca_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_megatron_maca_root="$(cd "${_megatron_maca_script_dir}/.." && pwd)"
_megatron_maca_upstream="${_megatron_maca_root}/Megatron-LM"
_megatron_maca_tools="${_megatron_maca_root}/tools"

# ---------- conda 环境 ----------
# 整套沐曦软件栈（MetaX torch / triton / TE 等）都装在 mxc 环境里，这里直接
# 激活它，保证 `python3` / `torch.distributed.run` 都指向正确的解释器。
if [[ -d "/root/miniconda3" ]]; then
  # shellcheck disable=SC1091
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate mxc
fi

if [[ ! -d "${_megatron_maca_upstream}" ]]; then
  echo "未找到上游 Megatron-LM 源码目录: ${_megatron_maca_upstream}"
  return 1
fi

# ---------- 沐曦 MACA 运行时环境 ----------
if [[ -d "/opt/maca" ]]; then
  export MACA_PATH="${MACA_PATH:-/opt/maca}"
  export MACA_CLANG_PATH="${MACA_CLANG_PATH:-${MACA_PATH}/mxgpu_llvm/bin}"
  export MACA_CLANG="${MACA_CLANG:-${MACA_PATH}/mxgpu_llvm}"
  export DEVINFO_ROOT="${DEVINFO_ROOT:-${MACA_PATH}}"
  export CUCC_PATH="${CUCC_PATH:-${MACA_PATH}/tools/cu-bridge}"
  export CUDA_PATH="${CUCC_PATH}"
  export PATH="${CUCC_PATH}:${MACA_PATH}/bin:${MACA_CLANG}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUCC_PATH}/lib:${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${MACA_PATH}/ompi/lib:${LD_LIBRARY_PATH:-}"
  # 沐曦 Megatron 运行所需的优化开关（参考沐曦官方示例）
  export MACA_SMALL_PAGESIZE_ENABLE=1
  export SET_DEVICE_NUMA_PREFERRED=1
  export PYTORCH_ENABLE_SAME_RAND_A100=1
  # 注意：不再在此设置 NVTE_FLASH_ATTN / NVTE_FUSED_ATTN / NVTE_UNFUSED_ATTN。
  # Megatron 0.12 + TE 2.3.0 会根据 `--attention-backend` 参数自行设置这三个
  # 环境变量；若在此预置 NVTE_FLASH_ATTN=1 + NVTE_FUSED_ATTN=0 这种非 auto
  # 组合，会在 language_module.py 的 _set_attention_backend 断言中报错。
  # 沐曦官方 qwen3 示例使用 flash attention（--attention-backend flash 即
  # flash=1/fused=0/unfused=0），由 launch 脚本传入该参数。
fi

# 固定 C/C++ 编译器为系统 GCC，避免 Megatron 编译 C++ 扩展（如 dataset
# helpers_cpp）时把 cu-bridge/bin 里的 cucc 误当作 C++ 编译器。
export CC="${CC:-/usr/bin/gcc}"
export CXX="${CXX:-/usr/bin/g++}"

export MEGATRON_PROJECT_ROOT="${_megatron_maca_root}"
export MEGATRON_UPSTREAM_ROOT="${_megatron_maca_upstream}"
export MEGATRON_MACA_ENABLED="1"
export PYTHONPATH="${_megatron_maca_tools}:${_megatron_maca_upstream}:${PYTHONPATH:-}"

echo "[megatron-maca] MEGATRON_PROJECT_ROOT=${MEGATRON_PROJECT_ROOT}"
echo "[megatron-maca] MEGATRON_UPSTREAM_ROOT=${MEGATRON_UPSTREAM_ROOT}"
echo "[megatron-maca] MACA_PATH=${MACA_PATH}"
echo "[megatron-maca] PYTHONPATH 已追加 tools 和 Megatron-LM"
echo "[megatron-maca] 如需自检，请执行: python3 tools/check_megatron_maca_env.py"
