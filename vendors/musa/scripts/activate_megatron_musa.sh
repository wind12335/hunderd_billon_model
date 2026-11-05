#!/usr/bin/env bash

# MUSA(摩尔线程)版 Megatron 环境激活脚本。
# 用法：
#   source scripts/activate_megatron_musa.sh
#
# 由海光版 activate_megatron_hygon.sh 适配而来：
#   - DTK(海光) 环境变量 → MUSA(摩尔线程) 环境变量
#   - 环境变量取值参考摩尔线程官方 megatron-lm-musa-patch 示例
#     examples/llama2/7B/run_pretrain_llama2_musa.sh
#   - 额外挂载 te_stub（屏蔽 MT-TransformerEngine：其 .so 用 AVX-512 编译，
#     而本机 CPU(海光 C86-4G) 无 AVX-512，import 即 SIGILL）

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "请使用 source 激活环境: source scripts/activate_megatron_musa.sh"
  exit 1
fi

_megatron_musa_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_megatron_musa_root="$(cd "${_megatron_musa_script_dir}/.." && pwd)"
_megatron_musa_tools="${_megatron_musa_root}/tools"
_megatron_musa_te_stub="${_megatron_musa_root}/te_stub"

# ---------- 上游 Megatron-LM 源码目录 ----------
# 摩尔线程适配版 Megatron 源码。默认在容器内 /home/Megatron-LM，
# 可通过环境变量 MEGATRON_UPSTREAM_ROOT 覆盖。
if [[ -z "${MEGATRON_UPSTREAM_ROOT:-}" ]]; then
  if [[ -d "/home/Megatron-LM" ]]; then
    export MEGATRON_UPSTREAM_ROOT="/home/Megatron-LM"
  elif [[ -d "${_megatron_musa_root}/Megatron-LM" ]]; then
    export MEGATRON_UPSTREAM_ROOT="${_megatron_musa_root}/Megatron-LM"
  fi
fi
if [[ ! -d "${MEGATRON_UPSTREAM_ROOT:-}" ]]; then
  echo "未找到上游 Megatron-LM 源码目录: ${MEGATRON_UPSTREAM_ROOT:-<未设置>}"
  echo "请设置 MEGATRON_UPSTREAM_ROOT 指向摩尔线程适配版 Megatron 源码。"
  return 1
fi

# ---------- conda 环境（可选，容器内通常已就绪） ----------
if [[ -d "/root/miniconda3" ]]; then
  # shellcheck disable=SC1091
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate musa 2>/dev/null || conda activate mxc 2>/dev/null || true
fi

# ---------- MUSA 运行时环境 ----------
# 参考摩尔线程官方示例 run_pretrain_llama2_musa.sh 的 MUSA 环境变量
export ACCELERATOR_BACKEND="${ACCELERATOR_BACKEND:-musa}"
export MUSA_VISIBLE_DEVICES="${MUSA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MUSA_KERNEL_TIMEOUT="${MUSA_KERNEL_TIMEOUT:-3200000}"
export MCCL_PROTOS="${MCCL_PROTOS:-2}"
export MCCL_CHECK_POINTERS="${MCCL_CHECK_POINTERS:-0}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

# 关闭 torch.compile/dynamo：MUSA 下 dynamo 追踪会在
# torch._dynamo.device_interface 崩溃(musa 设备字符串作列表索引)。
# 需在 python 启动前设置，torch._dynamo.config 在 import 时读取。
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export NVTE_TORCH_COMPILE="${NVTE_TORCH_COMPILE:-0}"

# 固定 C/C++ 编译器为系统 GCC，避免 Megatron 编译 C++ 扩展时误用其它工具链
export CC="${CC:-/usr/bin/gcc}"
export CXX="${CXX:-/usr/bin/g++}"

export MEGATRON_PROJECT_ROOT="${_megatron_musa_root}"
export MEGATRON_MUSA_ENABLED="1"
# 关键：te_stub 必须在 Megatron 之前进入 PYTHONPATH，
# 这样 `import transformer_engine` 走 stub 抛 ModuleNotFoundError，
# Megatron 全局 HAVE_TE=False，回退到 apex/torch 实现。
export PYTHONPATH="${_megatron_musa_tools}:${MEGATRON_UPSTREAM_ROOT}:${_megatron_musa_te_stub}:${_megatron_musa_root}:${PYTHONPATH:-}"

echo "[megatron-musa] MEGATRON_PROJECT_ROOT=${MEGATRON_PROJECT_ROOT}"
echo "[megatron-musa] MEGATRON_UPSTREAM_ROOT=${MEGATRON_UPSTREAM_ROOT}"
echo "[megatron-musa] ACCELERATOR_BACKEND=${ACCELERATOR_BACKEND}"
echo "[megatron-musa] MUSA_VISIBLE_DEVICES=${MUSA_VISIBLE_DEVICES}"
echo "[megatron-musa] PYTHONPATH 已追加 tools / Megatron-LM / te_stub / musa_patch"
echo "[megatron-musa] 如需自检，请执行: python3 tools/check_megatron_musa_env.py"
