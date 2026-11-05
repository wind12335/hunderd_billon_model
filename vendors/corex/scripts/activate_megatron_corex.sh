#!/usr/bin/env bash

# CoreX(天数智芯)版 Megatron 环境激活脚本。
# 用法：
#   source scripts/activate_megatron_corex.sh
#
# 由摩尔线程版 activate_megatron_musa.sh 适配而来：
#   - MUSA 环境变量(MUSA_VISIBLE_DEVICES/MCCL_*) → CUDA 兼容层环境变量
#   - CoreX 自带 libnccl + corex 编译版 apex/flash_attn，无需额外垫片
#   - te_stub 保留（容器未安装 TransformerEngine，--transformer-impl local）
#
# 说明：本机共 16 张 BI-V150；8 卡微调默认用前 8 张(CUDA_VISIBLE_DEVICES=0-7)。

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "请使用 source 激活环境: source scripts/activate_megatron_corex.sh"
  exit 1
fi

_megatron_corex_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_megatron_corex_root="$(cd "${_megatron_corex_script_dir}/.." && pwd)"
_megatron_corex_tools="${_megatron_corex_root}/tools"
_megatron_corex_te_stub="${_megatron_corex_root}/te_stub"

# ---------- 上游 Megatron-LM 源码目录 ----------
# 上游干净源码(core_v0.14.0rc2)。默认在仓库内 Megatron-LM，
# 可通过环境变量 MEGATRON_UPSTREAM_ROOT 覆盖。
if [[ -z "${MEGATRON_UPSTREAM_ROOT:-}" ]]; then
  if [[ -d "/home/Megatron-LM" ]]; then
    export MEGATRON_UPSTREAM_ROOT="/home/Megatron-LM"
  elif [[ -d "${_megatron_corex_root}/Megatron-LM" ]]; then
    export MEGATRON_UPSTREAM_ROOT="${_megatron_corex_root}/Megatron-LM"
  fi
fi
if [[ ! -d "${MEGATRON_UPSTREAM_ROOT:-}" ]]; then
  echo "未找到上游 Megatron-LM 源码目录: ${MEGATRON_UPSTREAM_ROOT:-<未设置>}"
  echo "请设置 MEGATRON_UPSTREAM_ROOT 指向 Megatron-LM 源码。"
  return 1
fi

# ---------- CoreX 运行时环境 ----------
# CoreX 通过 CUDA 兼容层暴露设备；驱动/工具链已由 /etc/profile.d 注入：
#   PATH=/usr/local/corex/bin:/usr/local/iluvatar/bin:$PATH
#   LD_LIBRARY_PATH=/usr/local/corex/lib64:/usr/local/iluvatar/lib64:$LD_LIBRARY_PATH
export ACCELERATOR_BACKEND="${ACCELERATOR_BACKEND:-corex}"
# 8 卡微调取前 8 张卡；本机共 16 张 BI-V150（torch 设备与物理卡 1:1）
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

# 关闭 torch.compile/dynamo：与 bootstrap 的 jit.py 补丁一致（首发稳态）。
# 需在 python 启动前设置，torch._dynamo.config 在 import 时读取。
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export NVTE_TORCH_COMPILE="${NVTE_TORCH_COMPILE:-0}"

# 固定 C/C++ 编译器为系统 GCC，避免 Megatron 编译 C++ 扩展时误用其它工具链
export CC="${CC:-/usr/bin/gcc}"
export CXX="${CXX:-/usr/bin/g++}"

export MEGATRON_PROJECT_ROOT="${_megatron_corex_root}"
export MEGATRON_COREX_ENABLED="1"
# 关键：te_stub 必须在 Megatron 之前进入 PYTHONPATH，
# 这样 `import transformer_engine` 走 stub 抛 ModuleNotFoundError，
# Megatron 全局 HAVE_TE=False，回退到 apex/torch/local 实现。
export PYTHONPATH="${_megatron_corex_tools}:${MEGATRON_UPSTREAM_ROOT}:${_megatron_corex_te_stub}:${_megatron_corex_root}:${PYTHONPATH:-}"

echo "[megatron-corex] MEGATRON_PROJECT_ROOT=${MEGATRON_PROJECT_ROOT}"
echo "[megatron-corex] MEGATRON_UPSTREAM_ROOT=${MEGATRON_UPSTREAM_ROOT}"
echo "[megatron-corex] ACCELERATOR_BACKEND=${ACCELERATOR_BACKEND}"
echo "[megatron-corex] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[megatron-corex] PYTHONPATH 已追加 tools / Megatron-LM / te_stub"
echo "[megatron-corex] 如需自检，请执行: python3 tools/check_megatron_corex_env.py"
