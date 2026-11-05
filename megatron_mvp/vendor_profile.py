"""多厂商档位（vendor profile）注册表。

百亿参数线支持四家国产加速卡：海光 DCU（基线）、沐曦 MetaX MACA、摩尔线程
MUSA、天数智芯 CoreX；千亿参数 EP/MoE 线当前仅海光 BW1000（2026-11-05 定的
范围，三家厂商档走百亿稠密管线）。

每家一个 VendorProfile，收敛渲染启动链路所需的全部厂商差异：

- 激活脚本 / 运行包装器：三家厂商的四件套整体放在 vendors/<厂商>/ 下，
  脚本内根目录按 BASH_SOURCE 相对推导，自包含、整体搬运不需要改路径；
- 能力位：是否允许 sequence parallel（无厂商版 TransformerEngine、走 local
  实现的厂商，torch LayerNorm 不支持 SP，规划与渲染两处都强制关）、
  transformer 实现是否固定 local、是否支持专家并行（EP/MoE）；
- 设备前缀与后端 import：摩尔线程 torch_musa 以 privateuse1 后端挂载，
  需显式 import 注册、设备名是 musa:N；其余三家 torch.cuda 可用。

选择规则：环境变量 AUTOPLAN_VENDOR 显式指定优先（供无厂商硬件的开发机做
跨厂商纯规划，与 AUTOPLAN_STANDIN_GPUS 配套，替身规格也随档位切换）；否则按
detect_runtime_environment() 报出的 accelerator_type 自动匹配；未识别类型
（含通用 "cuda"）回落海光档——即多厂商合并前框架的既有行为，保证旧环境的
渲染产物一字节不变。
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class VendorProfile:
    """一家厂商的启动链路与能力差异描述。"""

    key: str
    display_name: str
    accelerator_types: tuple[str, ...]  # detect_runtime_environment 的匹配值
    activate_script: str                 # 仓库根相对路径
    run_wrapper: str                     # 仓库根相对路径
    allow_sequence_parallel: bool
    fixed_transformer_impl: str | None   # 非 None 时渲染 --transformer-impl
    supports_ep_moe: bool                # 千亿 EP/MoE 仅海光
    device_prefix: str                   # PyTorch 后端示例的设备名前缀
    backend_import_line: str | None      # 需要显式注册后端的厂商（musa）
    te_stub_dir: str | None              # 无 TransformerEngine 厂商的替身目录
    standin_device_name: str             # 纯规划替身卡规格
    standin_memory_bytes: int
    notes: str


_GIB = 1024**3

HYGON = VendorProfile(
    key="hygon",
    display_name="海光 DCU",
    accelerator_types=("hip",),
    activate_script="scripts/activate_megatron_hygon.sh",
    run_wrapper="tools/run_megatron_hygon.py",
    allow_sequence_parallel=True,
    fixed_transformer_impl=None,  # dcu_megatron 0.13 参数体系无 --transformer-impl
    supports_ep_moe=True,         # 千亿 GLM-4.5-Air EP/MoE 与通信模型校准唯一档位
    device_prefix="cuda",
    backend_import_line=None,
    te_stub_dir=None,
    standin_device_name="BW1000-64G-planning-standin",
    standin_memory_bytes=64 * _GIB,
    notes="Phase6 实测常数（机内 1.1TB/s、机间 50GB/s、302.5 TFLOPS）回填在本档。",
)

MACA = VendorProfile(
    key="maca",
    display_name="沐曦 MetaX MACA",
    accelerator_types=("maca",),
    activate_script="vendors/maca/scripts/activate_megatron_maca.sh",
    run_wrapper="vendors/maca/tools/run_megatron_maca.py",
    allow_sequence_parallel=True,  # 厂商版 TransformerEngine 2.3.0 在位
    fixed_transformer_impl="local",
    supports_ep_moe=False,
    device_prefix="cuda",  # cu-bridge 兼容层，torch.cuda 接口直接可用
    backend_import_line=None,
    te_stub_dir=None,
    standin_device_name="MetaX-C500-64G-planning-standin",
    standin_memory_bytes=64 * _GIB,
    notes="厂商版 Megatron-LM 0.12 源码直挂；探测靠 torch 版本串 +metax。",
)

MUSA = VendorProfile(
    key="musa",
    display_name="摩尔线程 MUSA",
    accelerator_types=("musa",),
    activate_script="vendors/musa/scripts/activate_megatron_musa.sh",
    run_wrapper="vendors/musa/tools/run_megatron_musa.py",
    allow_sequence_parallel=False,  # 无 MT-TE：local 实现 torch LayerNorm 不支持 SP
    fixed_transformer_impl="local",
    supports_ep_moe=False,
    device_prefix="musa",
    backend_import_line=(
        'import torch_musa  # noqa: F401'
        '  # 注册 MUSA 为 privateuse1 backend，torch.device("musa:N") 才可用'
    ),
    te_stub_dir="vendors/musa/te_stub",
    standin_device_name="MTT-S5000-80G-planning-standin",
    standin_memory_bytes=80 * _GIB,
    notes="宿主 CPU 无 AVX-512：TE/apex 融合内核必须以 te_stub 屏蔽；"
    "8×S5000 13B 全量微调 66s/步×1250 步≈23h 完整跑通（12.49→2.34）。",
)

COREX = VendorProfile(
    key="corex",
    display_name="天数智芯 CoreX",
    accelerator_types=("corex",),
    activate_script="vendors/corex/scripts/activate_megatron_corex.sh",
    run_wrapper="vendors/corex/tools/run_megatron_corex.py",
    allow_sequence_parallel=False,
    fixed_transformer_impl="local",
    supports_ep_moe=False,
    device_prefix="cuda",  # CUDA 兼容层，torch.cuda 原生可用
    backend_import_line=None,
    te_stub_dir="vendors/corex/te_stub",
    standin_device_name="Iluvatar-BI-V150-32G-planning-standin",
    standin_memory_bytes=32 * _GIB,
    notes="官方 Megatron 0.14rc2 干净源码+6 处幂等补丁；32G 卡 13B 需"
    "TP4/PP2+优化器 CPU 卸载（实测四种纯 GPU 拓扑皆 OOM）。",
)

PROFILES: dict[str, VendorProfile] = {
    profile.key: profile for profile in (HYGON, MACA, MUSA, COREX)
}

_BY_ACCELERATOR: dict[str, VendorProfile] = {
    accelerator: profile
    for profile in PROFILES.values()
    for accelerator in profile.accelerator_types
}


def pick_profile(accelerator_type: str | None) -> VendorProfile:
    """按 detect_runtime_environment() 的 accelerator_type 匹配厂商档。

    未识别类型（含通用 "cuda"）回落海光档——多厂商合并前框架对任何环境都
    渲染海光脚本，回落即保持该历史行为，旧环境产物一字节不变。
    """
    return _BY_ACCELERATOR.get((accelerator_type or "").strip().lower(), HYGON)


def current_profile(accelerator_type: str | None = None) -> VendorProfile:
    """当前厂商档：AUTOPLAN_VENDOR 显式覆盖优先。

    显式覆盖用于无厂商硬件的开发机做跨厂商纯规划（与 AUTOPLAN_STANDIN_GPUS
    配套）；值非法时报错并列出可选档位，防止拼错静默回落。
    """
    override = os.environ.get("AUTOPLAN_VENDOR", "").strip().lower()
    if override:
        if override not in PROFILES:
            raise ValueError(
                f"AUTOPLAN_VENDOR={override!r} 无效；可选: {sorted(PROFILES)}"
            )
        return PROFILES[override]
    return pick_profile(accelerator_type)
