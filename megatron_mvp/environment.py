"""运行环境检测。"""

from __future__ import annotations

import os

import torch

from megatron_mvp.types import DeviceInfo
from megatron_mvp.types import RuntimeEnvironment
from megatron_mvp.vendor_profile import current_profile

# 无 GPU 开发机做纯规划(cross-compile)时的单卡替身规格：默认与集群 BW1000
# 对齐；设 AUTOPLAN_VENDOR=<maca|musa|corex> 时切到对应厂商卡替身规格
# （见 megatron_mvp/vendor_profile.py）。


def _topology_overrides() -> dict[str, float | int]:
    """读取 AUTOPLAN_TOPO_* 拓扑覆盖（缺省回落到 types.py 的占位默认值）。

    覆盖项：GPUS_PER_NODE（每节点卡数）、INTRA_BW_GBPS / INTER_BW_GBPS
    （机内/机间有效带宽，GB/s）、INTRA_LAT_US / INTER_LAT_US（时延，微秒）。
    """
    def _env_int(name: str) -> int | None:
        raw = os.environ.get(name, "").strip()
        return int(float(raw)) if raw else None

    overrides: dict[str, float | int] = {}
    mapping = {
        "AUTOPLAN_TOPO_GPUS_PER_NODE": "gpus_per_node",
        "AUTOPLAN_TOPO_INTRA_BW_GBPS": "intra_node_bandwidth_bytes_per_sec",
        "AUTOPLAN_TOPO_INTER_BW_GBPS": "inter_node_bandwidth_bytes_per_sec",
        "AUTOPLAN_TOPO_INTRA_LAT_US": "intra_node_latency_seconds",
        "AUTOPLAN_TOPO_INTER_LAT_US": "inter_node_latency_seconds",
    }
    for env_name, field_name in mapping.items():
        value = _env_int(env_name)
        if value is None:
            continue
        if env_name.endswith("_GBPS"):
            value = float(value) * 1024**3
        elif env_name.endswith("_LAT_US"):
            value = float(value) * 1e-6
        overrides[field_name] = value
    return overrides


def detect_runtime_environment() -> RuntimeEnvironment:
    """检测运行环境（海光 ROCm／沐曦 MACA／摩尔 MUSA／天数 CoreX／通用 CUDA）。

    无 GPU/DCU 的机器（开发机）默认报错；设置环境变量
    AUTOPLAN_STANDIN_GPUS=<卡数> 可返回伪造的 BW1000 替身环境，
    专供 cross-compile 规划用（真实 LAUNCH 在集群上执行，那里不设此变量、
    走真卡检测，不受影响）。替身显式设置时优先于真卡检测——开发机即使
    偶尔挂了 1 张真卡，规划视角也以替身卡数为准。

    设 AUTOPLAN_VENDOR=<maca|musa|corex> 可把替身卡规格与厂商档决策一并
    切换（缺省海光 BW1000），供跨厂商纯规划。

    拓扑字段（G3）统一由 _topology_overrides 从环境变量覆盖，默认值是
    与 BW1000 每节点 8 卡对齐的保守占位，待集群实测校准。
    """
    topology = _topology_overrides()
    standin_count = os.environ.get("AUTOPLAN_STANDIN_GPUS", "").strip()
    if standin_count:
        count = max(1, int(standin_count))
        profile = current_profile()  # AUTOPLAN_VENDOR 优先，缺省海光 BW1000
        devices = [
            DeviceInfo(
                index=index,
                name=profile.standin_device_name,
                total_memory_bytes=profile.standin_memory_bytes,
                multi_processor_count=None,
            )
            for index in range(count)
        ]
        return RuntimeEnvironment(
            accelerator_type=profile.accelerator_types[0],
            device_count=count,
            devices=devices,
            primary_device_name=profile.standin_device_name,
            primary_device_memory_bytes=profile.standin_memory_bytes,
            torch_version=str(torch.__version__),
            hip_version=getattr(torch.version, "hip", None),
            **topology,
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 0:
        raise RuntimeError("当前未检测到可用的 GPU/DCU 设备，无法进行 Megatron 并行搜索。")

    devices: list[DeviceInfo] = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        devices.append(
            DeviceInfo(
                index=index,
                name=props.name,
                total_memory_bytes=int(props.total_memory),
                multi_processor_count=getattr(props, "multi_processor_count", None),
            )
        )

    primary_device = devices[0]
    # 多厂商探测：天数按设备名/构建串、摩尔按 torch.musa 可用性识别；沐曦走
    # cu-bridge 兼容层（torch.cuda 可用）只能靠版本串 "+metax" 识别；其余回落
    # 海光 ROCm（hip）或通用 cuda。档位映射见 vendor_profile.pick_profile。
    device_name = primary_device.name
    torch_build = str(torch.__version__)
    if "Iluvatar" in device_name or "+corex" in torch_build:
        detected_type = "corex"
    elif getattr(torch, "musa", None) is not None and torch.musa.is_available():
        detected_type = "musa"
    elif "+metax" in torch_build:
        detected_type = "maca"
    elif getattr(torch.version, "hip", None):
        detected_type = "hip"
    else:
        detected_type = "cuda"
    return RuntimeEnvironment(
        accelerator_type=detected_type,
        device_count=len(devices),
        devices=devices,
        primary_device_name=primary_device.name,
        primary_device_memory_bytes=primary_device.total_memory_bytes,
        torch_version=str(torch.__version__),
        hip_version=getattr(torch.version, "hip", None),
        **topology,
    )

