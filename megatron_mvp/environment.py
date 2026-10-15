"""运行环境检测。"""

from __future__ import annotations

import torch

from megatron_mvp.types import DeviceInfo
from megatron_mvp.types import RuntimeEnvironment


def detect_runtime_environment() -> RuntimeEnvironment:
    """检测当前海光/ROCm 运行环境。"""
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
    return RuntimeEnvironment(
        accelerator_type="hip" if getattr(torch.version, "hip", None) else "cuda",
        device_count=len(devices),
        devices=devices,
        primary_device_name=primary_device.name,
        primary_device_memory_bytes=primary_device.total_memory_bytes,
        torch_version=str(torch.__version__),
        hip_version=getattr(torch.version, "hip", None),
    )

