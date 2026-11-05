#!/usr/bin/env python3
"""沐曦版 Megatron 环境自检脚本。"""

from __future__ import annotations

import json
import platform
import sys

from megatron_maca_bootstrap import bootstrap


def main() -> int:
    """输出当前环境的关键检查结果。"""
    info = bootstrap(verbose=False)

    import megatron.training.arguments as megatron_args
    import torch
    import transformer_engine

    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    device_name = (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available() and device_count > 0
        else None
    )

    summary = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "repo_root": info["repo_root"],
        "upstream_root": info["upstream_root"],
        "megatron_parse_args_module": megatron_args.parse_args.__module__,
        "torch_version": torch.__version__,
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_device_count": device_count,
        "device_name": device_name,
        "transformer_engine_version": getattr(transformer_engine, "__version__", "unknown"),
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if not summary["megatron_parse_args_module"].startswith("megatron.training."):
        print("检查失败: 沐曦 Megatron 接口未接管 megatron.training.arguments.parse_args")
        return 1

    if not summary["torch_cuda_available"]:
        print("检查失败: 当前未检测到可用的 CUDA/MXC 设备")
        return 1

    print("检查通过: 当前 Megatron 环境已按沐曦链路完成 bootstrap。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
