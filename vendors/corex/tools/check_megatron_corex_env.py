#!/usr/bin/env python3
"""CoreX(天数智芯)版 Megatron 环境自检脚本。

与摩尔线程版 check_megatron_musa_env.py 的关键差异：
1. 设备检查走 torch.cuda（CoreX 通过 CUDA 兼容层原生暴露，无需垫片），
   同时校验设备名含 "Iluvatar"、torch 为 +corex 官方构建。
2. 不 import transformer_engine（容器未安装）。正确状态是 te_stub 让它
   import 时抛 ImportError —— 检查脚本断言这一点，从而证明 te_stub 已
   正确屏蔽 TE、Megatron 会走 HAVE_TE=False 分支。
3. 额外验证 CoreX 训练链路的三个关键依赖：
   - apex（FusedAdam/FusedLayerNorm，corex 编译版）
   - flash_attn（bf16 前向实测调用）
   - torch.distributed NCCL backend 可用性（libnccl 由 CoreX 提供）
"""

from __future__ import annotations

import importlib
import json
import platform
import sys

from megatron_corex_bootstrap import bootstrap


def main() -> int:
    """输出当前环境的关键检查结果。"""
    info = bootstrap(verbose=False)

    import megatron.training.arguments as megatron_args
    import torch

    device_count = torch.cuda.device_count()
    device_name = (
        torch.cuda.get_device_name(0) if device_count > 0 else None
    )

    # te_stub 屏蔽状态：TE 应 import 失败（容器未安装/被 stub 屏蔽）。
    te_import_error = None
    try:
        importlib.import_module("transformer_engine")
        te_blocked = False
    except ImportError as exc:
        te_blocked = True
        te_import_error = str(exc)
    except BaseException:
        te_blocked = False
        te_import_error = "unexpected exception (te_stub 未生效?)"

    # apex / flash_attn / NCCL：CoreX 官方容器自带，训练链路关键依赖。
    apex_ok, apex_detail = False, None
    try:
        from apex.optimizers import FusedAdam  # noqa: F401
        from apex.normalization import FusedLayerNorm  # noqa: F401

        apex_ok = True
    except Exception as exc:  # noqa: BLE001
        apex_detail = str(exc)

    flash_attn_ok, flash_attn_detail = False, None
    if device_count > 0:
        try:
            from flash_attn import flash_attn_func

            q = torch.randn(1, 64, 4, 64, device="cuda:0", dtype=torch.bfloat16)
            o = flash_attn_func(q, q, q, causal=True)
            flash_attn_ok = tuple(o.shape) == (1, 64, 4, 64)
        except Exception as exc:  # noqa: BLE001
            flash_attn_detail = str(exc)

    import torch.distributed as dist

    nccl_available = dist.is_nccl_available()

    summary = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "repo_root": info["repo_root"],
        "upstream_root": info["upstream_root"],
        "megatron_parse_args_module": megatron_args.parse_args.__module__,
        "torch_version": torch.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_device_count": device_count,
        "device_name": device_name,
        "accelerator_type": "corex" if device_name and "Iluvatar" in device_name else "unknown",
        "apex_ok": apex_ok,
        "apex_detail": apex_detail,
        "flash_attn_ok": flash_attn_ok,
        "flash_attn_detail": flash_attn_detail,
        "nccl_available": nccl_available,
        "transformer_engine_blocked_by_stub": te_blocked,
        "transformer_engine_stub_error": te_import_error,
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    failures: list[str] = []

    if not summary["megatron_parse_args_module"].startswith("megatron.training."):
        failures.append(
            "CoreX Megatron 接口未接管 megatron.training.arguments.parse_args"
        )

    if not summary["torch_cuda_available"] or summary["torch_cuda_device_count"] <= 0:
        failures.append("当前未检测到可用的 CoreX GPU(torch.cuda)")
    elif "Iluvatar" not in (device_name or ""):
        failures.append(f"设备名不含 Iluvatar(当前: {device_name})，非天数智芯 GPU?")

    if not summary["apex_ok"]:
        failures.append(f"apex 不可用: {apex_detail}")

    if not summary["flash_attn_ok"]:
        failures.append(f"flash_attn 不可用: {flash_attn_detail}")

    if not summary["nccl_available"]:
        failures.append("torch.distributed NCCL backend 不可用")

    if not summary["transformer_engine_blocked_by_stub"]:
        failures.append("te_stub 未生效：transformer_engine 本应 import 失败")

    if failures:
        print("检查失败:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("检查通过: CoreX Megatron 运行环境就绪。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
