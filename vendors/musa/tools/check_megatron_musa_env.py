#!/usr/bin/env python3
"""MUSA(摩尔线程)版 Megatron 环境自检脚本。

与沐曦版 check_megatron_maca_env.py 的关键差异：
1. 设备检查走 torch.musa（torch.cuda 在此容器未编译，是未知 device type，
   由 musa_patch 映射成 torch.musa）。
2. 不 import transformer_engine。真实 MT-TransformerEngine 的 .so 用 AVX-512
   编译，本机 CPU(海光 C86-4G) 无 AVX-512，import 即 SIGILL(硬崩溃)。
   正确状态是 te_stub 让它 import 时抛 ImportError —— 检查脚本断言这一点，
   从而证明 te_stub 已正确屏蔽 TE、Megatron 会走 HAVE_TE=False 分支。
"""

from __future__ import annotations

import importlib
import json
import platform
import sys

from megatron_musa_bootstrap import bootstrap


def main() -> int:
    """输出当前环境的关键检查结果。"""
    info = bootstrap(verbose=False)

    import megatron.training.arguments as megatron_args
    import torch

    device_count = torch.musa.device_count()
    device_name = (
        torch.musa.get_device_name(0) if device_count > 0 else None
    )

    # te_stub 屏蔽状态：真实 TE 应 import 失败（模拟未安装），而不是 SIGILL。
    te_import_error = None
    try:
        importlib.import_module("transformer_engine")
        te_blocked = False
    except ImportError as exc:
        te_blocked = True
        te_import_error = str(exc)
    except BaseException:
        # 任何其它异常（SIGILL 不会到这里，进程直接崩）说明 te_stub 未生效
        te_blocked = False
        te_import_error = "unexpected exception (SIGILL? te_stub 未生效)"

    summary = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "repo_root": info["repo_root"],
        "upstream_root": info["upstream_root"],
        "megatron_parse_args_module": megatron_args.parse_args.__module__,
        "torch_version": torch.__version__,
        "torch_musa_available": torch.musa.is_available(),
        "torch_musa_device_count": device_count,
        "device_name": device_name,
        "transformer_engine_blocked_by_stub": te_blocked,
        "transformer_engine_stub_error": te_import_error,
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    failures: list[str] = []

    if not summary["megatron_parse_args_module"].startswith("megatron.training."):
        failures.append(
            "MUSA Megatron 接口未接管 megatron.training.arguments.parse_args"
        )

    if not summary["torch_musa_available"] or summary["torch_musa_device_count"] <= 0:
        failures.append("当前未检测到可用的 MUSA 设备")

    if not summary["transformer_engine_blocked_by_stub"]:
        failures.append("te_stub 未生效：transformer_engine 本应 import 失败")

    if failures:
        print("检查失败:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("检查通过: 当前 Megatron 环境已按 MUSA 链路完成 bootstrap。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
