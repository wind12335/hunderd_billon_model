#!/usr/bin/env python3
"""沐曦版 Megatron 环境 bootstrap。

由海光版 megatron_hygon_bootstrap.py 适配而来：
  - 海光需要额外导入 `dcu_megatron` 打补丁；
  - 沐曦的 Megatron-LM 是完整可运行源码，无需补丁层，
    只需把本地 `Megatron-LM` 源码挂进 sys.path 并验证官方接口可用。

设计目标：
1. 不改动官方源码目录。
2. 运行入口统一走沐曦 Megatron-LM 源码链路。
3. 在当前仓库任意位置执行时都能得到一致结果。
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


_BOOTSTRAPPED = False


def _repo_root() -> Path:
    """返回当前仓库根目录。"""
    return Path(__file__).resolve().parent.parent


def _upstream_root() -> Path:
    """返回本地 Megatron-LM 源码目录。"""
    return _repo_root() / "Megatron-LM"


def _prepend_sys_path(path: Path) -> None:
    """把路径放到 sys.path 前部，同时避免重复项。"""
    path_str = str(path)
    sys.path[:] = [entry for entry in sys.path if entry != path_str]
    sys.path.insert(0, path_str)


def bootstrap(verbose: bool = False) -> dict[str, str]:
    """初始化沐曦 Megatron 运行环境。

    返回值用于调试和自检，包含仓库根目录和上游源码目录信息。
    """
    global _BOOTSTRAPPED

    repo_root = _repo_root()
    upstream_root = _upstream_root()

    if not upstream_root.is_dir():
        raise FileNotFoundError(
            f"未找到上游 Megatron-LM 源码目录: {upstream_root}"
        )

    # 先把本地 Megatron-LM 源码放到 import 链路前面，保证
    # `megatron.training` 等官方接口可用。
    _prepend_sys_path(upstream_root)

    # 记录环境，便于脚本和排障时直接查看。
    os.environ.setdefault("MEGATRON_PROJECT_ROOT", str(repo_root))
    os.environ.setdefault("MEGATRON_UPSTREAM_ROOT", str(upstream_root))
    os.environ.setdefault("MEGATRON_MACA_ENABLED", "1")

    # 加载沐曦 Megatron 官方训练接口，确认可正常导入。
    if not _BOOTSTRAPPED:
        importlib.import_module("megatron.training.arguments")
        _BOOTSTRAPPED = True

    patched_module = importlib.import_module("megatron.training.arguments")
    patched_parse_args = getattr(patched_module, "parse_args", None)
    patched_module_name = getattr(patched_parse_args, "__module__", "")

    if not patched_module_name.startswith("megatron.training."):
        raise RuntimeError(
            "沐曦 Megatron 接口异常: megatron.training.arguments.parse_args "
            f"当前来自 {patched_module_name or 'unknown'}"
        )

    info = {
        "repo_root": str(repo_root),
        "upstream_root": str(upstream_root),
        "parse_args_module": patched_module_name,
    }

    if verbose:
        print("[megatron-maca-bootstrap] repo_root =", info["repo_root"])
        print("[megatron-maca-bootstrap] upstream_root =", info["upstream_root"])
        print(
            "[megatron-maca-bootstrap] parse_args_module =",
            info["parse_args_module"],
        )

    return info


if __name__ == "__main__":
    bootstrap(verbose=True)
