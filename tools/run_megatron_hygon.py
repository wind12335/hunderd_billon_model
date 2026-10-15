#!/usr/bin/env python3
"""海光版 Megatron 运行包装器。

用途：
1. 先完成 bootstrap，确保官方接口层和海光补丁层正确拼接。
2. 再执行指定的 Megatron 入口脚本，例如 `pretrain_gpt.py`。
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

from megatron_hygon_bootstrap import bootstrap


def _resolve_script(script_arg: str) -> Path:
    """解析目标脚本路径，优先兼容仓库相对路径。

    搜索顺序：
    1. 原样（绝对/相对当前目录）。
    2. <repo>/<script_arg>，例如 Megatron-LM/pretrain_gpt.py。
    3. hcu-megatron 布局：把 "Megatron-LM/" 前缀重定向到
       Megatron-LM-das/3rdparty/Megatron-LM 或 Megatron-LM-das 根目录，
       兼容 entrypoint 固定为 "Megatron-LM/pretrain_gpt.py" 的旧配置。
    """
    raw_path = Path(script_arg)
    if raw_path.is_file():
        return raw_path.resolve()

    repo_root = Path(__file__).resolve().parent.parent

    candidates = [repo_root / script_arg]
    if script_arg.startswith("Megatron-LM/"):
        stripped = script_arg[len("Megatron-LM/"):]
        candidates.append(repo_root / "Megatron-LM-das" / "3rdparty" / "Megatron-LM" / stripped)
        candidates.append(repo_root / "Megatron-LM-das" / stripped)
    candidates.append(repo_root / "Megatron-LM-das" / "3rdparty" / "Megatron-LM" / script_arg)
    candidates.append(repo_root / "Megatron-LM-das" / script_arg)

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    searched = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"未找到目标脚本: {script_arg}\n已搜索:\n  {searched}\n"
        "提示: 若使用 Megatron-LM-das 布局，请先执行 "
        "git submodule update --init --recursive"
    )


def _strip_flag_pairs(args: list[str], flag_names: tuple[str, ...]) -> list[str]:
    """移除指定 flag 及其取值，避免静态参数覆盖 torchrun 的真实环境。"""
    normalized_flags = set(flag_names)
    stripped: list[str] = []
    index = 0
    while index < len(args):
        current = args[index]
        current_name = current.split("=", 1)[0]
        if current_name in normalized_flags:
            if "=" not in current and index + 1 < len(args):
                index += 2
                continue
            index += 1
            continue
        stripped.append(current)
        index += 1
    return stripped


def _inject_runtime_distributed_args(args: list[str]) -> list[str]:
    """按当前子进程环境注入 rank/world-size/local-rank/dist-url。"""
    sanitized = _strip_flag_pairs(
        args,
        flag_names=(
            "--rank",
            "--world-size",
            "--dist-url",
            "--local-rank",
            "--local_rank",
        ),
    )
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = os.environ.get("MASTER_PORT", "29500")
    rank = os.environ.get("RANK", "0")
    world_size = os.environ.get("WORLD_SIZE", "1")
    local_rank = os.environ.get("LOCAL_RANK", "0")
    return [
        *sanitized,
        "--rank",
        rank,
        "--world-size",
        world_size,
        "--dist-url",
        f"tcp://{master_addr}:{master_port}",
        "--local-rank",
        local_rank,
    ]


def main() -> int:
    """执行海光版 Megatron 入口脚本。"""
    if len(sys.argv) < 2:
        print(
            "用法: python3 tools/run_megatron_hygon.py <script.py> [args ...]\n"
            "示例: python3 tools/run_megatron_hygon.py Megatron-LM/pretrain_gpt.py "
            "--help"
        )
        return 2

    target_script = _resolve_script(sys.argv[1])
    forwarded_args = _inject_runtime_distributed_args(sys.argv[2:])

    # dcu_megatron 在导入期间会介入参数链路，因此要先切换 sys.argv，
    # 这样 `--help` 等输出会显示真正的目标脚本名，而不是包装器脚本名。
    sys.argv = [str(target_script), *forwarded_args]
    bootstrap(verbose=False)

    runpy.run_path(str(target_script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
