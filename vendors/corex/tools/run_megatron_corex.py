#!/usr/bin/env python3
"""CoreX(天数智芯)版 Megatron 运行包装器。

由摩尔线程版 run_megatron_musa.py 适配而来，仅替换 bootstrap 模块：
  - 摩尔线程：megatron_musa_bootstrap（te_stub + 源码补丁 + musa_patch
    的 torch.cuda->torch.musa 兼容层）
  - 天数智芯：megatron_corex_bootstrap（te_stub + 源码补丁；
    CoreX 走 CUDA 兼容层，torch.cuda 原生可用，无需任何设备垫片）

用途：
1. 先完成 bootstrap，确保上游 Megatron-LM 正确挂进 import 链路、
   te_stub 屏蔽 TransformerEngine、corex 源码补丁生效。
2. 再执行指定的 Megatron 入口脚本，例如 `pretrain_gpt.py`。
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

from megatron_corex_bootstrap import bootstrap


def _resolve_script(
    script_arg: str, upstream_root: Path | None = None
) -> Path:
    """解析目标脚本路径，优先兼容仓库相对路径。

    上游 Megatron 位于仓库内 Megatron-LM/（或 MEGATRON_UPSTREAM_ROOT），
    render 生成的 launch 脚本用 `Megatron-LM/pretrain_gpt.py` 相对路径，
    这里依次按 原样路径 → 仓库相对 → 上游相对 兜底解析。
    """
    raw_path = Path(script_arg)
    if raw_path.is_file():
        return raw_path.resolve()

    repo_root = Path(__file__).resolve().parent.parent
    candidate = repo_root / script_arg
    if candidate.is_file():
        return candidate.resolve()

    if upstream_root is not None:
        rel_arg = script_arg
        # 上游 Megatron 直接位于 upstream_root（如 /home/Megatron-LM）时不带
        # Megatron-LM/ 前缀；render 生成的路径带此前缀，这里剥掉再拼。
        if rel_arg.startswith("Megatron-LM/"):
            rel_arg = rel_arg[len("Megatron-LM/"):]
        candidate = upstream_root / rel_arg
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(f"未找到目标脚本: {script_arg}")


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


def _strip_distributed_args(args: list[str]) -> list[str]:
    """剔除 rank/world-size/dist-url 等显式分布式参数。

    上游 Megatron 是官方参数体系：分布式信息由 torchrun 通过
    RANK/WORLD_SIZE/LOCAL_RANK 环境变量注入，pretrain_gpt.py 不识别
    `--rank/--world-size/--dist-url` 参数。因此这里只剔除这些参数，
    其余参数原样透传给 Megatron。
    """
    return _strip_flag_pairs(
        args,
        flag_names=(
            "--rank",
            "--world-size",
            "--dist-url",
            "--local-rank",
            "--local_rank",
        ),
    )


def main() -> int:
    """执行 CoreX 版 Megatron 入口脚本。"""
    if len(sys.argv) < 2:
        print(
            "用法: python3 tools/run_megatron_corex.py <script.py> [args ...]\n"
            "示例: python3 tools/run_megatron_corex.py Megatron-LM/pretrain_gpt.py "
            "--help"
        )
        return 2

    info = bootstrap(verbose=False)
    target_script = _resolve_script(
        sys.argv[1], upstream_root=Path(info["upstream_root"])
    )
    forwarded_args = _strip_distributed_args(sys.argv[2:])

    # 先切换 sys.argv，保证 `--help` 等输出显示真正的目标脚本名。
    sys.argv = [str(target_script), *forwarded_args]

    runpy.run_path(str(target_script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
