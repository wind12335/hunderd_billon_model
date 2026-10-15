"""解释器级 typing 兼容垫片（与 tools/megatron_hygon_bootstrap.py 的进程内垫片同源）。

为什么需要它：checkpoint 保存（ckpt-format torch_dist + fully-parallel-save）
会经 filesystem_async.py 用 multiprocessing **spawn** 上下文起子进程
（Manager + 写盘 worker）。spawn 子进程在 _fixup_main_from_path 里重新
import pretrain_gpt.py，不经过 run_megatron_hygon.py → megatron_hygon_bootstrap，
于是 Python 3.11 上 `from typing import override`（hybrid.py:5）直接
ImportError，子进程全灭，主进程随后在 ctx.Manager().Queue() 处 EOFError
（2026-10-09 run-8 双机 16 卡实测踩中，16 rank 全体崩溃在 iteration 20 保存时）。

生效机制：scripts/activate_megatron_hygon.sh 已把本目录(tools/)放进
PYTHONPATH；Python 启动时 site 模块会自动 import sitecustomize，且环境
变量被子进程继承——因此训练主进程、Manager、写盘 worker 全部覆盖。

安全性：幂等（3.12+ 环境或已补齐时 hasattr 直接跳过）、永不抛错；
不影响生成期命令（其 PYTHONPATH 只含仓库根，不含 tools/）。
"""
import typing

if not hasattr(typing, "override"):
    try:
        from typing_extensions import override as _override  # torch 自带该依赖
    except ImportError:  # 极端兜底：恒等装饰器，与 bootstrap 垫片行为一致

        def _override(method):
            return method

    typing.override = _override
