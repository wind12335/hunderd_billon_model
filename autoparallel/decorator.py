"""自动并行 decorator。"""

from __future__ import annotations

import functools
from typing import Any
from typing import Callable
from typing import TypeVar

from autoparallel.config import Config
from autoparallel.config import coerce_config
from autoparallel.runtime import AutoParallelRuntime
from megatron_mvp.types import ParallelPlan


_F = TypeVar("_F", bound=Callable[..., Any])


def parallelize(config: Config | ParallelPlan) -> Callable[[_F], _F]:
    """把普通训练入口提升为声明式自动并行入口。

    注意：当前 decorator 的语义不是“在原 Python 函数体里直接做分布式改写”，
    而是把这个函数视为训练意图边界，再自动下放到选定 backend 执行。
    """
    normalized = coerce_config(config).normalized()

    def decorator(fn: _F) -> _F:
        runtime = AutoParallelRuntime(normalized)

        @functools.wraps(fn)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            """按运行模式决定返回计划、下放程序或直接启动。"""
            with runtime.activate():
                if normalized.run_mode == "PLAN":
                    return runtime.compile()
                if normalized.run_mode == "LOWER":
                    return runtime.lower()
                if normalized.run_mode == "LAUNCH":
                    return runtime.launch()
                raise ValueError(f"不支持的 run_mode: {normalized.run_mode}")

        setattr(wrapped, "autoparallel_config", normalized)
        setattr(wrapped, "compile_plan", runtime.compile)
        setattr(wrapped, "lower_program", runtime.lower)
        setattr(wrapped, "launch_program", runtime.launch)
        setattr(wrapped, "autoparallel_runtime", runtime)
        return wrapped  # type: ignore[return-value]

    return decorator
