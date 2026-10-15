"""自动并行运行时。"""

from __future__ import annotations

import contextlib
import contextvars
from typing import Any

from autoparallel.compiler import compile_plan
from autoparallel.config import Config
from autoparallel.config import coerce_config
from autoparallel.launcher import launch_lowered_program
from autoparallel.lowering import lower_to_backend
from autoparallel.types import AutoParallelPlan
from autoparallel.types import LoweredProgram
from megatron_mvp.types import ParallelPlan


_CURRENT_RUNTIME: contextvars.ContextVar["AutoParallelRuntime | None"] = contextvars.ContextVar(
    "autoparallel_runtime",
    default=None,
)


def current_runtime() -> "AutoParallelRuntime | None":
    """返回当前激活的自动并行运行时。"""
    return _CURRENT_RUNTIME.get()


class AutoParallelRuntime:
    """声明式自动并行运行时。

    当前运行时只负责三件事：
    1. 编译自动并行计划。
    2. 把计划下放到选定 backend。
    3. 按需真正启动生成后的程序。
    """

    def __init__(self, config: Config | ParallelPlan) -> None:
        self.config = coerce_config(config).normalized()
        self.plan: AutoParallelPlan | None = None
        self.lowered_program: LoweredProgram | None = None
        self._token: contextvars.Token[Any] | None = None

    def compile(self) -> AutoParallelPlan:
        """编译自动并行计划。"""
        if self.plan is None:
            self.plan = compile_plan(self.config)
        return self.plan

    def lower(self) -> LoweredProgram:
        """把计划下放到选定 backend。"""
        if self.lowered_program is None:
            self.lowered_program = lower_to_backend(self.config, self.compile())
        return self.lowered_program

    def launch(self):
        """执行已经下放好的 Megatron 程序。"""
        return launch_lowered_program(self.lower())

    @contextlib.contextmanager
    def activate(self) -> "AutoParallelRuntime":
        """激活运行时上下文。"""
        if self._token is None:
            self._token = _CURRENT_RUNTIME.set(self)
        try:
            yield self
        finally:
            if self._token is not None:
                _CURRENT_RUNTIME.reset(self._token)
                self._token = None
