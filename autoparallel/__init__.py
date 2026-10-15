"""声明式自动并行框架入口。"""

from autoparallel.config import Config
from autoparallel.decorator import parallelize
from autoparallel.launcher import launch_lowered_program
from autoparallel.lowering import lower_to_backend
from autoparallel.lowering import lower_to_megatron
from autoparallel.lowering import lower_to_pytorch
from autoparallel.runtime import AutoParallelRuntime
from autoparallel.types import AutoParallelPlan
from autoparallel.types import LoweredProgram
from autoparallel.types import ParallelSpec
from megatron_mvp.types import ParallelPlan
from megatron_mvp.search import available_search_strategies
from megatron_mvp.search import register_search_strategy

__all__ = [
    "AutoParallelPlan",
    "AutoParallelRuntime",
    "Config",
    "LoweredProgram",
    "ParallelPlan",
    "ParallelSpec",
    "available_search_strategies",
    "launch_lowered_program",
    "lower_to_backend",
    "lower_to_megatron",
    "lower_to_pytorch",
    "parallelize",
    "register_search_strategy",
]
