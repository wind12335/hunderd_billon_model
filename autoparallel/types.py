"""自动并行框架内部数据结构。"""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

from megatron_mvp.types import ParallelPlan


@dataclass(frozen=True)
class ParallelSpec:
    """描述最终选择的 Megatron 后端并行规格。"""

    parallel_mode: str
    execution_mode: str
    world_size: int
    tp_size: int
    dp_size: int
    pp_size: int
    micro_batch_size: int
    global_batch_size: int
    sequence_parallel: bool

    def to_dict(self) -> dict[str, Any]:
        """转成可序列化字典。"""
        return asdict(self)


@dataclass(frozen=True)
class AutoParallelPlan:
    """描述自动并行编译后的执行计划。"""

    model_path: Path
    output_dir: Path
    parallel_plan: ParallelPlan
    model_summary: dict[str, Any]
    runtime_summary: dict[str, Any]
    search_summary: dict[str, Any]
    coarse_ir: dict[str, Any]
    megatron_config: dict[str, Any]
    launch_script: str
    generated_files: dict[str, str]
    notes: list[str]

    @property
    def parallel_spec(self) -> ParallelSpec:
        """为 Megatron 下放保留后端规格视图。"""
        return ParallelSpec(
            parallel_mode=self.parallel_plan.parallel_mode,
            execution_mode=self.parallel_plan.execution_mode,
            world_size=self.parallel_plan.world_size,
            tp_size=self.parallel_plan.tp,
            dp_size=self.parallel_plan.dp,
            pp_size=self.parallel_plan.pp,
            micro_batch_size=self.parallel_plan.micro_batch_size,
            global_batch_size=self.parallel_plan.global_batch_size,
            sequence_parallel=self.parallel_plan.sequence_parallel,
        )

    def to_dict(self) -> dict[str, Any]:
        """转成可序列化字典。"""
        return {
            "model_path": str(self.model_path),
            "output_dir": str(self.output_dir),
            "parallel_plan": self.parallel_plan.to_dict(),
            "parallel_spec": self.parallel_spec.to_dict(),
            "model_summary": self.model_summary,
            "runtime_summary": self.runtime_summary,
            "search_summary": self.search_summary,
            "coarse_ir": self.coarse_ir,
            "megatron_config": self.megatron_config,
            "launch_script": self.launch_script,
            "generated_files": self.generated_files,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class LoweredProgram:
    """描述下放到指定 backend 后的可执行程序。"""

    backend: str
    project_root: Path
    output_dir: Path
    launch_script_path: Path
    runtime_config_path: Path
    planning_launch_script_path: Path
    planning_config_path: Path
    manual_example_path: Path
    declarative_example_path: Path
    command: list[str]
    env_overrides: dict[str, str]
    code_reduction_summary: dict[str, Any]
    notes: list[str]
    extra_artifact_paths: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转成可序列化字典。"""
        return {
            "backend": self.backend,
            "project_root": str(self.project_root),
            "output_dir": str(self.output_dir),
            "launch_script_path": str(self.launch_script_path),
            "runtime_config_path": str(self.runtime_config_path),
            "planning_launch_script_path": str(self.planning_launch_script_path),
            "planning_config_path": str(self.planning_config_path),
            "manual_example_path": str(self.manual_example_path),
            "declarative_example_path": str(self.declarative_example_path),
            "command": list(self.command),
            "env_overrides": dict(self.env_overrides),
            "code_reduction_summary": self.code_reduction_summary,
            "extra_artifact_paths": dict(self.extra_artifact_paths),
            "notes": list(self.notes),
        }
