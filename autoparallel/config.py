"""自动并行框架的声明式配置。"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

from megatron_mvp.types import ParallelPlan


@dataclass
class Config:
    """描述自动并行框架的最小声明式配置。

    这层配置显式区分两种设备语义：
    1. 搜索规划视角：按未来目标卡数规划并行方案。
    2. 轻量验证视角：按当前可用卡数做 smoke test。

    同时支持两种用户入口：
    1. `Config(...)` 触发自动搜索。
    2. `Config(explicit_plan=ParallelPlan(...))` 直接指定显式并行方案。
    """

    model_path: str | Path = "."
    num_devices: int | None = None
    search_target_devices: int | None = None
    runtime_test_devices: int | None = 1
    memory_target_gb: float | None = None
    prefer: str = "memory"
    parallel_mode: str = "AUTO"
    search_strategy: str = "BLOCK_LEVEL_FAST"
    search_policy: str = "BALANCED"
    sequence_length: int | None = None
    # 搜索上限提升到 128，覆盖 16 节点 × 8 卡 BW1000 集群。
    search_max_world_size: int = 128
    # 流水线并行搜索上限：默认 1 保持旧的 DP+TP 行为；
    # 多机大模型场景（如 GLM-4.5-Air 105B）建议设为 8 或 16。
    search_max_pp: int = 1
    # MoE 专家并行搜索上限：只对 MoE 模型生效，默认 1。
    search_max_ep: int = 1
    # 多机集群拓扑：用于生成正确的 GPUS_PER_NODE 默认值和多机启动说明。
    num_nodes: int = 1
    gpus_per_node: int | None = None
    master_addr: str | None = None
    master_port: int | None = None
    output_dir: str | Path | None = None
    generate_artifacts: bool = True
    backend: str = "MEGATRON"
    run_mode: str = "PLAN"
    entrypoint: str = "Megatron-LM/pretrain_gpt.py"
    data_path: str | Path | None = None
    tokenizer_path: str | Path | None = None
    data_cache_path: str | Path | None = None
    use_mock_data: bool = False
    train_iters: int = 2
    lr: float = 1.0e-4
    min_lr: float = 1.0e-5
    weight_decay: float = 0.1
    log_interval: int = 1
    save_interval: int = 1000
    eval_interval: int = 0
    eval_iters: int = 0
    explicit_plan: ParallelPlan | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def resolved_model_path(self) -> Path:
        """返回规范化后的模型路径。"""
        return Path(self.model_path).expanduser().resolve()

    def resolved_output_dir(self) -> Path:
        """返回规范化后的输出目录。"""
        if self.output_dir is None:
            return Path.cwd() / "generated" / "autoparallel"
        return Path(self.output_dir).expanduser().resolve()

    def resolved_data_path(self) -> Path | None:
        """返回规范化后的数据集路径。"""
        if self.data_path is None:
            return None
        return Path(self.data_path).expanduser().resolve()

    def resolved_tokenizer_path(self) -> Path | None:
        """返回规范化后的 tokenizer 路径。"""
        if self.tokenizer_path is None:
            return None
        return Path(self.tokenizer_path).expanduser().resolve()

    def resolved_data_cache_path(self) -> Path:
        """返回规范化后的数据缓存路径。"""
        if self.data_cache_path is None:
            return self.resolved_output_dir() / "data_cache"
        return Path(self.data_cache_path).expanduser().resolve()

    def resolved_search_target_devices(self, detected_device_count: int | None = None) -> int:
        """返回规划阶段使用的目标卡数。"""
        if self.search_target_devices is not None:
            return max(self.search_target_devices, 1)
        if self.num_devices is not None:
            return max(self.num_devices, 1)
        if detected_device_count is not None:
            return max(detected_device_count, 1)
        return 1

    def resolved_num_nodes(self) -> int:
        """返回规划用节点数。"""
        return max(int(self.num_nodes), 1)

    def resolved_gpus_per_node(self) -> int | None:
        """返回显式给定的每节点卡数（None 表示由 world_size/num_nodes 推导）。"""
        if self.gpus_per_node is None:
            return None
        return max(int(self.gpus_per_node), 1)

    def resolved_runtime_test_devices(self) -> int:
        """返回当前 smoke test 使用的卡数。"""
        if self.runtime_test_devices is None:
            return 1
        return max(self.runtime_test_devices, 1)

    def has_explicit_plan(self) -> bool:
        """返回当前配置是否显式指定了并行方案。"""
        return self.explicit_plan is not None

    def with_updates(self, **kwargs: Any) -> "Config":
        """返回带增量修改的新配置。"""
        payload = self.__dict__.copy()
        payload["extra"] = dict(self.extra)
        payload.update(kwargs)
        return Config(**payload)

    def normalized(self) -> "Config":
        """返回规范化后的配置副本。"""
        resolved_model_path = self.resolved_model_path()
        normalized_plan = None
        if self.explicit_plan is not None:
            normalized_plan = self.explicit_plan.normalized(default_model_path=resolved_model_path)

        return Config(
            model_path=str(resolved_model_path),
            num_devices=self.num_devices,
            search_target_devices=self.search_target_devices,
            runtime_test_devices=self.resolved_runtime_test_devices(),
            memory_target_gb=self.memory_target_gb,
            prefer=self.prefer.upper(),
            parallel_mode=self.parallel_mode.upper(),
            search_strategy=self.search_strategy.upper(),
            search_policy=self.search_policy.upper(),
            sequence_length=self.sequence_length,
            search_max_world_size=self.search_max_world_size,
            search_max_pp=max(int(self.search_max_pp), 1),
            search_max_ep=max(int(self.search_max_ep), 1),
            num_nodes=self.resolved_num_nodes(),
            gpus_per_node=self.resolved_gpus_per_node(),
            master_addr=self.master_addr,
            master_port=self.master_port,
            output_dir=str(self.resolved_output_dir()),
            generate_artifacts=self.generate_artifacts,
            backend=self.backend.upper(),
            run_mode=self.run_mode.upper(),
            entrypoint=self.entrypoint,
            data_path=str(self.resolved_data_path()) if self.data_path is not None else None,
            tokenizer_path=(
                str(self.resolved_tokenizer_path())
                if self.tokenizer_path is not None
                else None
            ),
            data_cache_path=str(self.resolved_data_cache_path()),
            use_mock_data=self.use_mock_data,
            train_iters=self.train_iters,
            lr=self.lr,
            min_lr=self.min_lr,
            weight_decay=self.weight_decay,
            log_interval=self.log_interval,
            save_interval=self.save_interval,
            eval_interval=self.eval_interval,
            eval_iters=self.eval_iters,
            explicit_plan=normalized_plan,
            extra=dict(self.extra),
        )


def coerce_config(config_or_plan: Config | ParallelPlan) -> Config:
    """把用户传入的 `Config | ParallelPlan` 统一成 Config。"""
    if isinstance(config_or_plan, Config):
        return config_or_plan

    resolved_model_path = config_or_plan.resolved_model_path(default_path=".")
    return Config(
        model_path=str(resolved_model_path) if resolved_model_path is not None else ".",
        parallel_mode=config_or_plan.parallel_mode,
        explicit_plan=config_or_plan,
    )
