"""MVP 工具链的数据结构定义。"""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelArtifacts:
    """描述 Hugging Face 模型目录中的关键文件。"""

    model_dir: Path
    config_path: Path
    weight_files: list[Path]
    tokenizer_files: list[Path]
    shard_count: int
    weight_format: str

    def to_dict(self) -> dict[str, Any]:
        """转成可序列化字典。"""
        return {
            "model_dir": str(self.model_dir),
            "config_path": str(self.config_path),
            "weight_files": [str(path) for path in self.weight_files],
            "tokenizer_files": [str(path) for path in self.tokenizer_files],
            "shard_count": self.shard_count,
            "weight_format": self.weight_format,
        }


@dataclass(frozen=True)
class ModelSpec:
    """描述模型结构规格。"""

    model_dir: Path
    model_type: str
    architecture: str
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    vocab_size: int
    max_seq_len: int
    rope_theta: float
    rms_norm_eps: float
    tie_word_embeddings: bool
    source_torch_dtype: str
    estimated_total_params: int
    estimated_params_per_layer: int
    # MoE 相关字段：0 / None 表示稠密模型（LLaMA 风格）。
    # GLM-4.5-Air (glm4_moe)、DeepSeek、Qwen3-MoE 等会填充这些字段。
    head_dim: int | None = None
    partial_rotary_factor: float | None = None
    num_experts: int = 0
    num_experts_per_tok: int = 0
    moe_intermediate_size: int | None = None
    shared_expert_intermediate_size: int | None = None
    first_k_dense_replace: int = 0
    num_nextn_predict_layers: int = 0

    @property
    def is_moe(self) -> bool:
        """返回是否为 MoE 模型。"""
        return self.num_experts > 0

    def to_dict(self) -> dict[str, Any]:
        """转成可序列化字典。"""
        payload = asdict(self)
        payload["model_dir"] = str(self.model_dir)
        return payload


@dataclass(frozen=True)
class DeviceInfo:
    """描述单张设备的硬件信息。"""

    index: int
    name: str
    total_memory_bytes: int
    multi_processor_count: int | None

    def to_dict(self) -> dict[str, Any]:
        """转成可序列化字典。"""
        return asdict(self)


@dataclass(frozen=True)
class RuntimeEnvironment:
    """描述当前运行环境。"""

    accelerator_type: str
    device_count: int
    devices: list[DeviceInfo]
    primary_device_name: str
    primary_device_memory_bytes: int
    torch_version: str
    hip_version: str | None

    def to_dict(self) -> dict[str, Any]:
        """转成可序列化字典。"""
        return {
            "accelerator_type": self.accelerator_type,
            "device_count": self.device_count,
            "devices": [device.to_dict() for device in self.devices],
            "primary_device_name": self.primary_device_name,
            "primary_device_memory_bytes": self.primary_device_memory_bytes,
            "torch_version": self.torch_version,
            "hip_version": self.hip_version,
        }


@dataclass(frozen=True)
class MemoryEstimate:
    """描述 block-level 粗显存估算。"""

    weights_bytes: int
    grads_bytes: int
    optimizer_bytes: int
    activation_bytes: int
    total_bytes: int

    def to_dict(self) -> dict[str, Any]:
        """转成可序列化字典。"""
        return asdict(self)


@dataclass(frozen=True)
class ParallelPlan:
    """描述用户可见的显式并行方案。

    这是自动搜索和显式指定共用的一等对象。
    用户看到的应该始终是这个结构，而不是内部搜索候选。
    """

    model_path: str | Path | None = None
    tp: int = 1
    dp: int = 1
    pp: int = 1
    # 专家并行维度：EP 只细分 DP（要求 ep | dp），不改变 world_size 语义。
    # Megatron 语义: world_size = tp * dp * pp，EP 在 DP 组内部再切分专家。
    ep: int = 1
    attention_policy: str = "replicate"
    mlp_policy: str = "replicate"
    attention_policy_front: str | None = None
    attention_policy_mid: str | None = None
    attention_policy_back: str | None = None
    mlp_policy_front: str | None = None
    mlp_policy_mid: str | None = None
    mlp_policy_back: str | None = None
    embedding_policy: str = "replicate"
    lm_head_policy: str = "replicate"
    sequence_parallel: bool = False
    micro_batch_size: int = 1
    global_batch_size: int = 1
    parallel_mode: str = "MODEL_PARALLEL"
    execution_mode: str = "manual"
    strategy_name: str = "MANUAL"
    strategy_policy: str = "USER_DEFINED"
    search_time_seconds: float = 0.0
    estimated_memory_bytes: int | None = None
    memory_margin_bytes: int | None = None
    fits_current_environment: bool = True
    requires_additional_devices: int = 0
    megatron_lowerable: bool | None = None
    notes: tuple[str, ...] = ()

    @property
    def world_size(self) -> int:
        """返回总并行规模。"""
        return max(self.tp, 1) * max(self.dp, 1) * max(self.pp, 1)

    @property
    def mesh(self) -> tuple[int, int]:
        """返回 `(tp, dp)` 形式的全局 mesh。"""
        return (max(self.tp, 1), max(self.dp, 1))

    def resolved_attention_policy_front(self) -> str:
        """返回前层组 attention 策略。"""
        return self.attention_policy_front or self.attention_policy

    def resolved_attention_policy_mid(self) -> str:
        """返回中层组 attention 策略。"""
        return self.attention_policy_mid or self.attention_policy

    def resolved_attention_policy_back(self) -> str:
        """返回后层组 attention 策略。"""
        return self.attention_policy_back or self.attention_policy

    def resolved_mlp_policy_front(self) -> str:
        """返回前层组 MLP 策略。"""
        return self.mlp_policy_front or self.mlp_policy

    def resolved_mlp_policy_mid(self) -> str:
        """返回中层组 MLP 策略。"""
        return self.mlp_policy_mid or self.mlp_policy

    def resolved_mlp_policy_back(self) -> str:
        """返回后层组 MLP 策略。"""
        return self.mlp_policy_back or self.mlp_policy

    def grouped_attention_policies(self) -> dict[str, str]:
        """返回三段式 attention 策略。"""
        return {
            "front": self.resolved_attention_policy_front(),
            "mid": self.resolved_attention_policy_mid(),
            "back": self.resolved_attention_policy_back(),
        }

    def grouped_mlp_policies(self) -> dict[str, str]:
        """返回三段式 MLP 策略。"""
        return {
            "front": self.resolved_mlp_policy_front(),
            "mid": self.resolved_mlp_policy_mid(),
            "back": self.resolved_mlp_policy_back(),
        }

    def has_uniform_group_policies(self) -> bool:
        """返回三段式策略是否退化为统一策略。"""
        attention_policies = set(self.grouped_attention_policies().values())
        mlp_policies = set(self.grouped_mlp_policies().values())
        return len(attention_policies) == 1 and len(mlp_policies) == 1

    def resolved_model_path(self, default_path: str | Path | None = None) -> Path | None:
        """返回规范化后的模型路径。"""
        candidate = self.model_path if self.model_path is not None else default_path
        if candidate is None:
            return None
        return Path(candidate).expanduser().resolve()

    def normalized(self, default_model_path: str | Path | None = None) -> "ParallelPlan":
        """返回规范化后的并行方案副本。"""
        resolved_model_path = self.resolved_model_path(default_model_path)
        return ParallelPlan(
            model_path=str(resolved_model_path) if resolved_model_path is not None else None,
            tp=max(int(self.tp), 1),
            dp=max(int(self.dp), 1),
            pp=max(int(self.pp), 1),
            ep=max(int(self.ep), 1),
            attention_policy=str(self.attention_policy),
            mlp_policy=str(self.mlp_policy),
            attention_policy_front=(
                str(self.attention_policy_front)
                if self.attention_policy_front is not None
                else None
            ),
            attention_policy_mid=(
                str(self.attention_policy_mid)
                if self.attention_policy_mid is not None
                else None
            ),
            attention_policy_back=(
                str(self.attention_policy_back)
                if self.attention_policy_back is not None
                else None
            ),
            mlp_policy_front=(
                str(self.mlp_policy_front)
                if self.mlp_policy_front is not None
                else None
            ),
            mlp_policy_mid=(
                str(self.mlp_policy_mid)
                if self.mlp_policy_mid is not None
                else None
            ),
            mlp_policy_back=(
                str(self.mlp_policy_back)
                if self.mlp_policy_back is not None
                else None
            ),
            embedding_policy=str(self.embedding_policy),
            lm_head_policy=str(self.lm_head_policy),
            sequence_parallel=bool(self.sequence_parallel),
            micro_batch_size=max(int(self.micro_batch_size), 1),
            global_batch_size=max(int(self.global_batch_size), 1),
            parallel_mode=str(self.parallel_mode).upper(),
            execution_mode=str(self.execution_mode),
            strategy_name=str(self.strategy_name).upper(),
            strategy_policy=str(self.strategy_policy).upper(),
            search_time_seconds=float(self.search_time_seconds),
            estimated_memory_bytes=self.estimated_memory_bytes,
            memory_margin_bytes=self.memory_margin_bytes,
            fits_current_environment=bool(self.fits_current_environment),
            requires_additional_devices=max(int(self.requires_additional_devices), 0),
            megatron_lowerable=self.megatron_lowerable,
            notes=tuple(str(note) for note in self.notes),
        )

    def with_updates(self, **kwargs: Any) -> "ParallelPlan":
        """返回带增量修改的新并行方案。"""
        return replace(self, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        """转成可序列化字典。"""
        return {
            "model_path": (
                str(self.model_path)
                if self.model_path is not None
                else None
            ),
            "tp": self.tp,
            "dp": self.dp,
            "pp": self.pp,
            "ep": self.ep,
            "mesh": {
                "tp": self.mesh[0],
                "dp": self.mesh[1],
            },
            "world_size": self.world_size,
            "attention_policy": self.attention_policy,
            "mlp_policy": self.mlp_policy,
            "attention_policy_front": self.resolved_attention_policy_front(),
            "attention_policy_mid": self.resolved_attention_policy_mid(),
            "attention_policy_back": self.resolved_attention_policy_back(),
            "mlp_policy_front": self.resolved_mlp_policy_front(),
            "mlp_policy_mid": self.resolved_mlp_policy_mid(),
            "mlp_policy_back": self.resolved_mlp_policy_back(),
            "embedding_policy": self.embedding_policy,
            "lm_head_policy": self.lm_head_policy,
            "sequence_parallel": self.sequence_parallel,
            "micro_batch_size": self.micro_batch_size,
            "global_batch_size": self.global_batch_size,
            "parallel_mode": self.parallel_mode,
            "execution_mode": self.execution_mode,
            "strategy_name": self.strategy_name,
            "strategy_policy": self.strategy_policy,
            "search_time_seconds": self.search_time_seconds,
            "estimated_memory_bytes": self.estimated_memory_bytes,
            "memory_margin_bytes": self.memory_margin_bytes,
            "fits_current_environment": self.fits_current_environment,
            "requires_additional_devices": self.requires_additional_devices,
            "megatron_lowerable": self.megatron_lowerable,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class SearchCandidate:
    """描述一个 DP/TP 候选方案。

    内部搜索仍可保留这个结构，但它必须始终携带显式 ParallelPlan。
    """

    world_size: int
    tp_size: int
    dp_size: int
    micro_batch_size: int
    global_batch_size: int
    seq_length: int
    estimated_memory: MemoryEstimate
    memory_margin_bytes: int
    fits_current_environment: bool
    requires_additional_devices: int
    parallel_plan: ParallelPlan
    score: tuple[int, ...] = field(repr=False)

    def to_dict(self) -> dict[str, Any]:
        """转成可序列化字典。"""
        return {
            "world_size": self.world_size,
            "tp_size": self.tp_size,
            "dp_size": self.dp_size,
            "micro_batch_size": self.micro_batch_size,
            "global_batch_size": self.global_batch_size,
            "seq_length": self.seq_length,
            "estimated_memory": self.estimated_memory.to_dict(),
            "memory_margin_bytes": self.memory_margin_bytes,
            "fits_current_environment": self.fits_current_environment,
            "requires_additional_devices": self.requires_additional_devices,
            "parallel_plan": self.parallel_plan.to_dict(),
        }


@dataclass(frozen=True)
class SearchBenchmark:
    """描述一次搜索策略执行的统计信息。"""

    strategy_name: str
    strategy_policy: str
    elapsed_seconds: float
    feasible_current_count: int
    scalable_future_count: int

    def to_dict(self) -> dict[str, Any]:
        """转成可序列化字典。"""
        return asdict(self)


@dataclass(frozen=True)
class SearchResult:
    """描述粗搜索整体结果。"""

    strategy_name: str
    strategy_policy: str
    planning_device_count: int
    search_time_seconds: float
    execution_mode: str | None
    selected_plan: ParallelPlan | None
    selected_candidate: SearchCandidate | None
    feasible_current_candidates: list[SearchCandidate]
    scalable_future_candidates: list[SearchCandidate]
    feasible_current_plans: list[ParallelPlan]
    scalable_future_plans: list[ParallelPlan]
    benchmarks: list[SearchBenchmark]
    rejected_reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        """转成可序列化字典。"""
        return {
            "strategy_name": self.strategy_name,
            "strategy_policy": self.strategy_policy,
            "planning_device_count": self.planning_device_count,
            "search_time_seconds": self.search_time_seconds,
            "execution_mode": self.execution_mode,
            "selected_plan": (
                self.selected_plan.to_dict() if self.selected_plan is not None else None
            ),
            "selected_candidate": (
                self.selected_candidate.to_dict()
                if self.selected_candidate is not None
                else None
            ),
            "feasible_current_candidates": [
                candidate.to_dict() for candidate in self.feasible_current_candidates
            ],
            "scalable_future_candidates": [
                candidate.to_dict() for candidate in self.scalable_future_candidates
            ],
            "feasible_current_plans": [
                plan.to_dict() for plan in self.feasible_current_plans
            ],
            "scalable_future_plans": [
                plan.to_dict() for plan in self.scalable_future_plans
            ],
            "benchmarks": [benchmark.to_dict() for benchmark in self.benchmarks],
            "rejected_reasons": list(self.rejected_reasons),
        }


@dataclass(frozen=True)
class CoarseIR:
    """描述 MVP 阶段的粗 IR。"""

    style: str
    execution_mode: str
    model_dir: Path
    model_type: str
    architecture: str
    parallelism: dict[str, Any]
    parallel_plan: dict[str, Any]
    train: dict[str, Any]
    environment: dict[str, Any]
    search: dict[str, Any]
    model: dict[str, Any]
    search_notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        """转成可序列化字典。"""
        return {
            "style": self.style,
            "execution_mode": self.execution_mode,
            "model_dir": str(self.model_dir),
            "model_type": self.model_type,
            "architecture": self.architecture,
            "parallelism": self.parallelism,
            "parallel_plan": self.parallel_plan,
            "train": self.train,
            "environment": self.environment,
            "search": self.search,
            "model": self.model,
            "search_notes": self.search_notes,
        }


@dataclass(frozen=True)
class GenerationOutputs:
    """描述生成物路径。"""

    output_dir: Path
    detection_summary_path: Path
    search_result_path: Path
    coarse_ir_path: Path
    megatron_config_path: Path
    launch_script_path: Path

    def to_dict(self) -> dict[str, Any]:
        """转成可序列化字典。"""
        return {
            "output_dir": str(self.output_dir),
            "detection_summary_path": str(self.detection_summary_path),
            "search_result_path": str(self.search_result_path),
            "coarse_ir_path": str(self.coarse_ir_path),
            "megatron_config_path": str(self.megatron_config_path),
            "launch_script_path": str(self.launch_script_path),
        }
