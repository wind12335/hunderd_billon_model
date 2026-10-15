"""DP/TP/PP 粗搜索与策略注册实现。"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from dataclasses import replace
from typing import Callable

from megatron_mvp.ppo_search import ControllerEvaluation
from megatron_mvp.ppo_search import DecisionNode
from megatron_mvp.ppo_search import run_rl_guided_ppo_search
from megatron_mvp.types import MemoryEstimate
from megatron_mvp.types import ModelSpec
from megatron_mvp.types import ParallelPlan
from megatron_mvp.types import RuntimeEnvironment
from megatron_mvp.types import SearchBenchmark
from megatron_mvp.types import SearchCandidate
from megatron_mvp.types import SearchResult


_BYTES_PER_BF16 = 2
_ADAM_STATE_BYTES = 12
_ACTIVATION_MULTIPLIER = 10
_RECOMPUTE_RATIO = 0.35
_MEMORY_UTILIZATION = 0.92
_DEFAULT_MICRO_BATCHES = (1, 2, 4)
# 多机版本：搜索上限从 16 提到 128，覆盖 16 节点 × 8 卡的 BW1000 集群。
_SEARCH_MAX_WORLD_SIZE = 128
_CURRENT_LIMIT = 16
_FUTURE_LIMIT = 32
_REJECT_LIMIT = 64
_BUILTIN_STRATEGIES = ("BLOCK_LEVEL_FAST", "HIERARCHICAL", "RL_GUIDED_PPO")
_HEURISTIC_BENCHMARK_STRATEGIES = ("BLOCK_LEVEL_FAST", "HIERARCHICAL")
_SUPPORTED_POLICIES = {
    "BALANCED",
    "MEMORY_FIRST",
    "THROUGHPUT_FIRST",
    "TP_FIRST",
    "DP_FIRST",
}
_EMBEDDING_POLICIES = ("replicate", "vocab_shard")
_STANDARD_TP_ATTENTION_POLICY = "head_shard"
_STANDARD_TP_MLP_POLICY = "colrow_shard"
_STANDARD_TP_LM_HEAD_POLICY = "vocab_shard"
# PP 候选除 num_layers 因子外，额外纳入的 2 的幂模板。
# Megatron 支持不均匀 stage 切分（自动尽量均分），因此 PP 不必整除 num_layers。
_EXTRA_PP_TEMPLATE = (2, 4, 8, 16)

StrategyOutput = tuple[list[SearchCandidate], list[SearchCandidate], list[str]]
SearchStrategyFn = Callable[..., StrategyOutput]
_SEARCH_STRATEGY_REGISTRY: dict[str, SearchStrategyFn] = {}


@dataclass(frozen=True)
class MeshChoice:
    """描述一个 `(tp, dp, pp)` 全局 mesh 候选。"""

    tp_size: int
    dp_size: int
    pp_size: int = 1

    @property
    def world_size(self) -> int:
        """返回总进程数。"""
        return self.tp_size * self.dp_size * self.pp_size

    def label(self) -> str:
        """返回稳定可序列化标签。"""
        return f"tp={self.tp_size},dp={self.dp_size},pp={self.pp_size}"


@dataclass(frozen=True)
class GroupedPolicyProposal:
    """描述一次完整的分组式并行策略提案。"""

    mesh: MeshChoice
    attn_policy_front: str
    mlp_policy_front: str
    attn_policy_mid: str
    mlp_policy_mid: str
    attn_policy_back: str
    mlp_policy_back: str
    embedding_policy: str
    lm_head_policy: str
    # 专家并行维度：只对 MoE 模型有意义，且必须整除 dp_size。
    expert_parallel_size: int = 1

    def grouped_attention_policies(self) -> dict[str, str]:
        """返回三段式 attention 策略。"""
        return {
            "front": self.attn_policy_front,
            "mid": self.attn_policy_mid,
            "back": self.attn_policy_back,
        }

    def grouped_mlp_policies(self) -> dict[str, str]:
        """返回三段式 MLP 策略。"""
        return {
            "front": self.mlp_policy_front,
            "mid": self.mlp_policy_mid,
            "back": self.mlp_policy_back,
        }

    def has_uniform_attention_policy(self) -> bool:
        """返回 attention 三段策略是否统一。"""
        return len(set(self.grouped_attention_policies().values())) == 1

    def has_uniform_mlp_policy(self) -> bool:
        """返回 MLP 三段策略是否统一。"""
        return len(set(self.grouped_mlp_policies().values())) == 1

    def attention_policy_summary(self) -> str:
        """返回面向用户的 attention 策略摘要。"""
        if self.has_uniform_attention_policy():
            return self.attn_policy_mid
        return "grouped"

    def mlp_policy_summary(self) -> str:
        """返回面向用户的 MLP 策略摘要。"""
        if self.has_uniform_mlp_policy():
            return self.mlp_policy_mid
        return "grouped"

    def label(self) -> str:
        """返回提案摘要标签。"""
        ep_suffix = "" if self.expert_parallel_size <= 1 else f"|ep={self.expert_parallel_size}"
        return (
            f"{self.mesh.label()}|"
            f"attn=({self.attn_policy_front},{self.attn_policy_mid},{self.attn_policy_back})|"
            f"mlp=({self.mlp_policy_front},{self.mlp_policy_mid},{self.mlp_policy_back})|"
            f"emb={self.embedding_policy}|head={self.lm_head_policy}{ep_suffix}"
        )


@dataclass(frozen=True)
class _StrategyExecution:
    """描述一次具体策略执行后的中间结果。"""

    strategy_name: str
    strategy_policy: str
    elapsed_seconds: float
    execution_mode: str | None
    selected_candidate: SearchCandidate | None
    selected_plan: ParallelPlan | None
    feasible_current_candidates: list[SearchCandidate]
    scalable_future_candidates: list[SearchCandidate]
    rejected_reasons: list[str]
    benchmark: SearchBenchmark


def _divisors(value: int) -> list[int]:
    """返回正整数因子列表。"""
    divisors: set[int] = set()
    for candidate in range(1, int(math.sqrt(value)) + 1):
        if value % candidate != 0:
            continue
        divisors.add(candidate)
        divisors.add(value // candidate)
    return sorted(divisors)


def _normalize_strategy(strategy: str | None) -> str:
    """规范化策略名。"""
    normalized = (strategy or "BLOCK_LEVEL_FAST").upper()
    if normalized == "AUTO":
        return "BLOCK_LEVEL_FAST"
    return normalized


def _normalize_policy(policy: str | None) -> str:
    """规范化策略偏好。"""
    normalized = (policy or "BALANCED").upper()
    if normalized == "DEFAULT":
        normalized = "BALANCED"
    if normalized not in _SUPPORTED_POLICIES:
        raise ValueError(
            f"不支持的 search_policy: {normalized}，可选值为 {sorted(_SUPPORTED_POLICIES)}"
        )
    return normalized


def available_search_strategies() -> list[str]:
    """返回当前已注册的搜索策略名。"""
    return sorted(_SEARCH_STRATEGY_REGISTRY)


def register_search_strategy(name: str, strategy_fn: SearchStrategyFn) -> None:
    """注册自定义搜索策略。"""
    normalized = _normalize_strategy(name)
    _SEARCH_STRATEGY_REGISTRY[normalized] = strategy_fn


def _valid_tp_sizes(spec: ModelSpec, max_world_size: int) -> list[int]:
    """枚举合法 TP 值。"""
    candidates = []
    for tp_size in _divisors(spec.hidden_size):
        if tp_size > max_world_size:
            continue
        if spec.num_heads % tp_size != 0:
            continue
        if spec.num_kv_heads % tp_size != 0:
            continue
        candidates.append(tp_size)
    return candidates


def _valid_pp_sizes(spec: ModelSpec, max_world_size: int, max_pp: int) -> list[int]:
    """枚举合法 PP 候选值。

    1. num_layers 的因子保证每个 stage 层数完全均匀。
    2. 额外纳入常用 2 的幂模板：Megatron 支持不均匀 stage 切分
       （自动按尽量均分处理），GLM5-300B 官方示例 45 层用 PP=8 就是这种情况。
    """
    upper = max(1, min(int(max_pp), spec.num_layers, max_world_size))
    candidates = set(_divisors(spec.num_layers)) | set(_EXTRA_PP_TEMPLATE)
    return sorted(pp for pp in candidates if 1 <= pp <= upper)


def _build_world_size_choices(
    planning_device_count: int,
    max_world_size: int,
) -> list[int]:
    """构造 world size 搜索候选。

    设计原则：
    1. 当前卡数先纳入搜索，保证存在当前可行候选。
    2. 再纳入一个更大的未来规模，保留未来扩卡规划能力。
    3. 若 4 卡可用，则保留 4 卡模板，兼容 `(1,4)/(2,2)/(4,1)` 这类典型 mesh。
    """
    world_sizes = {max(1, min(planning_device_count, max_world_size))}
    if max_world_size >= 2:
        world_sizes.add(min(max_world_size, max(2, planning_device_count)))
    if max_world_size >= 4:
        world_sizes.add(min(max_world_size, max(4, planning_device_count * 2)))
    if max_world_size >= 8:
        world_sizes.add(min(max_world_size, max(8, planning_device_count * 4)))
    return sorted(world_size for world_size in world_sizes if world_size >= 1)


def _build_mesh_choices(
    spec: ModelSpec,
    env: RuntimeEnvironment,
    max_world_size: int,
    max_pp: int = 1,
) -> list[MeshChoice]:
    """构造 `(tp, dp, pp)` 三维 mesh 搜索空间。"""
    valid_tp_sizes = _valid_tp_sizes(spec, max_world_size=max_world_size)
    valid_pp_sizes = _valid_pp_sizes(spec, max_world_size=max_world_size, max_pp=max_pp)
    world_size_choices = _build_world_size_choices(
        planning_device_count=env.device_count,
        max_world_size=max_world_size,
    )

    meshes: list[MeshChoice] = []
    for world_size in world_size_choices:
        for pp_size in valid_pp_sizes:
            if world_size % pp_size != 0:
                continue
            stage_world_size = world_size // pp_size
            for tp_size in valid_tp_sizes:
                if stage_world_size % tp_size != 0:
                    continue
                meshes.append(
                    MeshChoice(
                        tp_size=tp_size,
                        dp_size=stage_world_size // tp_size,
                        pp_size=pp_size,
                    )
                )
    return meshes


def _split_layers_into_three_groups(num_layers: int) -> dict[str, int]:
    """把层数切分为 front/mid/back 三组。"""
    base = num_layers // 3
    remain = num_layers % 3
    counts = {"front": base, "mid": base, "back": base}
    for group_name in ("front", "mid", "back")[:remain]:
        counts[group_name] += 1
    return counts


def _estimate_transformer_component_params(spec: ModelSpec) -> dict[str, int]:
    """按 LLaMA / MoE 结构估算每层 attention/MLP/norm 参数量。

    返回字段：
    - attention: 单层 attention 参数（支持 head_dim 上采样，如 GLM-4.5）。
    - dense_mlp: 稠密层 MLP 参数（SwiGLU 3 矩阵）。
    - routed_experts: 单个 MoE 层全部路由专家参数（未做 EP 切分）。
    - shared_expert: 单个 MoE 层共享专家参数。
    - norm: 单层 norm 参数。
    - moe_layers / dense_layers: MoE / 稠密层数（按 first_k_dense_replace 切分）。
    """
    head_dim = spec.head_dim
    if head_dim is None or head_dim <= 0:
        head_dim = (
            spec.hidden_size // spec.num_heads
            if spec.hidden_size % spec.num_heads == 0
            else spec.hidden_size
        )
    kv_hidden_size = spec.num_kv_heads * head_dim
    q_hidden_size = spec.num_heads * head_dim
    attention_params = (
        spec.hidden_size * q_hidden_size
        + spec.hidden_size * kv_hidden_size
        + spec.hidden_size * kv_hidden_size
        + q_hidden_size * spec.hidden_size
    )
    dense_mlp_params = spec.hidden_size * spec.intermediate_size * 3

    if spec.is_moe:
        routed_inter = spec.moe_intermediate_size or spec.intermediate_size
        routed_experts = spec.num_experts * spec.hidden_size * routed_inter * 3
        shared_expert = (
            spec.hidden_size * spec.shared_expert_intermediate_size * 3
            if spec.shared_expert_intermediate_size
            else 0
        )
        moe_layers = max(spec.num_layers - spec.first_k_dense_replace, 0)
    else:
        routed_experts = 0
        shared_expert = 0
        moe_layers = 0
    dense_layers = spec.num_layers - moe_layers
    norm_params = spec.hidden_size * 2
    return {
        "attention": attention_params,
        "dense_mlp": dense_mlp_params,
        "routed_experts": routed_experts,
        "shared_expert": shared_expert,
        "norm": norm_params,
        "moe_layers": moe_layers,
        "dense_layers": dense_layers,
    }


def _attention_shard_factor(policy: str, tp_size: int) -> int:
    """返回 attention 参数/激活分片因子。"""
    if tp_size > 1 and policy in {"head_shard", "hidden_shard"}:
        return tp_size
    return 1


def _mlp_shard_factor(policy: str, tp_size: int) -> int:
    """返回 MLP 参数/激活分片因子。"""
    if tp_size > 1 and policy in {"ffn_shard", "colrow_shard"}:
        return tp_size
    return 1


def _proposal_structural_penalty(proposal: GroupedPolicyProposal) -> float:
    """估算提案的结构复杂度惩罚。"""
    penalty = 0.0
    if not proposal.has_uniform_attention_policy():
        penalty += 8.0
    if not proposal.has_uniform_mlp_policy():
        penalty += 8.0
    penalty += 5.0 * sum(
        policy == "hidden_shard"
        for policy in proposal.grouped_attention_policies().values()
    )
    penalty += 5.0 * sum(
        policy == "ffn_shard"
        for policy in proposal.grouped_mlp_policies().values()
    )
    if proposal.mesh.tp_size == 1:
        if proposal.embedding_policy != "replicate":
            penalty += 6.0
        if proposal.lm_head_policy != "replicate":
            penalty += 6.0
    return penalty


def _common_parallel_reasons(
    spec: ModelSpec,
    tp_size: int,
    attention_policies: tuple[str, ...],
    mlp_policies: tuple[str, ...],
    embedding_policy: str,
    lm_head_policy: str,
) -> list[str]:
    """返回标准 DP+TP 公共策略域的违规原因。"""
    reasons: list[str] = []

    if tp_size == 1:
        if any(policy != "replicate" for policy in attention_policies):
            reasons.append("TP=1 时 attention 三段策略必须都是 replicate。")
        if any(policy != "replicate" for policy in mlp_policies):
            reasons.append("TP=1 时 MLP 三段策略必须都是 replicate。")
        if embedding_policy != "replicate":
            reasons.append("TP=1 时 embedding_policy 必须是 replicate。")
        if lm_head_policy != "replicate":
            reasons.append("TP=1 时 lm_head_policy 必须是 replicate。")
    else:
        if any(policy != _STANDARD_TP_ATTENTION_POLICY for policy in attention_policies):
            reasons.append(
                "TP>1 时标准 DP+TP 公共域要求 attention 三段统一为 head_shard。"
            )
        if any(policy != _STANDARD_TP_MLP_POLICY for policy in mlp_policies):
            reasons.append(
                "TP>1 时标准 DP+TP 公共域要求 MLP 三段统一为 colrow_shard。"
            )
        if lm_head_policy != _STANDARD_TP_LM_HEAD_POLICY:
            reasons.append("TP>1 时标准 DP+TP 公共域要求 lm_head_policy 为 vocab_shard。")
        if embedding_policy not in _EMBEDDING_POLICIES:
            reasons.append(
                "TP>1 时标准 DP+TP 公共域要求 embedding_policy 为 replicate 或 vocab_shard。"
            )

    if spec.tie_word_embeddings and embedding_policy != lm_head_policy:
        reasons.append("当前模型 tied embeddings 打开时，embedding_policy 必须与 lm_head_policy 一致。")

    return reasons


def validate_standard_parallel_plan(
    spec: ModelSpec,
    plan: ParallelPlan,
) -> list[str]:
    """校验显式 ParallelPlan 是否属于标准 DP+TP 公共策略域。"""
    return _common_parallel_reasons(
        spec=spec,
        tp_size=plan.tp,
        attention_policies=(
            plan.resolved_attention_policy_front(),
            plan.resolved_attention_policy_mid(),
            plan.resolved_attention_policy_back(),
        ),
        mlp_policies=(
            plan.resolved_mlp_policy_front(),
            plan.resolved_mlp_policy_mid(),
            plan.resolved_mlp_policy_back(),
        ),
        embedding_policy=plan.embedding_policy,
        lm_head_policy=plan.lm_head_policy,
    )


def _check_megatron_lowerability(
    spec: ModelSpec,
    proposal: GroupedPolicyProposal,
) -> tuple[bool, list[str]]:
    """检查提案是否落在当前标准 DP+TP 公共策略域。"""
    reasons = _common_parallel_reasons(
        spec=spec,
        tp_size=proposal.mesh.tp_size,
        attention_policies=tuple(proposal.grouped_attention_policies().values()),
        mlp_policies=tuple(proposal.grouped_mlp_policies().values()),
        embedding_policy=proposal.embedding_policy,
        lm_head_policy=proposal.lm_head_policy,
    )
    return len(reasons) == 0, reasons


def _estimate_memory(
    spec: ModelSpec,
    proposal: GroupedPolicyProposal,
    seq_length: int,
    micro_batch_size: int,
) -> MemoryEstimate:
    """估算每张卡的显存占用（TP/PP/EP/SP 感知）。"""
    tp_size = proposal.mesh.tp_size
    pp_size = max(proposal.mesh.pp_size, 1)
    ep_size = max(proposal.expert_parallel_size, 1)
    components = _estimate_transformer_component_params(spec)
    group_layers = _split_layers_into_three_groups(spec.num_layers)

    # 把稠密层（MoE 模型前 first_k_dense_replace 层）优先分配到 front 组。
    remaining_dense = components["dense_layers"]
    group_specs: list[tuple[str, int, int]] = []
    for group_name in ("front", "mid", "back"):
        layer_count = group_layers[group_name]
        dense_in_group = min(remaining_dense, layer_count)
        remaining_dense -= dense_in_group
        group_specs.append((group_name, dense_in_group, layer_count - dense_in_group))

    transformer_params_per_rank = 0
    activation_bytes = 0
    for group_name, dense_count, moe_count in group_specs:
        attn_policy = proposal.grouped_attention_policies()[group_name]
        mlp_policy = proposal.grouped_mlp_policies()[group_name]
        attn_factor = _attention_shard_factor(attn_policy, tp_size)
        mlp_factor = _mlp_shard_factor(mlp_policy, tp_size)

        layer_count = dense_count + moe_count
        transformer_params_per_rank += int(
            layer_count * math.ceil(components["attention"] / attn_factor)
        )
        transformer_params_per_rank += int(
            dense_count * math.ceil(components["dense_mlp"] / mlp_factor)
        )
        # 路由专家按 EP×TP 双重切分（Megatron 默认 ETP=TP）；共享专家只按 TP 切分。
        transformer_params_per_rank += int(
            moe_count
            * math.ceil(int(components["routed_experts"] / ep_size) / mlp_factor)
        )
        transformer_params_per_rank += int(
            moe_count * math.ceil(components["shared_expert"] / mlp_factor)
        )
        transformer_params_per_rank += layer_count * components["norm"]

        activation_factor = max(attn_factor, mlp_factor)
        activation_bytes += int(
            layer_count
            * micro_batch_size
            * seq_length
            * spec.hidden_size
            * _BYTES_PER_BF16
            * _ACTIVATION_MULTIPLIER
            * _RECOMPUTE_RATIO
            / activation_factor
        )

    # PP 切分：每个 rank 只持有本 stage 的层（≈ num_layers/pp，向上取整偏保守）。
    layers_per_stage = math.ceil(spec.num_layers / pp_size)
    pp_fraction = layers_per_stage / max(spec.num_layers, 1)
    transformer_params_per_rank = int(transformer_params_per_rank * pp_fraction)
    activation_bytes = int(activation_bytes * pp_fraction)

    embedding_params = spec.vocab_size * spec.hidden_size
    lm_head_params = 0 if spec.tie_word_embeddings else spec.vocab_size * spec.hidden_size
    final_norm_params = spec.hidden_size

    embedding_factor = tp_size if tp_size > 1 and proposal.embedding_policy == "vocab_shard" else 1
    lm_head_factor = tp_size if tp_size > 1 and proposal.lm_head_policy == "vocab_shard" else 1

    params_per_rank = (
        transformer_params_per_rank
        + math.ceil(embedding_params / embedding_factor)
        + math.ceil(lm_head_params / lm_head_factor)
        + final_norm_params
    )
    weights_bytes = params_per_rank * _BYTES_PER_BF16
    grads_bytes = params_per_rank * _BYTES_PER_BF16
    optimizer_bytes = params_per_rank * _ADAM_STATE_BYTES

    if tp_size > 1 and proposal.has_uniform_attention_policy() and proposal.has_uniform_mlp_policy():
        if proposal.attn_policy_mid == "head_shard" and proposal.mlp_policy_mid == "colrow_shard":
            activation_bytes = int(activation_bytes * 0.82)

    total_bytes = weights_bytes + grads_bytes + optimizer_bytes + activation_bytes
    return MemoryEstimate(
        weights_bytes=weights_bytes,
        grads_bytes=grads_bytes,
        optimizer_bytes=optimizer_bytes,
        activation_bytes=activation_bytes,
        total_bytes=total_bytes,
    )


def _build_parallel_plan(
    spec: ModelSpec,
    env: RuntimeEnvironment,
    proposal: GroupedPolicyProposal,
    micro_batch_size: int,
    estimated_memory: MemoryEstimate,
    memory_margin_bytes: int,
    megatron_lowerable: bool,
) -> ParallelPlan:
    """从提案构造用户可见的 ParallelPlan。"""
    global_batch_size = micro_batch_size * proposal.mesh.dp_size
    execution_mode = (
        "current_feasible"
        if proposal.mesh.world_size <= env.device_count
        else "future_scalable"
    )
    sequence_parallel = (
        proposal.mesh.tp_size > 1
        and megatron_lowerable
        and proposal.attn_policy_mid == "head_shard"
        and proposal.mlp_policy_mid == "colrow_shard"
    )
    return ParallelPlan(
        model_path=spec.model_dir,
        tp=proposal.mesh.tp_size,
        dp=proposal.mesh.dp_size,
        pp=proposal.mesh.pp_size,
        ep=proposal.expert_parallel_size,
        attention_policy=proposal.attention_policy_summary(),
        mlp_policy=proposal.mlp_policy_summary(),
        attention_policy_front=proposal.attn_policy_front,
        attention_policy_mid=proposal.attn_policy_mid,
        attention_policy_back=proposal.attn_policy_back,
        mlp_policy_front=proposal.mlp_policy_front,
        mlp_policy_mid=proposal.mlp_policy_mid,
        mlp_policy_back=proposal.mlp_policy_back,
        embedding_policy=proposal.embedding_policy,
        lm_head_policy=proposal.lm_head_policy,
        sequence_parallel=sequence_parallel,
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
        parallel_mode="MODEL_PARALLEL" if proposal.mesh.tp_size > 1 else "DATA_PARALLEL",
        execution_mode=execution_mode,
        strategy_name="PENDING",
        strategy_policy="PENDING",
        search_time_seconds=0.0,
        estimated_memory_bytes=estimated_memory.total_bytes,
        memory_margin_bytes=memory_margin_bytes,
        fits_current_environment=proposal.mesh.world_size <= env.device_count,
        requires_additional_devices=max(0, proposal.mesh.world_size - env.device_count),
        megatron_lowerable=megatron_lowerable,
        notes=(
            f"mesh={proposal.mesh.label()}",
            f"proposal={proposal.label()}",
            f"pp={proposal.mesh.pp_size},ep={proposal.expert_parallel_size}"
            if proposal.mesh.pp_size > 1 or proposal.expert_parallel_size > 1
            else "pp=1,ep=1",
        ),
    )


def _candidate_score(candidate: SearchCandidate, policy: str) -> tuple[int, ...]:
    """按照用户指定偏好构造排序分数。"""
    normalized_policy = _normalize_policy(policy)
    current_rank = 0 if candidate.fits_current_environment else 1
    device_gap = candidate.requires_additional_devices
    world_size = candidate.world_size
    balance_gap = abs(candidate.tp_size - candidate.dp_size)
    negative_dp = -candidate.dp_size
    negative_tp = -candidate.tp_size
    negative_margin = -candidate.memory_margin_bytes
    negative_gbs = -candidate.global_batch_size
    embedding_rank = 0 if candidate.parallel_plan.embedding_policy == "vocab_shard" else 1

    if normalized_policy == "MEMORY_FIRST":
        return (
            current_rank,
            device_gap,
            negative_margin,
            embedding_rank,
            world_size,
            negative_tp,
        )
    if normalized_policy == "THROUGHPUT_FIRST":
        return (current_rank, device_gap, negative_gbs, world_size, negative_dp)
    if normalized_policy == "TP_FIRST":
        return (current_rank, device_gap, negative_tp, world_size, negative_dp)
    if normalized_policy == "DP_FIRST":
        return (current_rank, device_gap, negative_dp, world_size, negative_tp)
    return (
        current_rank,
        device_gap,
        world_size,
        balance_gap,
        embedding_rank,
        negative_tp,
        negative_dp,
    )


def _candidate_reward(candidate: SearchCandidate, policy: str) -> float:
    """把候选质量映射成 RL 可用的标量奖励。"""
    normalized_policy = _normalize_policy(policy)
    reward = 0.0

    if candidate.parallel_plan.megatron_lowerable:
        reward += 35.0

    if candidate.fits_current_environment:
        reward += 120.0
    else:
        reward += max(20.0 - 10.0 * candidate.requires_additional_devices, -60.0)

    reward += candidate.memory_margin_bytes / float(8 * 1024**3)
    reward -= 0.5 * candidate.world_size
    reward += 0.25 * candidate.global_batch_size

    if candidate.parallel_plan.sequence_parallel:
        reward += 2.0
    if candidate.parallel_plan.embedding_policy == "vocab_shard":
        reward += 1.0
    if candidate.parallel_plan.lm_head_policy == "vocab_shard":
        reward += 1.5
    if candidate.tp_size > 1 and candidate.dp_size > 1:
        reward += 3.0

    if normalized_policy == "MEMORY_FIRST":
        reward += candidate.memory_margin_bytes / float(4 * 1024**3)
    elif normalized_policy == "THROUGHPUT_FIRST":
        reward += 1.5 * candidate.global_batch_size + 0.8 * candidate.dp_size
    elif normalized_policy == "TP_FIRST":
        reward += 2.5 * candidate.tp_size
    elif normalized_policy == "DP_FIRST":
        reward += 2.5 * candidate.dp_size
    else:
        reward += 0.8 * candidate.tp_size + 0.8 * candidate.dp_size
        reward -= 0.5 * abs(candidate.tp_size - candidate.dp_size)
    return reward


def _evaluate_proposal(
    spec: ModelSpec,
    env: RuntimeEnvironment,
    seq_length: int,
    proposal: GroupedPolicyProposal,
    policy: str,
    usable_memory_bytes: int,
) -> tuple[list[SearchCandidate], list[str]]:
    """把单个提案展开成若干 micro-batch 候选。"""
    megatron_lowerable, lowerability_reasons = _check_megatron_lowerability(
        spec=spec,
        proposal=proposal,
    )
    if not megatron_lowerable:
        return [], [
            "proposal=%s: %s" % (proposal.label(), "；".join(lowerability_reasons))
        ]

    candidates: list[SearchCandidate] = []
    rejected_reasons: list[str] = []
    for micro_batch_size in _DEFAULT_MICRO_BATCHES:
        estimated_memory = _estimate_memory(
            spec=spec,
            proposal=proposal,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
        )
        if estimated_memory.total_bytes > usable_memory_bytes:
            rejected_reasons.append(
                "proposal=%s mbs=%d: 显存硬约束不满足"
                % (proposal.label(), micro_batch_size)
            )
            continue

        memory_margin_bytes = usable_memory_bytes - estimated_memory.total_bytes
        parallel_plan = _build_parallel_plan(
            spec=spec,
            env=env,
            proposal=proposal,
            micro_batch_size=micro_batch_size,
            estimated_memory=estimated_memory,
            memory_margin_bytes=memory_margin_bytes,
            megatron_lowerable=megatron_lowerable,
        )
        candidate = SearchCandidate(
            world_size=proposal.mesh.world_size,
            tp_size=proposal.mesh.tp_size,
            dp_size=proposal.mesh.dp_size,
            micro_batch_size=micro_batch_size,
            global_batch_size=micro_batch_size * proposal.mesh.dp_size,
            seq_length=seq_length,
            estimated_memory=estimated_memory,
            memory_margin_bytes=memory_margin_bytes,
            fits_current_environment=proposal.mesh.world_size <= env.device_count,
            requires_additional_devices=max(0, proposal.mesh.world_size - env.device_count),
            parallel_plan=parallel_plan,
            score=(),
        )
        candidates.append(replace(candidate, score=_candidate_score(candidate, policy)))
    return candidates, rejected_reasons


def _select_ep_candidate(spec: ModelSpec, dp_size: int, max_ep: int) -> int:
    """为 MoE 模型选择一个 EP 候选值。

    规则：在 1..min(max_ep, dp) 内，选择同时整除 dp 和 num_experts 的最大值。
    返回 1 表示不启用 EP（或没有合法 EP 切分）。
    """
    if not spec.is_moe or max_ep <= 1 or dp_size <= 1:
        return 1
    upper = min(int(max_ep), dp_size)
    for ep_candidate in range(upper, 1, -1):
        if dp_size % ep_candidate == 0 and spec.num_experts % ep_candidate == 0:
            return ep_candidate
    return 1


def _build_standard_tp_dp_proposal(
    spec: ModelSpec,
    mesh: MeshChoice,
    embedding_policy: str | None = None,
    expert_parallel_size: int = 1,
) -> GroupedPolicyProposal:
    """构造标准 DP+TP(+PP+EP) 公共策略域中的 proposal。"""
    if mesh.tp_size == 1:
        return GroupedPolicyProposal(
            mesh=mesh,
            attn_policy_front="replicate",
            mlp_policy_front="replicate",
            attn_policy_mid="replicate",
            mlp_policy_mid="replicate",
            attn_policy_back="replicate",
            mlp_policy_back="replicate",
            embedding_policy="replicate",
            lm_head_policy="replicate",
            expert_parallel_size=expert_parallel_size,
        )

    resolved_embedding_policy = embedding_policy or "vocab_shard"
    if spec.tie_word_embeddings:
        resolved_embedding_policy = _STANDARD_TP_LM_HEAD_POLICY
    return GroupedPolicyProposal(
        mesh=mesh,
        attn_policy_front=_STANDARD_TP_ATTENTION_POLICY,
        mlp_policy_front=_STANDARD_TP_MLP_POLICY,
        attn_policy_mid=_STANDARD_TP_ATTENTION_POLICY,
        mlp_policy_mid=_STANDARD_TP_MLP_POLICY,
        attn_policy_back=_STANDARD_TP_ATTENTION_POLICY,
        mlp_policy_back=_STANDARD_TP_MLP_POLICY,
        embedding_policy=resolved_embedding_policy,
        lm_head_policy=_STANDARD_TP_LM_HEAD_POLICY,
        expert_parallel_size=expert_parallel_size,
    )


def _build_standard_proposals(
    spec: ModelSpec,
    meshes: list[MeshChoice],
    max_ep: int = 1,
) -> list[GroupedPolicyProposal]:
    """构造标准 DP+TP(+PP+EP) 公共策略域的完整 proposal 集。"""
    proposals: list[GroupedPolicyProposal] = []
    for mesh in meshes:
        if mesh.tp_size == 1:
            proposals.append(_build_standard_tp_dp_proposal(spec=spec, mesh=mesh))
            ep_candidate = _select_ep_candidate(spec, mesh.dp_size, max_ep)
            if ep_candidate > 1:
                proposals.append(
                    _build_standard_tp_dp_proposal(
                        spec=spec,
                        mesh=mesh,
                        expert_parallel_size=ep_candidate,
                    )
                )
            continue

        if spec.tie_word_embeddings:
            proposals.append(_build_standard_tp_dp_proposal(spec=spec, mesh=mesh))
            ep_candidate = _select_ep_candidate(spec, mesh.dp_size, max_ep)
            if ep_candidate > 1:
                proposals.append(
                    _build_standard_tp_dp_proposal(
                        spec=spec,
                        mesh=mesh,
                        expert_parallel_size=ep_candidate,
                    )
                )
            continue

        for embedding_policy in _EMBEDDING_POLICIES:
            proposals.append(
                _build_standard_tp_dp_proposal(
                    spec=spec,
                    mesh=mesh,
                    embedding_policy=embedding_policy,
                )
            )
        ep_candidate = _select_ep_candidate(spec, mesh.dp_size, max_ep)
        if ep_candidate > 1:
            proposals.append(
                _build_standard_tp_dp_proposal(
                    spec=spec,
                    mesh=mesh,
                    embedding_policy="vocab_shard",
                    expert_parallel_size=ep_candidate,
                )
            )
    return proposals


def _enumerate_proposals(
    spec: ModelSpec,
    env: RuntimeEnvironment,
    seq_length: int,
    proposals: list[GroupedPolicyProposal],
    policy: str,
) -> StrategyOutput:
    """在给定 proposal 集合上枚举候选。"""
    usable_memory_bytes = int(env.primary_device_memory_bytes * _MEMORY_UTILIZATION)
    feasible_current: list[SearchCandidate] = []
    feasible_future: list[SearchCandidate] = []
    rejected_reasons: list[str] = []

    for proposal in proposals:
        proposal_candidates, proposal_rejections = _evaluate_proposal(
            spec=spec,
            env=env,
            seq_length=seq_length,
            proposal=proposal,
            policy=policy,
            usable_memory_bytes=usable_memory_bytes,
        )
        rejected_reasons.extend(proposal_rejections)
        for candidate in proposal_candidates:
            if candidate.fits_current_environment:
                feasible_current.append(candidate)
            else:
                feasible_future.append(candidate)
    return feasible_current, feasible_future, rejected_reasons


def _proposal_proxy_score(
    spec: ModelSpec,
    env: RuntimeEnvironment,
    proposal: GroupedPolicyProposal,
    policy: str,
) -> float:
    """构造 Hierarchical 第一阶段的粗评分。"""
    megatron_lowerable, lowerability_reasons = _check_megatron_lowerability(spec, proposal)
    reward = 0.0
    if megatron_lowerable:
        reward += 30.0
    else:
        reward -= 25.0 + 2.5 * len(lowerability_reasons)

    if proposal.mesh.world_size <= env.device_count:
        reward += 20.0
    else:
        reward -= 5.0 * max(0, proposal.mesh.world_size - env.device_count)

    reward -= _proposal_structural_penalty(proposal)
    if proposal.embedding_policy == "vocab_shard" and proposal.mesh.tp_size > 1:
        reward += 1.5
    if proposal.lm_head_policy == "vocab_shard" and proposal.mesh.tp_size > 1:
        reward += 2.0
    if proposal.mesh.tp_size > 1 and proposal.mesh.dp_size > 1:
        reward += 3.0

    normalized_policy = _normalize_policy(policy)
    if normalized_policy == "MEMORY_FIRST":
        reward += 1.2 * proposal.mesh.tp_size
    elif normalized_policy == "THROUGHPUT_FIRST":
        reward += 1.5 * proposal.mesh.dp_size
    elif normalized_policy == "TP_FIRST":
        reward += 2.5 * proposal.mesh.tp_size
    elif normalized_policy == "DP_FIRST":
        reward += 2.5 * proposal.mesh.dp_size
    else:
        reward += 0.8 * proposal.mesh.tp_size + 0.8 * proposal.mesh.dp_size
        reward -= 0.5 * abs(proposal.mesh.tp_size - proposal.mesh.dp_size)
    return reward


def _build_decision_nodes(meshes: list[MeshChoice]) -> list[DecisionNode]:
    """构造 PPO 的标准 DP+TP 决策节点。"""
    decision_nodes = [DecisionNode("mesh", tuple(mesh.label() for mesh in meshes))]
    if any(mesh.tp_size > 1 for mesh in meshes):
        decision_nodes.append(DecisionNode("embedding_policy", _EMBEDDING_POLICIES))
    return decision_nodes


def _build_proposal_from_assignment(
    spec: ModelSpec,
    assignment: dict[str, str],
    mesh_lookup: dict[str, MeshChoice],
) -> GroupedPolicyProposal:
    """根据 PPO assignment 构造标准 DP+TP 提案。"""
    return _build_standard_tp_dp_proposal(
        spec=spec,
        mesh=mesh_lookup[assignment["mesh"]],
        embedding_policy=assignment.get("embedding_policy"),
    )


def _proposal_to_assignment(proposal: GroupedPolicyProposal) -> dict[str, str]:
    """把标准 proposal 转成 PPO 可消费的 assignment。"""
    assignment = {
        "mesh": proposal.mesh.label(),
    }
    if proposal.mesh.tp_size > 1:
        assignment["embedding_policy"] = proposal.embedding_policy
    return assignment


def _evaluate_assignment_for_ppo(
    spec: ModelSpec,
    env: RuntimeEnvironment,
    seq_length: int,
    policy: str,
    mesh_lookup: dict[str, MeshChoice],
    assignment: dict[str, str],
) -> ControllerEvaluation:
    """评估 PPO 控制器给出的一次完整 assignment。"""
    usable_memory_bytes = int(env.primary_device_memory_bytes * _MEMORY_UTILIZATION)
    proposal = _build_proposal_from_assignment(spec, assignment, mesh_lookup)
    current_candidates, rejected_reasons = _evaluate_proposal(
        spec=spec,
        env=env,
        seq_length=seq_length,
        proposal=proposal,
        policy=policy,
        usable_memory_bytes=usable_memory_bytes,
    )
    feasible_current = [candidate for candidate in current_candidates if candidate.fits_current_environment]
    feasible_future = [candidate for candidate in current_candidates if not candidate.fits_current_environment]
    all_candidates = feasible_current + feasible_future

    reward = -_proposal_structural_penalty(proposal)
    megatron_lowerable, lowerability_reasons = _check_megatron_lowerability(spec, proposal)
    if megatron_lowerable:
        reward += 35.0
    else:
        reward -= 35.0 + 2.5 * len(lowerability_reasons)

    if not all_candidates:
        reward -= 25.0
    else:
        best_candidate = max(all_candidates, key=lambda item: _candidate_reward(item, policy))
        reward += _candidate_reward(best_candidate, policy)

    rejected_payload = list(rejected_reasons)
    rejected_payload.extend(lowerability_reasons)
    rejected_payload.append("PPO proposal=%s" % proposal.label())

    return ControllerEvaluation(
        reward=reward,
        assignment=dict(assignment),
        feasible_current_candidates=feasible_current,
        scalable_future_candidates=feasible_future,
        rejected_reasons=rejected_payload,
    )


def _block_level_fast_strategy(
    spec: ModelSpec,
    env: RuntimeEnvironment,
    seq_length: int,
    max_world_size: int,
    policy: str,
    max_pp: int = 1,
    max_ep: int = 1,
) -> StrategyOutput:
    """直接在 block 级别搜索一小组真可执行方案。"""
    meshes = _build_mesh_choices(
        spec=spec,
        env=env,
        max_world_size=max_world_size,
        max_pp=max_pp,
    )
    if not meshes:
        return [], [], ["没有任何合法 mesh，无法继续 BLOCK_LEVEL_FAST 搜索。"]
    proposals = _build_standard_proposals(spec=spec, meshes=meshes, max_ep=max_ep)
    return _enumerate_proposals(
        spec=spec,
        env=env,
        seq_length=seq_length,
        proposals=proposals,
        policy=policy,
    )


def _hierarchical_strategy(
    spec: ModelSpec,
    env: RuntimeEnvironment,
    seq_length: int,
    max_world_size: int,
    policy: str,
    max_pp: int = 1,
    max_ep: int = 1,
) -> StrategyOutput:
    """两阶段分组式搜索。"""
    meshes = _build_mesh_choices(
        spec=spec,
        env=env,
        max_world_size=max_world_size,
        max_pp=max_pp,
    )
    if not meshes:
        return [], [], ["没有任何合法 mesh，无法继续 HIERARCHICAL 搜索。"]

    proposals = _build_standard_proposals(spec=spec, meshes=meshes, max_ep=max_ep)
    scored_proposals = sorted(
        proposals,
        key=lambda proposal: _proposal_proxy_score(spec, env, proposal, policy),
        reverse=True,
    )
    keep_count = min(len(scored_proposals), max(24, len(meshes) * 12))
    selected_proposals = scored_proposals[:keep_count]

    rejected_reasons = [
        "HIERARCHICAL 第一阶段保留 %d / %d 个 proposal。"
        % (len(selected_proposals), len(scored_proposals))
    ]
    feasible_current, feasible_future, stage_two_rejections = _enumerate_proposals(
        spec=spec,
        env=env,
        seq_length=seq_length,
        proposals=selected_proposals,
        policy=policy,
    )
    rejected_reasons.extend(stage_two_rejections)
    return feasible_current, feasible_future, rejected_reasons


def _rl_guided_ppo_strategy(
    spec: ModelSpec,
    env: RuntimeEnvironment,
    seq_length: int,
    max_world_size: int,
    policy: str,
    max_pp: int = 1,
    max_ep: int = 1,
) -> StrategyOutput:
    """第三策略：基于 clipped PPO 的在线搜索。"""
    meshes = _build_mesh_choices(
        spec=spec,
        env=env,
        max_world_size=max_world_size,
        max_pp=max_pp,
    )
    if not meshes:
        return [], [], ["没有任何合法 mesh，无法继续 RL_GUIDED_PPO 搜索。"]

    mesh_lookup = {mesh.label(): mesh for mesh in meshes}
    decision_nodes = _build_decision_nodes(meshes)
    standard_proposals = _build_standard_proposals(spec=spec, meshes=meshes, max_ep=max_ep)
    seed_assignments = [
        _proposal_to_assignment(proposal)
        for proposal in standard_proposals[: min(8, len(standard_proposals))]
    ]
    return run_rl_guided_ppo_search(
        spec=spec,
        env=env,
        decision_nodes=decision_nodes,
        evaluate_assignment_fn=lambda assignment: _evaluate_assignment_for_ppo(
            spec=spec,
            env=env,
            seq_length=seq_length,
            policy=policy,
            mesh_lookup=mesh_lookup,
            assignment=assignment,
        ),
        seed_assignments=seed_assignments,
    )


def _attach_search_metadata(
    candidate: SearchCandidate,
    strategy_name: str,
    policy: str,
    elapsed_seconds: float,
) -> SearchCandidate:
    """把搜索元数据回填到显式 ParallelPlan。"""
    return replace(
        candidate,
        parallel_plan=candidate.parallel_plan.with_updates(
            strategy_name=strategy_name,
            strategy_policy=policy,
            search_time_seconds=elapsed_seconds,
        ),
    )


def _select_recommended_candidate(
    feasible_current: list[SearchCandidate],
    feasible_future: list[SearchCandidate],
) -> tuple[SearchCandidate | None, str | None]:
    """从候选列表中挑选推荐方案。"""
    if feasible_current:
        return feasible_current[0], "current_feasible"
    if feasible_future:
        return feasible_future[0], "future_scalable"
    return None, None


def _finalize_strategy_result(
    strategy_name: str,
    policy: str,
    elapsed_seconds: float,
    feasible_current: list[SearchCandidate],
    feasible_future: list[SearchCandidate],
    rejected_reasons: list[str],
) -> _StrategyExecution:
    """对策略执行结果做排序、截断和统计。"""
    full_current_count = len(feasible_current)
    full_future_count = len(feasible_future)
    feasible_current.sort(key=lambda item: item.score)
    feasible_future.sort(key=lambda item: item.score)

    feasible_current = [
        _attach_search_metadata(candidate, strategy_name, policy, elapsed_seconds)
        for candidate in feasible_current
    ]
    feasible_future = [
        _attach_search_metadata(candidate, strategy_name, policy, elapsed_seconds)
        for candidate in feasible_future
    ]
    selected_candidate, execution_mode = _select_recommended_candidate(
        feasible_current=feasible_current,
        feasible_future=feasible_future,
    )

    benchmark = SearchBenchmark(
        strategy_name=strategy_name,
        strategy_policy=policy,
        elapsed_seconds=elapsed_seconds,
        feasible_current_count=full_current_count,
        scalable_future_count=full_future_count,
    )
    dedup_rejected = list(dict.fromkeys(rejected_reasons))
    return _StrategyExecution(
        strategy_name=strategy_name,
        strategy_policy=policy,
        elapsed_seconds=elapsed_seconds,
        execution_mode=execution_mode,
        selected_candidate=selected_candidate,
        selected_plan=(
            selected_candidate.parallel_plan if selected_candidate is not None else None
        ),
        feasible_current_candidates=feasible_current[:_CURRENT_LIMIT],
        scalable_future_candidates=feasible_future[:_FUTURE_LIMIT],
        rejected_reasons=dedup_rejected[:_REJECT_LIMIT],
        benchmark=benchmark,
    )


def _execute_search_strategy(
    strategy_name: str,
    spec: ModelSpec,
    env: RuntimeEnvironment,
    seq_length: int,
    max_world_size: int,
    policy: str,
    max_pp: int = 1,
    max_ep: int = 1,
) -> _StrategyExecution:
    """执行单个搜索策略并记录耗时。"""
    strategy_fn = _SEARCH_STRATEGY_REGISTRY.get(strategy_name)
    if strategy_fn is None:
        raise ValueError(
            f"未注册的 search_strategy: {strategy_name}，可选值为 {available_search_strategies()}"
        )

    started = time.perf_counter()
    feasible_current, feasible_future, rejected_reasons = strategy_fn(
        spec,
        env,
        seq_length,
        max_world_size,
        policy,
        max_pp=max_pp,
        max_ep=max_ep,
    )
    elapsed_seconds = time.perf_counter() - started
    return _finalize_strategy_result(
        strategy_name=strategy_name,
        policy=policy,
        elapsed_seconds=elapsed_seconds,
        feasible_current=feasible_current,
        feasible_future=feasible_future,
        rejected_reasons=rejected_reasons,
    )


def search_parallel_candidates(
    spec: ModelSpec,
    env: RuntimeEnvironment,
    seq_length: int | None = None,
    max_world_size: int = _SEARCH_MAX_WORLD_SIZE,
    strategy: str = "BLOCK_LEVEL_FAST",
    policy: str = "BALANCED",
    benchmark_all_builtin: bool = True,
    max_pp: int = 1,
    max_ep: int = 1,
) -> SearchResult:
    """执行 DP/TP/PP(+EP) 搜索，并返回显式 ParallelPlan 结果。

    max_pp 控制流水线并行上限（默认 1 保持旧的 DP+TP 行为）。
    max_ep 控制 MoE 模型的专家并行上限（只影响 is_moe 的 spec）。
    """
    search_seq_length = min(seq_length or spec.max_seq_len, spec.max_seq_len)
    normalized_strategy = _normalize_strategy(strategy)
    normalized_policy = _normalize_policy(policy)

    strategy_order = [normalized_strategy]
    if benchmark_all_builtin:
        benchmark_candidates = list(_HEURISTIC_BENCHMARK_STRATEGIES)
        if normalized_strategy == "RL_GUIDED_PPO":
            benchmark_candidates.insert(0, normalized_strategy)
        elif normalized_strategy not in benchmark_candidates and normalized_strategy in _SEARCH_STRATEGY_REGISTRY:
            benchmark_candidates.insert(0, normalized_strategy)

        for strategy_name in benchmark_candidates:
            if strategy_name not in strategy_order:
                strategy_order.append(strategy_name)

    executions: dict[str, _StrategyExecution] = {}
    benchmarks: list[SearchBenchmark] = []
    for strategy_name in strategy_order:
        execution = _execute_search_strategy(
            strategy_name=strategy_name,
            spec=spec,
            env=env,
            seq_length=search_seq_length,
            max_world_size=max_world_size,
            policy=normalized_policy,
            max_pp=max_pp,
            max_ep=max_ep,
        )
        executions[strategy_name] = execution
        benchmarks.append(execution.benchmark)

    selected = executions[normalized_strategy]
    return SearchResult(
        strategy_name=selected.strategy_name,
        strategy_policy=selected.strategy_policy,
        planning_device_count=env.device_count,
        search_time_seconds=selected.elapsed_seconds,
        execution_mode=selected.execution_mode,
        selected_plan=selected.selected_plan,
        selected_candidate=selected.selected_candidate,
        feasible_current_candidates=selected.feasible_current_candidates,
        scalable_future_candidates=selected.scalable_future_candidates,
        feasible_current_plans=[
            candidate.parallel_plan for candidate in selected.feasible_current_candidates
        ],
        scalable_future_plans=[
            candidate.parallel_plan for candidate in selected.scalable_future_candidates
        ],
        benchmarks=benchmarks,
        rejected_reasons=selected.rejected_reasons,
    )


register_search_strategy("BLOCK_LEVEL_FAST", _block_level_fast_strategy)
register_search_strategy("HIERARCHICAL", _hierarchical_strategy)
register_search_strategy("RL_GUIDED_PPO", _rl_guided_ppo_strategy)
