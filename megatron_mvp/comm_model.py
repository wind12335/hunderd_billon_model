"""解析式通信代价模型（G2/G3）。

对给定 (spec, tp/dp/pp/ep, mbs, seq, env) 估算每卡每步的通信字节数与
时间，按通信组"是否跨节点"选择机内/机间带宽折算（G3 拓扑感知）。产物
只用于候选排序、RL 奖励塑形和存档节奏推导，不承诺绝对时间精度。

带宽/时延默认值已按 second_hygon Phase6 实测回填（见 types.py 出处注释）；
仍可用 AUTOPLAN_TOPO_* 覆盖，tools/bench_comm_hygon.py 微基准实测后可再回填。

模型假设（Megatron 标准 rank 布局：TP 最内层，EP 在 DP 内部，PP 最外层）：

1. TP 集合通信：每层 2 次 all-reduce（fwd+bwd），payload = mbs*seq*hidden*2B；
   ring 每卡通信量系数 2*(tp-1)/tp；序列并行时 all-reduce 拆成
   all-gather + reduce-scatter，总量不变。TP 组跨节点当且仅当 tp > 每节点卡数。
2. DP 梯度同步：分布式优化器 = 梯度 reduce-scatter + 参数 all-gather，
   与朴素 all-reduce 总量同为 2*(dp-1)/dp * grads；DP 组按 stage 内
   tp*dp 是否超过每节点卡数判定跨节点。
3. PP 点对点：每个 microbatch 跨 (pp-1) 个 stage 边界，fwd+bwd 各一次，
   payload = mbs*seq*hidden*2B；时延主导，按 max(时延, 字节/带宽) 折算。
4. EP all-to-all（a2a）：每个 MoE 层派发+回收各一次；单方向每卡字节 ≈
   mbs*seq*topk*(ep-1)/ep*hidden*2B（每 token 的 top-k 个专家期望落在
   其它卡的占比 (ep-1)/ep，同行专家去重忽略，略偏保守）。EP 组跨节点
   当且仅当 tp*ep > 每节点卡数（EP 组成员按 TP 间隔分布）。
"""

from __future__ import annotations

import math

from megatron_mvp.types import CommEstimate
from megatron_mvp.types import ModelSpec
from megatron_mvp.types import RuntimeEnvironment

_BYTES_PER_BF16 = 2


def _group_cross_node(group_world: int, gpus_per_node: int) -> bool:
    """按"组内卡数超过单节点卡数"判定通信组是否跨节点。"""
    return group_world > max(int(gpus_per_node), 1)


def _link(env: RuntimeEnvironment, cross_node: bool) -> tuple[float, float]:
    """返回 (带宽 bytes/s, 单消息时延 s)。"""
    if cross_node:
        return env.inter_node_bandwidth_bytes_per_sec, env.inter_node_latency_seconds
    return env.intra_node_bandwidth_bytes_per_sec, env.intra_node_latency_seconds


def _collective_seconds(
    payload_bytes: float,
    group_size: int,
    env: RuntimeEnvironment,
    cross_node: bool,
) -> tuple[int, float]:
    """按 ring 通信量系数折算集合通信的每卡字节与时间。

    ring 系数 2*(group-1)/group：tp=1 时退化为 0（本地无集合通信）。
    时间 = 通信量/带宽 + 时延*组数（时延项按每跳一消息的保守计数）。
    """
    if group_size <= 1 or payload_bytes <= 0:
        return 0, 0.0
    ring_factor = 2.0 * (group_size - 1) / group_size
    per_rank_bytes = int(payload_bytes * ring_factor)
    bandwidth, latency = _link(env, cross_node)
    seconds = per_rank_bytes / bandwidth + latency * group_size
    return per_rank_bytes, seconds


def estimate_communication(
    spec: ModelSpec,
    env: RuntimeEnvironment,
    tp_size: int,
    dp_size: int,
    pp_size: int,
    ep_size: int,
    micro_batch_size: int,
    seq_length: int,
    grads_bytes_per_rank: int,
) -> CommEstimate:
    """估算一个 (tp, dp, pp, ep, mbs) 组合每卡每步的通信代价。"""
    tp_size = max(int(tp_size), 1)
    dp_size = max(int(dp_size), 1)
    pp_size = max(int(pp_size), 1)
    ep_size = max(int(ep_size), 1)
    gpus_per_node = max(int(env.gpus_per_node), 1)
    hidden = spec.hidden_size
    act_payload = micro_batch_size * seq_length * hidden * _BYTES_PER_BF16
    layers_per_stage = math.ceil(spec.num_layers / pp_size)

    # --- TP：每层 2 次 all-reduce（fwd+bwd），PP 后按本 stage 层数折算 ---
    tp_cross = _group_cross_node(tp_size, gpus_per_node)
    tp_bytes, tp_seconds = _collective_seconds(
        act_payload * 2.0 * layers_per_stage, tp_size, env, tp_cross
    )

    # --- DP：梯度 reduce-scatter + 参数 all-gather，≈ 朴素 all-reduce 总量 ---
    dp_cross = _group_cross_node(tp_size * dp_size, gpus_per_node)
    dp_bytes, dp_seconds = _collective_seconds(
        float(grads_bytes_per_rank), dp_size, env, dp_cross
    )

    # --- PP：点对点激活搬运，时延主导 ---
    pp_cross = _group_cross_node(tp_size * dp_size * pp_size, gpus_per_node)
    pp_bytes = 0
    pp_seconds = 0.0
    if pp_size > 1:
        pp_bytes = int(act_payload * 2 * (pp_size - 1))
        bandwidth, latency = _link(env, pp_cross)
        pp_seconds = (pp_size - 1) * 2 * max(act_payload / bandwidth, latency)

    # --- EP：每个 MoE 层派发+回收（a2a 各一次，跨卡占比 (ep-1)/ep）---
    ep_bytes = 0
    ep_seconds = 0.0
    ep_cross = False
    moe_topk = max(int(spec.num_experts_per_tok or 1), 1)
    moe_layers = max(spec.num_layers - spec.first_k_dense_replace, 0)
    if ep_size > 1 and spec.is_moe and moe_layers > 0:
        ep_cross = _group_cross_node(tp_size * ep_size, gpus_per_node)
        # 序列并行下 token 已按 TP 切分；未开序列并行时 a2a 在完整 token 上发生，
        # 统一按保守的完整 mbs*seq 估。
        a2a_per_layer_bytes = act_payload * moe_topk * (ep_size - 1) / ep_size
        moe_layers_per_stage = math.ceil(moe_layers / pp_size)
        ep_bytes = int(a2a_per_layer_bytes * 2 * moe_layers_per_stage)
        bandwidth, latency = _link(env, ep_cross)
        # a2a 完成时间按最大单卡出流量折算（等效全互连树高近似为组数跳）。
        ep_seconds = ep_bytes / bandwidth + latency * ep_size * 2 * moe_layers_per_stage

    return CommEstimate(
        tp_bytes=tp_bytes,
        dp_bytes=dp_bytes,
        pp_bytes=pp_bytes,
        ep_bytes=ep_bytes,
        tp_seconds=tp_seconds,
        dp_seconds=dp_seconds,
        pp_seconds=pp_seconds,
        ep_seconds=ep_seconds,
        cross_node={
            "tp": tp_cross,
            "dp": dp_cross,
            "pp": pp_cross,
            "ep": ep_cross,
        },
    )


def estimate_active_params(spec: ModelSpec) -> int:
    """估算单 token 前向激活的参数量（用于计算时间与 MFU 折算）。

    attention 按全量（GLM-4.5 无 MLA 时含完整 QKV/o 投影），MoE 层按
    top-k 路由专家 + 共享专家；稠密层按完整 MLP。
    """
    head_dim = spec.head_dim
    if head_dim is None or head_dim <= 0:
        head_dim = (
            spec.hidden_size // spec.num_heads
            if spec.hidden_size % spec.num_heads == 0
            else spec.hidden_size
        )
    q_hidden = spec.num_heads * head_dim
    kv_hidden = spec.num_kv_heads * head_dim
    attention = (
        spec.hidden_size * q_hidden
        + spec.hidden_size * kv_hidden * 2
        + q_hidden * spec.hidden_size
    )
    moe_layers = max(spec.num_layers - spec.first_k_dense_replace, 0)
    dense_layers = spec.num_layers - moe_layers
    active = 0
    active += spec.num_layers * attention
    active += dense_layers * spec.hidden_size * spec.intermediate_size * 3
    if spec.is_moe:
        routed_inter = spec.moe_intermediate_size or spec.intermediate_size
        active += moe_layers * spec.num_experts_per_tok * spec.hidden_size * routed_inter * 3
        if spec.shared_expert_intermediate_size:
            active += moe_layers * spec.hidden_size * spec.shared_expert_intermediate_size * 3
    return int(active)


def estimate_step_seconds(
    spec: ModelSpec,
    env: RuntimeEnvironment,
    communication: CommEstimate,
    tp_size: int,
    dp_size: int,
    pp_size: int,
    micro_batch_size: int,
    seq_length: int,
    flops_per_device: float = 300.0e12,  # 2026-10-26 双机实测 matmul 302.5 TFLOPS(comm_bench_w2.csv)
) -> float:
    """估算每步墙钟时间 = 计算时间 + 通信时间（不建重叠收益模型，保守相加）。

    计算时间 = 6 * 激活参数 * 全局 token / (world * flops_per_device)。
    flops_per_device 默认 300 TFLOPS（2026-10-26 双机实测 302.5，原占位 250），可用
    AUTOPLAN_TOPO_TFLOPS 在调用方换算覆盖（此处只做纯函数）。
    """
    world = max(tp_size, 1) * max(dp_size, 1) * max(pp_size, 1)
    global_tokens = micro_batch_size * dp_size * seq_length
    active = estimate_active_params(spec)
    compute_seconds = 6.0 * active * global_tokens / (world * flops_per_device)
    return compute_seconds + communication.total_seconds
