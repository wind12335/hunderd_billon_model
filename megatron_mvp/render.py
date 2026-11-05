"""粗 IR、Megatron 配置和启动脚本渲染。"""

from __future__ import annotations

import json
import os
from pathlib import Path

from megatron_mvp.comm_model import estimate_step_seconds
from megatron_mvp.vendor_profile import current_profile
from megatron_mvp.types import CoarseIR
from megatron_mvp.types import GenerationOutputs
from megatron_mvp.types import ModelArtifacts
from megatron_mvp.types import ModelSpec
from megatron_mvp.types import ParallelPlan
from megatron_mvp.types import RuntimeEnvironment
from megatron_mvp.types import SearchCandidate
from megatron_mvp.types import SearchResult


def select_recommended_candidate(search_result: SearchResult) -> tuple[SearchCandidate, str]:
    """选择推荐候选。"""
    if search_result.selected_candidate is not None:
        return search_result.selected_candidate, search_result.execution_mode or "current_feasible"
    if search_result.feasible_current_candidates:
        return search_result.feasible_current_candidates[0], "current_feasible"
    if search_result.scalable_future_candidates:
        return search_result.scalable_future_candidates[0], "future_scalable"
    raise RuntimeError("粗搜索未找到任何可行候选，无法生成后续 IR 和模板。")


def select_recommended_plan(search_result: SearchResult) -> tuple[ParallelPlan, str]:
    """选择推荐的显式 ParallelPlan。"""
    if search_result.selected_plan is not None:
        return search_result.selected_plan, search_result.execution_mode or "current_feasible"
    candidate, execution_mode = select_recommended_candidate(search_result)
    return candidate.parallel_plan, execution_mode


def build_coarse_ir(
    spec: ModelSpec,
    env: RuntimeEnvironment,
    plan: ParallelPlan,
    search_result: SearchResult,
) -> CoarseIR:
    """构造 MVP 粗 IR。"""
    execution_mode = search_result.execution_mode or plan.execution_mode
    search_seq_length = spec.max_seq_len
    if search_result.selected_candidate is not None:
        search_seq_length = search_result.selected_candidate.seq_length

    search_notes = [
        "搜索域覆盖 DP + TP + PP（MoE 模型额外包含 EP 候选）。",
        "显存过滤使用 block-level 粗估，并假设开启激活重计算。",
        "选定搜索策略 %s，策略偏好 %s，耗时 %.4f 秒。"
        % (
            search_result.strategy_name,
            search_result.strategy_policy,
            search_result.search_time_seconds,
        ),
        "最终显式 ParallelPlan: TP=%d, DP=%d, PP=%d, EP=%d, attention=%s, mlp=%s, embedding=%s, lm_head=%s。"
        % (
            plan.tp,
            plan.dp,
            plan.pp,
            plan.ep,
            plan.attention_policy,
            plan.mlp_policy,
            plan.embedding_policy,
            plan.lm_head_policy,
        ),
        "三段式策略: attn(front/mid/back)=(%s,%s,%s), mlp(front/mid/back)=(%s,%s,%s)。"
        % (
            plan.resolved_attention_policy_front(),
            plan.resolved_attention_policy_mid(),
            plan.resolved_attention_policy_back(),
            plan.resolved_mlp_policy_front(),
            plan.resolved_mlp_policy_mid(),
            plan.resolved_mlp_policy_back(),
        ),
    ]
    if execution_mode == "future_scalable":
        search_notes.append(
            "当前单机卡数不足，推荐方案需要扩展到至少 %d 卡。" % plan.world_size
        )

    # 通信估算与存档节奏（G2/G3/G7）：跟随选定候选；无估算（手工 IR）时为 None。
    communication_payload = None
    selected_candidate = search_result.selected_candidate
    if selected_candidate is not None and selected_candidate.communication is not None:
        comm = selected_candidate.communication.to_dict()
        flops_per_device = float(os.environ.get("AUTOPLAN_TOPO_TFLOPS", "300")) * 1e12
        step_seconds = estimate_step_seconds(
            spec=spec,
            env=env,
            communication=selected_candidate.communication,
            tp_size=plan.tp,
            dp_size=plan.dp,
            pp_size=plan.pp,
            micro_batch_size=plan.micro_batch_size,
            seq_length=search_seq_length,
            flops_per_device=flops_per_device,
        )
        # 存档节奏（G7）：目标每 1 小时存一次（doc 预训练方案要求 1-2h），
        # 区间下限 50 / 上限 2000 步兜底；保留间隔取 10 倍存档间隔（引擎要求
        # 整除，checkpointing 按"非保留倍数的上一存档删除"滚动清理）。
        save_interval = int(min(2000, max(50, round(3600.0 / max(step_seconds, 1e-6)))))
        communication_payload = {
            **comm,
            "estimated_step_seconds": step_seconds,
            "flops_per_device": flops_per_device,
            "save_interval_recommendation": save_interval,
            "save_retain_interval_recommendation": save_interval * 10,
            "estimated_checkpoint_bytes": int(spec.estimated_total_params) * 14,
        }
        cross = [k for k, v in comm["cross_node"].items() if v]
        search_notes.append(
            "通信估算(G2/G3 占位模型): 每卡每步 %.1f MB / %.3f s，跨节点维度=%s。"
            % (comm["total_bytes"] / 1024**2, comm["total_seconds"], cross or "无")
        )
        search_notes.append(
            "每步墙钟估算 %.3f s → 存档间隔 %d 步（约 1h），保留间隔 %d 步；单份存档体积约 %.1f TB。"
            % (
                step_seconds,
                save_interval,
                save_interval * 10,
                int(spec.estimated_total_params) * 14 / 1024**4,
            )
        )

    benchmark_payload = [benchmark.to_dict() for benchmark in search_result.benchmarks]
    return CoarseIR(
        style="megatron_standard",
        execution_mode=execution_mode,
        model_dir=spec.model_dir,
        model_type=spec.model_type,
        architecture=spec.architecture,
        parallelism={
            "dp_size": plan.dp,
            "tp_size": plan.tp,
            "pp_size": plan.pp,
            "expert_model_parallel_size": plan.ep,
            "world_size": plan.world_size,
            "sequence_parallel": bool(
                plan.sequence_parallel
                and current_profile(env.accelerator_type).allow_sequence_parallel
            ),
        },
        parallel_plan=plan.to_dict(),
        train={
            "micro_batch_size": plan.micro_batch_size,
            "global_batch_size": plan.global_batch_size,
            "seq_length": search_seq_length,
            "precision": "bf16",
            "use_distributed_optimizer": plan.dp > 1 or plan.tp > 1,
            "use_activation_recompute": True,
        },
        environment={
            "detected_device_count": env.device_count,
            "device_name": env.primary_device_name,
            "device_memory_bytes": env.primary_device_memory_bytes,
            "accelerator_type": env.accelerator_type,
            "vendor_profile": current_profile(env.accelerator_type).key,
            "hip_version": env.hip_version,
        },
        search={
            "strategy_name": search_result.strategy_name,
            "strategy_policy": search_result.strategy_policy,
            "planning_device_count": search_result.planning_device_count,
            "search_time_seconds": search_result.search_time_seconds,
            "benchmarks": benchmark_payload,
        },
        model={
            "hidden_size": spec.hidden_size,
            "intermediate_size": spec.intermediate_size,
            "num_layers": spec.num_layers,
            "num_heads": spec.num_heads,
            "num_kv_heads": spec.num_kv_heads,
            "vocab_size": spec.vocab_size,
            "max_seq_len": spec.max_seq_len,
            "rope_theta": spec.rope_theta,
            "rms_norm_eps": spec.rms_norm_eps,
            "tie_word_embeddings": spec.tie_word_embeddings,
            "estimated_total_params": spec.estimated_total_params,
            # MoE 模型字段（稠密模型为 0/None，不影响旧行为）。
            "is_moe": spec.is_moe,
            "head_dim": spec.head_dim,
            "num_experts": spec.num_experts,
            "num_experts_per_tok": spec.num_experts_per_tok,
            "moe_intermediate_size": spec.moe_intermediate_size,
            "shared_expert_intermediate_size": spec.shared_expert_intermediate_size,
            "first_k_dense_replace": spec.first_k_dense_replace,
            "num_nextn_predict_layers": spec.num_nextn_predict_layers,
            "partial_rotary_factor": spec.partial_rotary_factor,
            "attention_bias": spec.attention_bias,
            "moe_router_score_function": spec.moe_router_score_function,
            "use_mla": spec.use_mla,
            "q_lora_rank": spec.q_lora_rank,
            "kv_lora_rank": spec.kv_lora_rank,
            "qk_nope_head_dim": spec.qk_nope_head_dim,
            "qk_rope_head_dim": spec.qk_rope_head_dim,
            "mla_v_head_dim": spec.mla_v_head_dim,
        },
        search_notes=search_notes,
        communication=communication_payload,
    )


def build_megatron_config(ir: CoarseIR) -> dict[str, object]:
    """把粗 IR 映射成 Megatron 配置模板。"""
    model = ir.model
    train = ir.train
    parallelism = ir.parallelism
    parallel_plan = ir.parallel_plan
    profile = current_profile(ir.environment.get("accelerator_type"))

    model_args: dict[str, object] = {
        "use_mcore_models": True,
        "num_layers": model["num_layers"],
        "hidden_size": model["hidden_size"],
        "ffn_hidden_size": model["intermediate_size"],
        "num_attention_heads": model["num_heads"],
        "seq_length": train["seq_length"],
        "max_position_embeddings": model["max_seq_len"],
        "position_embedding_type": "rope",
        "normalization": "RMSNorm",
        "swiglu": True,
        "untie_embeddings_and_output_weights": not model["tie_word_embeddings"],
        "disable_bias_linear": True,
        "rotary_base": int(model["rope_theta"]),
    }
    # 厂商能力位：无厂商版 TransformerEngine 的档位（摩尔/天数）固定 local
    # 实现（权重结构与转换脚本一致）；海光 dcu_megatron 参数体系无此旗标不渲染。
    if profile.fixed_transformer_impl:
        model_args["transformer_impl"] = profile.fixed_transformer_impl
    # GQA 模型（num_kv_heads < num_heads）需要显式 num_query_groups；
    # MLA 模型没有 GQA（引擎断言两者互斥），且头维由 MLA 维度标志给出。
    if not model.get("use_mla"):
        if int(model["num_kv_heads"]) < int(model["num_heads"]):
            model_args["num_query_groups"] = model["num_kv_heads"]
        # head_dim 与 hidden//heads 不一致时（如 GLM-4.5 上采样 q_proj）必须显式 kv_channels。
        head_dim = model.get("head_dim")
        if head_dim and int(model["hidden_size"]) % int(model["num_heads"]) != 0:
            model_args["kv_channels"] = int(head_dim)
    # MLA（KV 低秩压缩注意力，DeepSeek-V3/Ling 系）：维度命名映射见 detect.py。
    if model.get("use_mla"):
        model_args["multi_latent_attention"] = True
        if model.get("q_lora_rank"):
            model_args["q_lora_rank"] = model["q_lora_rank"]
        if model.get("kv_lora_rank"):
            model_args["kv_lora_rank"] = model["kv_lora_rank"]
        if model.get("qk_nope_head_dim"):
            model_args["qk_head_dim"] = model["qk_nope_head_dim"]
        if model.get("qk_rope_head_dim"):
            model_args["qk_pos_emb_head_dim"] = model["qk_rope_head_dim"]
        if model.get("mla_v_head_dim"):
            model_args["v_head_dim"] = model["mla_v_head_dim"]
    # 部分旋转位置编码（GLM-4.5 partial_rotary_factor=0.5 → --rotary-percent）。
    if model.get("partial_rotary_factor"):
        model_args["rotary_percent"] = model["partial_rotary_factor"]
    # QKV 投影带偏置而其余线性层不带（GLM-4.5 attention_bias=true →
    # --add-qkv-bias；配合上面的 disable_bias_linear 恰与 HF 结构逐位一致）。
    if model.get("attention_bias"):
        model_args["add_qkv_bias"] = True

    parallelism_args: dict[str, object] = {
        "tensor_model_parallel_size": parallelism["tp_size"],
        "pipeline_model_parallel_size": parallelism["pp_size"],
        "sequence_parallel": bool(
            parallelism["sequence_parallel"] and profile.allow_sequence_parallel
        ),
    }
    if int(parallelism.get("expert_model_parallel_size", 1)) > 1:
        parallelism_args["expert_model_parallel_size"] = parallelism[
            "expert_model_parallel_size"
        ]

    # MoE 模型参数（GLM-4.5-Air 等 glm4_moe 结构）。
    if model.get("is_moe") and int(model.get("num_experts", 0)) > 0:
        moe_layers = int(model["num_layers"]) - int(model.get("first_k_dense_replace", 0))
        model_args["num_experts"] = model["num_experts"]
        model_args["moe_router_topk"] = model.get("num_experts_per_tok") or 1
        if model.get("moe_intermediate_size"):
            model_args["moe_ffn_hidden_size"] = model["moe_intermediate_size"]
        if model.get("shared_expert_intermediate_size"):
            model_args["moe_shared_expert_intermediate_size"] = model[
                "shared_expert_intermediate_size"
            ]
        model_args["moe_layer_freq"] = "[0]*%d+[1]*%d" % (
            int(model.get("first_k_dense_replace", 0)),
            max(moe_layers, 0),
        )
        model_args["moe_aux_loss_coeff"] = 1.0e-4
        model_args["moe_router_load_balancing_type"] = "aux_loss"
        # DeepSeek-V3 式路由：sigmoid 打分 + 每专家偏置（探测依据见 detect.py）。
        if model.get("moe_router_score_function") == "sigmoid":
            model_args["moe_router_score_function"] = "sigmoid"
            model_args["moe_router_enable_expert_bias"] = True

    return {
        "entrypoint": "Megatron-LM/pretrain_gpt.py",
        "model_args": model_args,
        "parallelism": parallelism_args,
        "training_args": {
            "micro_batch_size": train["micro_batch_size"],
            "global_batch_size": train["global_batch_size"],
            "bf16": train["precision"] == "bf16",
            "use_distributed_optimizer": train["use_distributed_optimizer"],
            "recompute_granularity": "full" if train["use_activation_recompute"] else None,
            "recompute_method": "uniform" if train["use_activation_recompute"] else None,
            "recompute_num_layers": 1 if train["use_activation_recompute"] else None,
        },
        "data_args": {
            "tokenizer_type": "HuggingFaceTokenizer",
            "tokenizer_model": str(ir.model_dir),
            "vocab_size": model["vocab_size"],
            "data_path": "${DATA_PATH}",
            "split": "949,50,1",
            "data_cache_path": "${DATA_CACHE_PATH}",
        },
        "runtime_args": {
            "save": "${CHECKPOINT_DIR}",
            "load": "${CHECKPOINT_DIR}",
            "tensorboard_dir": "${TENSORBOARD_DIR}",
            "log_interval": 1,
            "eval_interval": 100,
            "eval_iters": 10,
            # 存档节奏（G7）：有通信/墙钟估算时按"约 1 小时存一次"推荐间隔；
            # 保留间隔取 10 倍（引擎要求是 save_interval 的整数倍），非保留
            # 倍数的上一存档由引擎滚动删除。--load + tracker 即自动续训。
            "save_interval": (
                int(ir.communication["save_interval_recommendation"])
                if ir.communication
                else 100
            ),
            "save_retain_interval": (
                int(ir.communication["save_retain_interval_recommendation"])
                if ir.communication
                else 1000
            ),
            "use_dist_ckpt": True,
            "async_save": True,
        },
        "parallel_plan": parallel_plan,
        "notes": ir.search_notes,
    }


def _flag(name: str) -> str:
    """把下划线参数转成 CLI flag。"""
    return "--" + name.replace("_", "-")


def _render_flag_block(payload: dict[str, object]) -> str:
    """渲染一组命令行参数。"""
    lines: list[str] = []
    for key, value in payload.items():
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                lines.append(f"    {_flag(key)} \\")
            continue
        lines.append(f"    {_flag(key)} {value} \\")
    return "\n".join(lines)


def _render_array_block(payload: dict[str, object]) -> str:
    """渲染 shell 数组风格参数。"""
    lines: list[str] = []
    for key, value in payload.items():
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                lines.append(f"    {_flag(key)}")
            continue
        lines.append(f"    {_flag(key)} {value}")
    return "\n".join(lines)


def _default_gpus_per_node(world_size: int, num_nodes: int, gpus_per_node: int | None) -> int:
    """计算 GPUS_PER_NODE 默认值。

    多机场景下默认值必须是 world_size / num_nodes，
    而不是旧的 world_size（那会把所有进程堆到单机上）。
    """
    if gpus_per_node is not None and gpus_per_node > 0:
        return int(gpus_per_node)
    num_nodes = max(int(num_nodes), 1)
    if num_nodes > 1 and world_size % num_nodes == 0:
        return world_size // num_nodes
    return world_size


def build_launch_script(
    ir: CoarseIR,
    megatron_config: dict[str, object],
    num_nodes: int = 1,
    gpus_per_node: int | None = None,
    master_addr: str | None = None,
    master_port: int | None = None,
) -> str:
    """生成 Megatron 启动脚本模板（多机感知；厂商档见 vendor_profile）。"""
    profile = current_profile(ir.environment.get("accelerator_type"))
    parallelism = ir.parallelism
    parallel_plan = ir.parallel_plan
    model_args = _render_flag_block(megatron_config["model_args"])
    parallel_args = _render_flag_block(megatron_config["parallelism"])
    training_args = _render_flag_block(megatron_config["training_args"])
    data_args = _render_array_block(megatron_config["data_args"])
    runtime_args = _render_flag_block(megatron_config["runtime_args"])

    world_size = int(parallelism["world_size"])
    ep_size = int(parallelism.get("expert_model_parallel_size", 1))
    default_gpus_per_node = _default_gpus_per_node(world_size, num_nodes, gpus_per_node)
    default_master_addr = master_addr or "127.0.0.1"
    default_master_port = master_port or 29500
    # 注记块：含通信估算（G2/G3）与存档节奏（G7）两行（无估算时为空）。
    notes_lines = "".join(f"# 注记: {note}\n" for note in megatron_config.get("notes", []))
    return f"""#!/usr/bin/env bash
set -euo pipefail

# 该脚本由 Megatron 自动并行 MVP 生成（支持多机）。
# 当前推荐模式: {ir.execution_mode}
# 搜索策略: {ir.search['strategy_name']} / {ir.search['strategy_policy']}
# 推荐并行: TP={parallelism['tp_size']}, DP={parallelism['dp_size']}, PP={parallelism['pp_size']}, EP={ep_size}, WORLD_SIZE={world_size}
{('# 通信估算(G2/G3 占位): 每卡每步 %.1f MB / %.3f s (tp %.3f, dp %.3f, pp %.3f, ep %.3f)，跨节点=%s' % (
    ir.communication['total_bytes'] / 1024**2,
    ir.communication['total_seconds'],
    ir.communication['tp_seconds'],
    ir.communication['dp_seconds'],
    ir.communication['pp_seconds'],
    ir.communication['ep_seconds'],
    [k for k, v in ir.communication['cross_node'].items() if v] or '无',
)) if ir.communication else '# 通信估算: 未计算'}
{('# 每步墙钟估算 %.3f s → 存档间隔 %d 步(约1h)/保留 %d 步，单份存档约 %.1f TB；--load 自动续训' % (
    ir.communication['estimated_step_seconds'],
    ir.communication['save_interval_recommendation'],
    ir.communication['save_retain_interval_recommendation'],
    ir.communication['estimated_checkpoint_bytes'] / 1024**4,
)) if ir.communication else '# 存档: 默认每 100 步/保留 1000 步'}
# 显式策略: attention={parallel_plan['attention_policy']}, mlp={parallel_plan['mlp_policy']}, embedding={parallel_plan['embedding_policy']}, lm_head={parallel_plan['lm_head_policy']}
# 三段式策略: attn=({parallel_plan['attention_policy_front']},{parallel_plan['attention_policy_mid']},{parallel_plan['attention_policy_back']}), mlp=({parallel_plan['mlp_policy_front']},{parallel_plan['mlp_policy_mid']},{parallel_plan['mlp_policy_back']})
{notes_lines}#
# 多机用法：在每台机器上以不同 NODE_RANK 启动同一份脚本：
#   node0: NNODES={max(num_nodes, 1)} NODE_RANK=0 MASTER_ADDR=<主节点IP> bash launch_megatron.sh
#   node1: NNODES={max(num_nodes, 1)} NODE_RANK=1 MASTER_ADDR=<主节点IP> bash launch_megatron.sh
# 也可以直接使用 tools/launch_cluster.py 按 hostfile 一键拉起全部节点。

ROOT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")/../.." && pwd)"
cd "${{ROOT_DIR}}"

source "${{ROOT_DIR}}/{profile.activate_script}"

export CUDA_DEVICE_MAX_CONNECTIONS="${{CUDA_DEVICE_MAX_CONNECTIONS:-1}}"
export TORCH_COMPILE_DISABLE="${{TORCH_COMPILE_DISABLE:-1}}"
export TORCHDYNAMO_DISABLE="${{TORCHDYNAMO_DISABLE:-1}}"
export MASTER_ADDR="${{MASTER_ADDR:-{default_master_addr}}}"
export MASTER_PORT="${{MASTER_PORT:-{default_master_port}}}"
export NNODES="${{NNODES:-{max(num_nodes, 1)}}}"
export NODE_RANK="${{NODE_RANK:-0}}"
export GPUS_PER_NODE="${{GPUS_PER_NODE:-{default_gpus_per_node}}}"
export WORLD_SIZE="$((NNODES * GPUS_PER_NODE))"
export DIST_TIMEOUT_MIN="${{DIST_TIMEOUT_MIN:-60}}"

export CHECKPOINT_DIR="${{CHECKPOINT_DIR:-${{ROOT_DIR}}/generated/megatron_mvp/checkpoints}}"
export TENSORBOARD_DIR="${{TENSORBOARD_DIR:-${{ROOT_DIR}}/generated/megatron_mvp/tensorboard}}"
export DATA_CACHE_PATH="${{DATA_CACHE_PATH:-${{ROOT_DIR}}/generated/megatron_mvp/data_cache}}"
export DATA_PATH="${{DATA_PATH:-MOCK}}"
mkdir -p "${{CHECKPOINT_DIR}}" "${{TENSORBOARD_DIR}}" "${{DATA_CACHE_PATH}}"

DATA_ARGS=()
if [[ "${{DATA_PATH}}" == "MOCK" ]]; then
  DATA_ARGS+=(
    --mock-data
    --tokenizer-type NullTokenizer
    --vocab-size {ir.model['vocab_size']}
  )
else
  DATA_ARGS+=(
{data_args}
  )
fi

python3 -m torch.distributed.run \
    --nproc_per_node "${{GPUS_PER_NODE}}" \
    --nnodes "${{NNODES}}" \
    --node_rank "${{NODE_RANK}}" \
    --master_addr "${{MASTER_ADDR}}" \
    --master_port "${{MASTER_PORT}}" \
    --rdzv-conf "timeout=${{RDZV_TIMEOUT:-2400}}" \
    "${{ROOT_DIR}}/{profile.run_wrapper}" Megatron-LM/pretrain_gpt.py \
    --rank "${{RANK:-0}}" \
    --world-size "${{WORLD_SIZE}}" \
    --dist-url "tcp://${{MASTER_ADDR}}:${{MASTER_PORT}}" \
    --distributed-timeout-minutes "${{DIST_TIMEOUT_MIN}}" \
{model_args}
{parallel_args}
{training_args}
{runtime_args}
    "${{DATA_ARGS[@]}}"
"""


def write_generation_outputs(
    artifacts: ModelArtifacts,
    spec: ModelSpec,
    env: RuntimeEnvironment,
    search_result: SearchResult,
    ir: CoarseIR,
    output_dir: Path,
    num_nodes: int = 1,
    gpus_per_node: int | None = None,
    master_addr: str | None = None,
    master_port: int | None = None,
) -> GenerationOutputs:
    """把检测、搜索和模板结果写入磁盘。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    detection_summary_path = output_dir / "detection_summary.json"
    search_result_path = output_dir / "search_result.json"
    coarse_ir_path = output_dir / "coarse_ir.json"
    megatron_config_path = output_dir / "megatron_config.json"
    launch_script_path = output_dir / "launch_megatron.sh"

    detection_summary = {
        "artifacts": artifacts.to_dict(),
        "model_spec": spec.to_dict(),
        "environment": env.to_dict(),
    }
    detection_summary_path.write_text(
        json.dumps(detection_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    search_result_path.write_text(
        json.dumps(search_result.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    coarse_ir_path.write_text(
        json.dumps(ir.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    megatron_config = build_megatron_config(ir)
    megatron_config_path.write_text(
        json.dumps(megatron_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    launch_script_path.write_text(
        build_launch_script(
            ir,
            megatron_config,
            num_nodes=num_nodes,
            gpus_per_node=gpus_per_node,
            master_addr=master_addr,
            master_port=master_port,
        ),
        encoding="utf-8",
    )
    launch_script_path.chmod(0o755)

    return GenerationOutputs(
        output_dir=output_dir,
        detection_summary_path=detection_summary_path,
        search_result_path=search_result_path,
        coarse_ir_path=coarse_ir_path,
        megatron_config_path=megatron_config_path,
        launch_script_path=launch_script_path,
    )
