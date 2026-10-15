"""把自动并行计划下放到不同 backend。"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from autoparallel.config import Config
from autoparallel.types import AutoParallelPlan
from autoparallel.types import LoweredProgram


def _flag(name: str) -> str:
    """把下划线参数名转成 CLI flag。"""
    return "--" + name.replace("_", "-")


def _render_flag_block(payload: dict[str, object]) -> str:
    """渲染普通命令行参数块。"""
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
    """渲染 shell 数组参数块。"""
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


def _count_nonempty_lines(text: str) -> int:
    """统计非空代码行数。"""
    return sum(1 for line in text.splitlines() if line.strip())


def _resolved_search_target_devices(config: Config) -> int | None:
    """返回示例代码里更直观的规划卡数。"""
    if config.search_target_devices is not None:
        return config.search_target_devices
    return config.num_devices


def _default_gpus_per_node(config: Config, default_world_size: int) -> int:
    """计算启动脚本 GPUS_PER_NODE 的默认值。

    多机场景（config.num_nodes > 1）默认值 = world_size / num_nodes；
    旧行为（单机）保持 world_size。这个默认值直接影响
    torch.distributed.run 的 --nproc_per_node，是框架从"单机多卡"
    扩展到"多机多卡"的关键修正点。
    """
    num_nodes = config.resolved_num_nodes()
    explicit = config.resolved_gpus_per_node()
    if explicit is not None:
        return explicit
    if num_nodes > 1 and default_world_size % num_nodes == 0:
        return default_world_size // num_nodes
    return default_world_size


def _build_hostfile_template(config: Config, default_world_size: int) -> str:
    """生成多机 hostfile 模板，供 tools/launch_cluster.py 消费。"""
    num_nodes = config.resolved_num_nodes()
    gpus_per_node = _default_gpus_per_node(config, default_world_size)
    lines = [
        "# autoparallel 生成的多机 hostfile 模板。",
        "# 格式: 每行一个节点 `hostname[:ssh_port] [slots=N]`，# 开头为注释。",
        "# 第一行是主节点（MASTER_ADDR / NODE_RANK=0）。",
        "# 请把下面的 localhost 替换成真实节点主机名或 IP。",
    ]
    lines.extend(
        [
            "localhost slots=%d  # node%d -> 替换为真实主机名" % (gpus_per_node, index)
            for index in range(num_nodes)
        ]
    )
    return "\n".join(lines) + "\n"


def _build_cluster_launcher_script(
    hostfile_name: str,
    launch_script_path: Path,
    default_gpus_per_node: int,
) -> str:
    """生成调用 tools/launch_cluster.py 的一键多机拉起脚本。"""
    return f'''#!/usr/bin/env bash
set -euo pipefail

# 该脚本由 autoparallel 生成：通过 SSH 按 hostfile 在所有节点拉起训练。
# 用法（在主节点执行）:
#   bash launch_cluster.sh [--dry-run]
#
# 前置条件:
#   1. 所有节点可通过 ssh 免密登录（端口可用 hostfile 的 host:port 指定）。
#   2. 所有节点在相同路径部署了本仓库（NFS 或手动同步）。
#   3. hostfile 已把 localhost 替换成真实主机名。

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
PROJECT_ROOT="$(cd "${{SCRIPT_DIR}}/../.." && pwd)"

python3 "${{PROJECT_ROOT}}/tools/launch_cluster.py" \\
    --hostfile "${{SCRIPT_DIR}}/{hostfile_name}" \\
    --script "${{launch_script_path}}" \\
    --gpus-per-node "${{GPUS_PER_NODE:-{default_gpus_per_node}}}" \\
    "$@"
'''


def _build_runtime_megatron_config(
    config: Config,
    plan: AutoParallelPlan,
    script_role: str,
) -> tuple[dict[str, Any], int, str]:
    """根据声明式配置修正 Megatron 配置。"""
    megatron_config = deepcopy(plan.megatron_config)
    megatron_config["entrypoint"] = config.entrypoint

    model_args = megatron_config["model_args"]
    training_args = megatron_config["training_args"]
    data_args = megatron_config["data_args"]
    runtime_args = megatron_config["runtime_args"]

    if config.sequence_length is not None:
        model_args["seq_length"] = config.sequence_length
        model_args["max_position_embeddings"] = max(
            int(model_args["max_position_embeddings"]),
            config.sequence_length,
        )

    training_args["train_iters"] = config.train_iters
    training_args["lr"] = config.lr
    training_args["min_lr"] = config.min_lr
    training_args["lr_decay_style"] = "cosine"
    training_args["weight_decay"] = config.weight_decay
    training_args["clip_grad"] = 1.0
    training_args["adam_beta1"] = 0.9
    training_args["adam_beta2"] = 0.95
    training_args["attention_dropout"] = 0.0
    training_args["hidden_dropout"] = 0.0
    training_args["init_method_std"] = 0.02

    runtime_args["log_interval"] = config.log_interval
    runtime_args["save_interval"] = config.save_interval
    runtime_args["eval_iters"] = config.eval_iters
    # Megatron 的训练代码里会直接执行 `train_iters // eval_interval`。
    # 因此“关闭评估”不能简单传 `eval_interval=0`，否则会触发除零。
    # 这里统一把“禁用评估”映射成：
    # 1. `eval_iters=0`，确保 do_valid/do_test 为 False。
    # 2. `eval_interval=1`，避免内部整除零。
    if config.eval_iters == 0 and config.eval_interval == 0:
        runtime_args["eval_interval"] = 1
    else:
        runtime_args["eval_interval"] = config.eval_interval
    runtime_args["save"] = "${CHECKPOINT_DIR}"
    runtime_args["load"] = None
    runtime_args["tensorboard_dir"] = "${TENSORBOARD_DIR}"

    if config.use_mock_data:
        data_args.clear()
        data_args.update(
            {
                "mock_data": True,
                "tokenizer_type": "NullTokenizer",
                "vocab_size": 32000,
            }
        )
    else:
        tokenizer_path = config.tokenizer_path or str(config.model_path)
        data_args["tokenizer_type"] = "HuggingFaceTokenizer"
        data_args["tokenizer_model"] = tokenizer_path
        data_args["data_path"] = (
            str(config.data_path) if config.data_path is not None else "${DATA_PATH}"
        )
        data_args["split"] = "949,50,1"
        data_args["data_cache_path"] = str(config.data_cache_path)

    if script_role == "planning":
        return (
            megatron_config,
            plan.parallel_spec.world_size,
            "该脚本默认按规划结果启动，用于未来双卡/多卡正式执行。",
        )

    # smoke test 的目标是验证真实训练链路，而不是验证 checkpoint 序列化。
    # 当前海光环境在训练完成后的保存阶段会触发段错误，因此这里默认关闭保存，
    # 避免把“训练后存档崩溃”误判成“训练链路未打通”。
    runtime_args["save"] = None

    runtime_test_devices = config.resolved_runtime_test_devices()
    if runtime_test_devices >= plan.parallel_spec.world_size:
        return (
            megatron_config,
            plan.parallel_spec.world_size,
            "当前 smoke test 卡数已覆盖规划 WORLD_SIZE，直接复用规划并行配置，并默认关闭 checkpoint 保存。",
        )

    megatron_config["parallelism"] = {
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "expert_model_parallel_size": 1,
        "sequence_parallel": False,
    }
    training_args["global_batch_size"] = int(training_args["micro_batch_size"])
    training_args["use_distributed_optimizer"] = False
    return (
        megatron_config,
        1,
        "当前 smoke test 默认回退到单卡串行配置，只验证数据链路和 Megatron 启动链路。",
    )


def _build_launch_script(
    config: Config,
    plan: AutoParallelPlan,
    megatron_config: dict[str, Any],
    default_world_size: int,
    project_root: Path,
    script_role: str,
    runtime_note: str,
    script_name: str = "launch_megatron_plan.sh",
) -> str:
    """构建 Megatron 下放启动脚本（多机感知）。"""
    model_args = _render_flag_block(megatron_config["model_args"])
    parallel_args = _render_flag_block(megatron_config["parallelism"])
    training_args = _render_flag_block(megatron_config["training_args"])
    data_args = _render_array_block(megatron_config["data_args"])
    runtime_args = _render_flag_block(megatron_config["runtime_args"])
    data_cache_path = config.resolved_data_cache_path()

    num_nodes = config.resolved_num_nodes()
    default_gpus_per_node = _default_gpus_per_node(config, default_world_size)
    default_master_addr = config.master_addr or "127.0.0.1"
    default_master_port = config.master_port or 29500

    data_path_export = ""
    if not config.use_mock_data and config.data_path is not None:
        data_path_export = f'export DATA_PATH="${{DATA_PATH:-{config.data_path}}}"\n'

    return f'''#!/usr/bin/env bash
set -euo pipefail

# 该脚本由 autoparallel 声明式前端生成，并下放到现有 Megatron 执行（支持多机）。
# 脚本角色: {script_role}
# 搜索策略: {plan.search_summary['selected_strategy']} / {plan.search_summary['selected_policy']}
# 搜索耗时: {plan.search_summary['selected_search_time_seconds']:.4f} 秒
# 规划推荐: TP={plan.parallel_spec.tp_size}, DP={plan.parallel_spec.dp_size}, PP={plan.parallel_spec.pp_size}, WORLD_SIZE={plan.parallel_spec.world_size}
# 显式策略: attention={plan.parallel_plan.attention_policy}, mlp={plan.parallel_plan.mlp_policy}, embedding={plan.parallel_plan.embedding_policy}, lm_head={plan.parallel_plan.lm_head_policy}
# 三段式策略: attn=({plan.parallel_plan.resolved_attention_policy_front()},{plan.parallel_plan.resolved_attention_policy_mid()},{plan.parallel_plan.resolved_attention_policy_back()}), mlp=({plan.parallel_plan.resolved_mlp_policy_front()},{plan.parallel_plan.resolved_mlp_policy_mid()},{plan.parallel_plan.resolved_mlp_policy_back()})
# 当前执行: {runtime_note}
#
# 单机用法:  bash {script_name}
# 多机用法:  在每台节点上用不同 NODE_RANK 启动同一份脚本:
#   node0: NNODES={num_nodes} NODE_RANK=0 MASTER_ADDR=<主节点IP> bash {script_name}
#   node1: NNODES={num_nodes} NODE_RANK=1 MASTER_ADDR=<主节点IP> bash {script_name}
# 或在主节点直接: bash launch_cluster.sh (由 hostfile 驱动 tools/launch_cluster.py)

ROOT_DIR="{project_root}"
cd "${{ROOT_DIR}}"

source "${{ROOT_DIR}}/scripts/activate_megatron_hygon.sh"

export AUTOPARALLEL_ROLE="{script_role}"
export CUDA_DEVICE_MAX_CONNECTIONS="${{CUDA_DEVICE_MAX_CONNECTIONS:-1}}"
export TORCH_COMPILE_DISABLE="${{TORCH_COMPILE_DISABLE:-1}}"
export TORCHDYNAMO_DISABLE="${{TORCHDYNAMO_DISABLE:-1}}"
export MASTER_ADDR="${{MASTER_ADDR:-{default_master_addr}}}"
export MASTER_PORT="${{MASTER_PORT:-{default_master_port}}}"
export NNODES="${{NNODES:-{num_nodes}}}"
export NODE_RANK="${{NODE_RANK:-0}}"
export GPUS_PER_NODE="${{GPUS_PER_NODE:-{default_gpus_per_node}}}"
export WORLD_SIZE="$((NNODES * GPUS_PER_NODE))"
export DIST_TIMEOUT_MIN="${{DIST_TIMEOUT_MIN:-60}}"

export CHECKPOINT_DIR="${{CHECKPOINT_DIR:-${{ROOT_DIR}}/generated/autoparallel/checkpoints}}"
export TENSORBOARD_DIR="${{TENSORBOARD_DIR:-${{ROOT_DIR}}/generated/autoparallel/tensorboard}}"
export DATA_CACHE_PATH="${{DATA_CACHE_PATH:-{data_cache_path}}}"
{data_path_export}mkdir -p "${{CHECKPOINT_DIR}}" "${{TENSORBOARD_DIR}}" "${{DATA_CACHE_PATH}}"

DATA_ARGS=(
{data_args}
)

python3 -m torch.distributed.run \
    --nproc_per_node "${{GPUS_PER_NODE}}" \
    --nnodes "${{NNODES}}" \
    --node_rank "${{NODE_RANK}}" \
    --master_addr "${{MASTER_ADDR}}" \
    --master_port "${{MASTER_PORT}}" \
    "${{ROOT_DIR}}/tools/run_megatron_hygon.py" {config.entrypoint} \
    --rank "${{RANK:-0}}" \
    --world-size "${{WORLD_SIZE}}" \
    --dist-url "tcp://${{MASTER_ADDR}}:${{MASTER_PORT}}" \
    --distributed-timeout-minutes "${{DIST_TIMEOUT_MIN}}" \
{model_args}
{parallel_args}
{training_args}
{runtime_args}
    "${{DATA_ARGS[@]}}"
'''


def _build_manual_example(config: Config, plan: AutoParallelPlan) -> str:
    """构建手动显式 PP/TP 并行参考代码。"""
    return f'''"""手动显式并行实现示例。

这个文件故意保留大量并行细节，作为和声明式 autoparallel 的对照基线。
"""

from __future__ import annotations

import torch
import torch.nn as nn


TP_SIZE = {plan.parallel_spec.tp_size}
PP_SIZE = max({plan.parallel_spec.pp_size}, 1)
NUM_LAYERS = {plan.model_summary['num_layers']}
HIDDEN_SIZE = {plan.model_summary['hidden_size']}
FFN_HIDDEN_SIZE = {plan.model_summary['intermediate_size']}
NUM_HEADS = {plan.model_summary['num_heads']}
SEQ_LENGTH = {plan.megatron_config['model_args']['seq_length']}
VOCAB_SIZE = {plan.megatron_config['data_args'].get('vocab_size', 32000)}


def get_pp_rank() -> int:
    """返回当前 PP rank。"""
    return 0


def get_tp_rank() -> int:
    """返回当前 TP rank。"""
    return 0


def pp_send(tensor: torch.Tensor) -> None:
    """显式发送 PP 激活。"""
    _ = tensor


def pp_recv(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    """显式接收 PP 激活。"""
    return torch.zeros(shape, device=device)


def AllReduce(tensor: torch.Tensor, group: object) -> torch.Tensor:
    """显式 TP AllReduce。"""
    _ = group
    return tensor


def build_pp_info(num_layers: int, pp_size: int) -> dict[str, list[int]]:
    """手工构造 PP 切分信息。"""
    base = num_layers // max(pp_size, 1)
    remain = num_layers % max(pp_size, 1)
    layer_splits = [base] * max(pp_size, 1)
    for idx in range(remain):
        layer_splits[idx] += 1
    return {{"num_layers": layer_splits}}


def build_tp_info(tp_size: int) -> dict[str, int]:
    """手工构造 TP 切分信息。"""
    return {{"tp_size": tp_size, "rank": get_tp_rank()}}


class LlamaEmbedding(nn.Module):
    """手工实现的 embedding 层。"""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(VOCAB_SIZE, HIDDEN_SIZE))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """执行 embedding lookup。"""
        return self.weight[input_ids]


class LlamaAttentionBlock(nn.Module):
    """手工显式 TP Attention block。"""

    def __init__(self, tp_info: dict[str, int], device_tp: torch.device) -> None:
        super().__init__()
        tp_size = tp_info["tp_size"]

        # attention 的 q/k/v/o 都需要手工按 TP 维度切分。
        self.q_proj = nn.Parameter(torch.empty(HIDDEN_SIZE, HIDDEN_SIZE // tp_size, device=device_tp))
        self.k_proj = nn.Parameter(torch.empty(HIDDEN_SIZE, HIDDEN_SIZE // tp_size, device=device_tp))
        self.v_proj = nn.Parameter(torch.empty(HIDDEN_SIZE, HIDDEN_SIZE // tp_size, device=device_tp))
        self.o_proj = nn.Parameter(torch.empty(HIDDEN_SIZE // tp_size, HIDDEN_SIZE, device=device_tp))

        # MLP 也需要手工按 TP 维度切分。
        self.gate_proj = nn.Parameter(torch.empty(HIDDEN_SIZE, FFN_HIDDEN_SIZE // tp_size, device=device_tp))
        self.up_proj = nn.Parameter(torch.empty(HIDDEN_SIZE, FFN_HIDDEN_SIZE // tp_size, device=device_tp))
        self.down_proj = nn.Parameter(torch.empty(FFN_HIDDEN_SIZE // tp_size, HIDDEN_SIZE, device=device_tp))

        for param in [
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.o_proj,
            self.gate_proj,
            self.up_proj,
            self.down_proj,
        ]:
            nn.init.normal_(param, mean=0.0, std=0.02)

    def forward(self, hidden_states: torch.Tensor, tp_group: object) -> torch.Tensor:
        """显式实现 TP 计算和通信。"""
        q_local = hidden_states @ self.q_proj
        k_local = hidden_states @ self.k_proj
        v_local = hidden_states @ self.v_proj
        attn_local = q_local + k_local + v_local
        attn_local = attn_local @ self.o_proj

        # attention 输出需要手工 AllReduce。
        attn_output = AllReduce(attn_local, tp_group)

        gate_local = hidden_states @ self.gate_proj
        up_local = hidden_states @ self.up_proj
        mlp_local = (gate_local.sigmoid() * gate_local) * up_local
        mlp_local = mlp_local @ self.down_proj

        # MLP 输出也需要手工 AllReduce。
        mlp_output = AllReduce(mlp_local, tp_group)
        return attn_output + mlp_output


class LlamaModel(nn.Module):
    """手动显式 PP/TP 并行模型。"""

    def __init__(self, config: dict[str, int], pp_info: dict[str, list[int]], tp_info: dict[str, int]) -> None:
        super().__init__()
        self.config = config
        self.pp_info = pp_info
        self.tp_info = tp_info
        self.device_tp = torch.device("cuda:0")
        self.tp_group = object()
        self.pp_first_stage = get_pp_rank() == 0
        self.pp_last_stage = get_pp_rank() == len(pp_info["num_layers"]) - 1

        self.emb = LlamaEmbedding()
        self.layers = self.build_layers(pp_info, tp_info)
        self.final_norm = nn.LayerNorm(HIDDEN_SIZE)
        self.lm_head = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE, bias=False)

    def build_layers(self, pp_info: dict[str, list[int]], tp_info: dict[str, int]) -> nn.ModuleList:
        """按照 PP/TP 信息手工构造本 stage 的层。"""
        rank_id = get_pp_rank()
        pp_num_layers = pp_info["num_layers"][rank_id]
        layers = nn.ModuleList()

        # 每一层都需要开发者显式手工做 attention/MLP shard。
        for _ in range(pp_num_layers):
            layers.append(LlamaAttentionBlock(tp_info=tp_info, device_tp=self.device_tp))
        return layers

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """显式实现 PP/TP 前向逻辑。"""
        if self.pp_first_stage:
            hidden_states = self.emb(input_ids)
        else:
            hidden_states = pp_recv((input_ids.shape[0], SEQ_LENGTH, HIDDEN_SIZE), self.device_tp)

        # 每一层都显式调用 TP 计算和通信。
        for layer in self.layers:
            hidden_states = layer(hidden_states, self.tp_group)

        output_layer = self.final_norm(hidden_states)
        if not self.pp_last_stage:
            pp_send(output_layer)
            return output_layer
        return self.lm_head(output_layer)


def build_optimizer(model: LlamaModel) -> torch.optim.Optimizer:
    """手工为切分参数构造优化器。"""
    return torch.optim.AdamW(model.parameters(), lr={config.lr}, weight_decay={config.weight_decay})


def train_step() -> None:
    """手工训练入口。"""
    config = {{
        "hidden_size": HIDDEN_SIZE,
        "ffn_hidden_size": FFN_HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "num_heads": NUM_HEADS,
    }}
    pp_info = build_pp_info(NUM_LAYERS, PP_SIZE)
    tp_info = build_tp_info(TP_SIZE)

    model = LlamaModel(config=config, pp_info=pp_info, tp_info=tp_info)
    optimizer = build_optimizer(model)

    input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LENGTH), device=torch.device("cuda:0"))
    logits = model(input_ids)
    loss = logits.float().mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


if __name__ == "__main__":
    train_step()
'''


def _build_declarative_example(config: Config) -> str:
    """构建声明式自动并行示例。"""
    model_literal = repr(str(config.model_path))
    prefer_literal = repr(config.prefer)
    data_literal = repr(str(config.data_path) if config.data_path is not None else None)
    tokenizer_literal = repr(str(config.tokenizer_path) if config.tokenizer_path is not None else None)
    memory_literal = "None" if config.memory_target_gb is None else str(config.memory_target_gb)
    search_target_devices = _resolved_search_target_devices(config)
    search_target_literal = "None" if search_target_devices is None else str(search_target_devices)
    runtime_test_literal = str(config.resolved_runtime_test_devices())
    search_strategy_literal = repr(config.search_strategy)
    search_policy_literal = repr(config.search_policy)
    return f'''"""声明式 autoparallel 示例。"""

from __future__ import annotations

from autoparallel import Config
from autoparallel import ParallelPlan
from autoparallel import parallelize


config = Config(
    model_path={model_literal},
    search_target_devices={search_target_literal},
    runtime_test_devices={runtime_test_literal},
    memory_target_gb={memory_literal},
    prefer={prefer_literal},
    search_strategy={search_strategy_literal},
    search_policy={search_policy_literal},
    data_path={data_literal},
    tokenizer_path={tokenizer_literal},
    run_mode="LAUNCH",
    train_iters={config.train_iters},
)
config.parallel_mode = "MODEL_PARALLEL"

# 如果你想显式固定搜索后的方案，也可以直接使用 ParallelPlan(...)：
# explicit_plan = ParallelPlan(
#     model_path={model_literal},
#     tp=1,
#     dp=1,
#     attention_policy="replicate",
#     mlp_policy="replicate",
#     embedding_policy="replicate",
#     lm_head_policy="replicate",
# )
# @parallelize(explicit_plan)
# def train() -> None:
#     pass


@parallelize(config)
def train() -> None:
    """这里的函数体只保留训练意图，不手写 PP/TP shard 和通信细节。"""
    pass


if __name__ == "__main__":
    train()
'''


def lower_to_megatron(config: Config, plan: AutoParallelPlan) -> LoweredProgram:
    """把自动并行计划下放到现有 Megatron。"""
    normalized = config.normalized()
    project_root = Path(__file__).resolve().parent.parent
    output_dir = normalized.resolved_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    examples_dir = output_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    planning_megatron_config, planning_world_size, planning_note = _build_runtime_megatron_config(
        normalized,
        plan,
        script_role="planning",
    )
    smoke_megatron_config, smoke_world_size, smoke_note = _build_runtime_megatron_config(
        normalized,
        plan,
        script_role="smoke_test",
    )

    planning_launch_script = _build_launch_script(
        normalized,
        plan,
        planning_megatron_config,
        planning_world_size,
        project_root=project_root,
        script_role="planning",
        runtime_note=planning_note,
        script_name="launch_megatron_plan.sh",
    )
    smoke_launch_script = _build_launch_script(
        normalized,
        plan,
        smoke_megatron_config,
        smoke_world_size,
        project_root=project_root,
        script_role="smoke_test",
        runtime_note=smoke_note,
        script_name="launch_megatron_smoke.sh",
    )

    launch_script_path = output_dir / "launch_megatron_smoke.sh"
    megatron_config_path = output_dir / "megatron_smoke_config.json"
    planning_launch_script_path = output_dir / "launch_megatron_plan.sh"
    planning_megatron_config_path = output_dir / "megatron_plan_config.json"
    manual_example_path = examples_dir / "manual_megatron_train.py"
    declarative_example_path = examples_dir / "declarative_autoparallel_train.py"

    launch_script_path.write_text(smoke_launch_script, encoding="utf-8")
    launch_script_path.chmod(0o755)
    megatron_config_path.write_text(
        json.dumps(smoke_megatron_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    planning_launch_script_path.write_text(planning_launch_script, encoding="utf-8")
    planning_launch_script_path.chmod(0o755)
    planning_megatron_config_path.write_text(
        json.dumps(planning_megatron_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 多机模式：额外生成 hostfile 模板 + 一键 SSH 拉起脚本。
    cluster_artifacts: dict[str, str] = {}
    if normalized.resolved_num_nodes() > 1:
        hostfile_path = output_dir / "hostfile"
        cluster_script_path = output_dir / "launch_cluster.sh"
        hostfile_path.write_text(
            _build_hostfile_template(normalized, planning_world_size),
            encoding="utf-8",
        )
        cluster_script_path.write_text(
            _build_cluster_launcher_script(
                hostfile_name="hostfile",
                launch_script_path=planning_launch_script_path,
                default_gpus_per_node=_default_gpus_per_node(normalized, planning_world_size),
            ),
            encoding="utf-8",
        )
        cluster_script_path.chmod(0o755)
        cluster_artifacts = {
            "hostfile_path": str(hostfile_path),
            "cluster_launcher_path": str(cluster_script_path),
        }

    manual_example = _build_manual_example(normalized, plan)
    declarative_example = _build_declarative_example(normalized)
    manual_example_path.write_text(manual_example, encoding="utf-8")
    declarative_example_path.write_text(declarative_example, encoding="utf-8")

    manual_lines = _count_nonempty_lines(manual_example)
    declarative_lines = _count_nonempty_lines(declarative_example)
    reduction_ratio = 0.0
    if manual_lines > 0:
        reduction_ratio = (manual_lines - declarative_lines) / manual_lines

    notes = [
        "运行时不自建 distributed runtime，而是把计划真实下放到现有 Megatron。",
        "默认 launch_program 执行 smoke_test 脚本；规划脚本单独保留，等待后续真双卡环境。",
        f"规划脚本默认 WORLD_SIZE={planning_world_size}，smoke_test 默认 WORLD_SIZE={smoke_world_size}。",
        smoke_note,
        "manual/declarative 两个 py 文件已生成，可直接比较代码量。",
    ]
    if cluster_artifacts:
        notes.append(
            "已生成多机 hostfile 模板与 launch_cluster.sh 一键拉起脚本："
            "请先把 hostfile 中的 localhost 替换成真实节点名，再在主节点执行 bash launch_cluster.sh。"
        )

    return LoweredProgram(
        backend="MEGATRON",
        project_root=project_root,
        output_dir=output_dir,
        launch_script_path=launch_script_path,
        runtime_config_path=megatron_config_path,
        planning_launch_script_path=planning_launch_script_path,
        planning_config_path=planning_megatron_config_path,
        manual_example_path=manual_example_path,
        declarative_example_path=declarative_example_path,
        command=["bash", str(launch_script_path)],
        env_overrides={"CUDA_DEVICE_MAX_CONNECTIONS": "1"},
        code_reduction_summary={
            "manual_nonempty_lines": manual_lines,
            "declarative_nonempty_lines": declarative_lines,
            "reduction_ratio": reduction_ratio,
        },
        notes=notes,
        extra_artifact_paths=cluster_artifacts,
    )


def _round_up_to_multiple(value: int, divisor: int) -> int:
    """把整数向上取整到指定倍数。"""
    if divisor <= 1:
        return value
    return ((value + divisor - 1) // divisor) * divisor


def _build_runtime_parallel_plan_for_pytorch(
    config: Config,
    plan: AutoParallelPlan,
    script_role: str,
):
    """为 PyTorch backend 选择真正执行时使用的并行方案。"""
    if script_role == "planning":
        return (
            plan.parallel_plan,
            plan.parallel_spec.world_size,
            "该脚本默认按规划结果启动，用于未来双卡/多卡正式执行。",
        )

    runtime_test_devices = config.resolved_runtime_test_devices()
    if runtime_test_devices >= plan.parallel_spec.world_size:
        return (
            plan.parallel_plan,
            plan.parallel_spec.world_size,
            "当前 smoke test 卡数已覆盖规划 WORLD_SIZE，直接复用规划并行配置。",
        )

    runtime_parallel_plan = plan.parallel_plan.with_updates(
        tp=1,
        dp=1,
        pp=1,
        attention_policy="replicate",
        mlp_policy="replicate",
        attention_policy_front="replicate",
        attention_policy_mid="replicate",
        attention_policy_back="replicate",
        mlp_policy_front="replicate",
        mlp_policy_mid="replicate",
        mlp_policy_back="replicate",
        embedding_policy="replicate",
        lm_head_policy="replicate",
        sequence_parallel=False,
        global_batch_size=int(plan.parallel_plan.micro_batch_size),
        parallel_mode="DATA_PARALLEL",
        execution_mode="current_feasible",
        fits_current_environment=True,
        requires_additional_devices=0,
        notes=tuple(plan.parallel_plan.notes)
        + ("PyTorch backend smoke_test 回退到单进程串行运行。",),
    )
    return (
        runtime_parallel_plan,
        1,
        "当前 smoke test 默认回退到单进程串行运行，并使用截断版真实 HF LLaMA 权重做最小训练验证。",
    )


def _build_pytorch_runtime_model_config(
    config: Config,
    plan: AutoParallelPlan,
    runtime_parallel_plan,
    script_role: str,
) -> dict[str, int]:
    """构造 PyTorch backend 使用的真实 LLaMA 运行配置。"""
    target_seq_length = config.sequence_length or int(plan.megatron_config["model_args"]["seq_length"])
    full_num_layers = int(plan.model_summary["num_layers"])
    using_full_parallel_plan = (
        int(runtime_parallel_plan.tp) == int(plan.parallel_plan.tp)
        and int(runtime_parallel_plan.dp) == int(plan.parallel_plan.dp)
    )
    if script_role == "planning" or using_full_parallel_plan:
        effective_num_layers = full_num_layers
        effective_seq_length = int(target_seq_length)
        train_iters = config.train_iters
    else:
        effective_num_layers = min(full_num_layers, 2)
        effective_seq_length = min(int(target_seq_length), 16)
        train_iters = min(config.train_iters, 1)

    return {
        "runtime_kind": "HF_LLAMA_TRUNCATED" if effective_num_layers < full_num_layers else "HF_LLAMA_FULL",
        "full_num_layers": full_num_layers,
        "effective_num_layers": effective_num_layers,
        "effective_seq_length": effective_seq_length,
        "parameter_dtype": "bfloat16",
        "train_iters": train_iters,
        "micro_batch_size": runtime_parallel_plan.micro_batch_size,
    }


def _build_runtime_pytorch_config(
    config: Config,
    plan: AutoParallelPlan,
    script_role: str,
) -> tuple[dict[str, Any], int, str]:
    """根据声明式配置生成 PyTorch backend 运行配置。"""
    runtime_parallel_plan, default_world_size, runtime_note = _build_runtime_parallel_plan_for_pytorch(
        config=config,
        plan=plan,
        script_role=script_role,
    )
    runtime_model = _build_pytorch_runtime_model_config(
        config=config,
        plan=plan,
        runtime_parallel_plan=runtime_parallel_plan,
        script_role=script_role,
    )
    runtime_config = {
        "backend": "PYTORCH",
        "script_role": script_role,
        "runtime_note": runtime_note,
        "model_path": str(plan.model_path),
        "planned_parallel_plan": plan.parallel_plan.to_dict(),
        "runtime_parallel_plan": runtime_parallel_plan.to_dict(),
        "model_summary": dict(plan.model_summary),
        "runtime_model": runtime_model,
        "train": {
            "train_iters": runtime_model["train_iters"],
            "micro_batch_size": runtime_parallel_plan.micro_batch_size,
            "global_batch_size": runtime_parallel_plan.global_batch_size,
            "seq_length": runtime_model["effective_seq_length"],
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "log_interval": config.log_interval,
            # 默认随机初始化，与 Megatron `init_method_std=0.02` 对齐，
            # 避免出现“一边预训练权重、一边随机初始化”的不可比 loss。
            "use_real_hf_weights": bool(
                config.extra.get("use_real_hf_weights", False)
            ),
            "init_method_std": float(config.extra.get("init_method_std", 0.02)),
            # 与 Megatron backend 共享同一套数据开关：
            # use_mock_data=False 且 data_path 指向 Megatron indexed dataset 前缀时，
            # PyTorch runtime 会直接读取同一份 wikitext2 token 流。
            "use_mock_data": bool(config.use_mock_data),
            "data_path": (
                str(config.data_path) if config.data_path is not None else None
            ),
        },
    }
    return runtime_config, default_world_size, runtime_note


def _build_pytorch_launch_script(
    config: Config,
    plan: AutoParallelPlan,
    runtime_config_path: Path,
    default_world_size: int,
    project_root: Path,
    script_role: str,
    runtime_note: str,
) -> str:
    """构建 PyTorch backend 启动脚本（多机感知）。"""
    num_nodes = config.resolved_num_nodes()
    default_gpus_per_node = _default_gpus_per_node(config, default_world_size)
    default_master_addr = config.master_addr or "127.0.0.1"
    default_master_port = config.master_port or 29520
    return f'''#!/usr/bin/env bash
set -euo pipefail

# 该脚本由 autoparallel 声明式前端生成，并下放到 PyTorch backend runtime（支持多机）。
# 脚本角色: {script_role}
# 搜索策略: {plan.search_summary['selected_strategy']} / {plan.search_summary['selected_policy']}
# 搜索耗时: {plan.search_summary['selected_search_time_seconds']:.4f} 秒
# 规划推荐: TP={plan.parallel_spec.tp_size}, DP={plan.parallel_spec.dp_size}, WORLD_SIZE={plan.parallel_spec.world_size}
# 当前执行: {runtime_note}
#
# 多机用法: 每台节点用不同 NODE_RANK 启动同一份脚本:
#   node0: NNODES={num_nodes} NODE_RANK=0 MASTER_ADDR=<主节点IP> bash launch_pytorch_plan.sh
#   node1: NNODES={num_nodes} NODE_RANK=1 MASTER_ADDR=<主节点IP> bash launch_pytorch_plan.sh

ROOT_DIR="{project_root}"
cd "${{ROOT_DIR}}"

source "${{ROOT_DIR}}/scripts/activate_megatron_hygon.sh"

export AUTOPARALLEL_ROLE="{script_role}"
export MASTER_ADDR="${{MASTER_ADDR:-{default_master_addr}}}"
export MASTER_PORT="${{MASTER_PORT:-{default_master_port}}}"
export NNODES="${{NNODES:-{num_nodes}}}"
export NODE_RANK="${{NODE_RANK:-0}}"
export GPUS_PER_NODE="${{GPUS_PER_NODE:-{default_gpus_per_node}}}"
export WORLD_SIZE="$((NNODES * GPUS_PER_NODE))"

python3 -m torch.distributed.run \
    --nproc_per_node "${{GPUS_PER_NODE}}" \
    --nnodes "${{NNODES}}" \
    --node_rank "${{NODE_RANK}}" \
    --master_addr "${{MASTER_ADDR}}" \
    --master_port "${{MASTER_PORT}}" \
    "${{ROOT_DIR}}/tools/run_pytorch_llama_tp_dp.py" \
    --config "{runtime_config_path}"
'''


def _build_pytorch_inspect_script(
    runtime_config_path: Path,
    project_root: Path,
    script_role: str,
) -> str:
    """构建 PyTorch backend 的静态 inspect-only 脚本。"""
    return f'''#!/usr/bin/env bash
set -euo pipefail

# 该脚本不会真正启动 distributed 训练。
# 它只读取 lowering 生成的 planning/runtime 配置，并输出静态切分摘要。
# 脚本角色: {script_role}

ROOT_DIR="{project_root}"
cd "${{ROOT_DIR}}"

source "${{ROOT_DIR}}/scripts/activate_megatron_hygon.sh"

python3 "${{ROOT_DIR}}/tools/run_pytorch_llama_tp_dp.py" \
    --config "{runtime_config_path}" \
    --inspect-only
'''


def lower_to_pytorch(config: Config, plan: AutoParallelPlan) -> LoweredProgram:
    """把自动并行计划下放到 PyTorch backend 运行时。"""
    normalized = config.normalized()
    project_root = Path(__file__).resolve().parent.parent
    output_dir = normalized.resolved_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    examples_dir = output_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    planning_runtime_config, planning_world_size, planning_note = _build_runtime_pytorch_config(
        normalized,
        plan,
        script_role="planning",
    )
    smoke_runtime_config, smoke_world_size, smoke_note = _build_runtime_pytorch_config(
        normalized,
        plan,
        script_role="smoke_test",
    )

    launch_script_path = output_dir / "launch_pytorch_smoke.sh"
    runtime_config_path = output_dir / "pytorch_smoke_config.json"
    planning_launch_script_path = output_dir / "launch_pytorch_plan.sh"
    planning_config_path = output_dir / "pytorch_plan_config.json"
    inspect_script_path = output_dir / "inspect_pytorch_smoke.sh"
    planning_inspect_script_path = output_dir / "inspect_pytorch_plan.sh"
    manual_example_path = examples_dir / "manual_parallel_train.py"
    declarative_example_path = examples_dir / "declarative_autoparallel_train.py"

    smoke_launch_script = _build_pytorch_launch_script(
        normalized,
        plan,
        runtime_config_path=runtime_config_path,
        default_world_size=smoke_world_size,
        project_root=project_root,
        script_role="smoke_test",
        runtime_note=smoke_note,
    )
    planning_launch_script = _build_pytorch_launch_script(
        normalized,
        plan,
        runtime_config_path=planning_config_path,
        default_world_size=planning_world_size,
        project_root=project_root,
        script_role="planning",
        runtime_note=planning_note,
    )
    smoke_inspect_script = _build_pytorch_inspect_script(
        runtime_config_path=runtime_config_path,
        project_root=project_root,
        script_role="smoke_test",
    )
    planning_inspect_script = _build_pytorch_inspect_script(
        runtime_config_path=planning_config_path,
        project_root=project_root,
        script_role="planning",
    )

    launch_script_path.write_text(smoke_launch_script, encoding="utf-8")
    launch_script_path.chmod(0o755)
    runtime_config_path.write_text(
        json.dumps(smoke_runtime_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    planning_launch_script_path.write_text(planning_launch_script, encoding="utf-8")
    planning_launch_script_path.chmod(0o755)
    planning_config_path.write_text(
        json.dumps(planning_runtime_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    inspect_script_path.write_text(smoke_inspect_script, encoding="utf-8")
    inspect_script_path.chmod(0o755)
    planning_inspect_script_path.write_text(planning_inspect_script, encoding="utf-8")
    planning_inspect_script_path.chmod(0o755)

    # 多机模式：额外生成 hostfile 模板 + 一键 SSH 拉起脚本。
    cluster_artifacts: dict[str, str] = {}
    if normalized.resolved_num_nodes() > 1:
        hostfile_path = output_dir / "hostfile"
        cluster_script_path = output_dir / "launch_cluster.sh"
        hostfile_path.write_text(
            _build_hostfile_template(normalized, planning_world_size),
            encoding="utf-8",
        )
        cluster_script_path.write_text(
            _build_cluster_launcher_script(
                hostfile_name="hostfile",
                launch_script_path=planning_launch_script_path,
                default_gpus_per_node=_default_gpus_per_node(normalized, planning_world_size),
            ),
            encoding="utf-8",
        )
        cluster_script_path.chmod(0o755)
        cluster_artifacts = {
            "hostfile_path": str(hostfile_path),
            "cluster_launcher_path": str(cluster_script_path),
        }

    manual_example = _build_manual_example(normalized, plan)
    declarative_example = _build_declarative_example(normalized)
    manual_example_path.write_text(manual_example, encoding="utf-8")
    declarative_example_path.write_text(declarative_example, encoding="utf-8")

    manual_lines = _count_nonempty_lines(manual_example)
    declarative_lines = _count_nonempty_lines(declarative_example)
    reduction_ratio = 0.0
    if manual_lines > 0:
        reduction_ratio = (manual_lines - declarative_lines) / manual_lines

    notes = [
        "运行时使用 PyTorch 原生 distributed API + DDP + 自定义 TP 模块执行标准 DP+TP。",
        "当前 smoke_test 在卡数足够时直接执行完整层数的真实 HF LLaMA；只有卡数不足时才回退到单进程截断验证。",
        "规划脚本保留完整 world size 和完整层数，用于后续双卡正式执行。",
        f"规划脚本默认 WORLD_SIZE={planning_world_size}，smoke_test 默认 WORLD_SIZE={smoke_world_size}。",
        smoke_note,
        "已额外生成 inspect-only 脚本，可在单卡环境下静态检查 planning/runtime 切分布局。",
        "manual/declarative 两个 py 文件已生成，可直接比较代码量。",
    ]
    if cluster_artifacts:
        notes.append(
            "已生成多机 hostfile 模板与 launch_cluster.sh 一键拉起脚本："
            "请先把 hostfile 中的 localhost 替换成真实节点名，再在主节点执行 bash launch_cluster.sh。"
        )

    return LoweredProgram(
        backend="PYTORCH",
        project_root=project_root,
        output_dir=output_dir,
        launch_script_path=launch_script_path,
        runtime_config_path=runtime_config_path,
        planning_launch_script_path=planning_launch_script_path,
        planning_config_path=planning_config_path,
        manual_example_path=manual_example_path,
        declarative_example_path=declarative_example_path,
        command=["bash", str(launch_script_path)],
        env_overrides={},
        code_reduction_summary={
            "manual_nonempty_lines": manual_lines,
            "declarative_nonempty_lines": declarative_lines,
            "reduction_ratio": reduction_ratio,
        },
        notes=notes,
        extra_artifact_paths={
            "inspect_script_path": str(inspect_script_path),
            "planning_inspect_script_path": str(planning_inspect_script_path),
            **cluster_artifacts,
        },
    )


def lower_to_backend(config: Config, plan: AutoParallelPlan) -> LoweredProgram:
    """根据 backend 选择具体下放路径。"""
    normalized = config.normalized()
    if normalized.backend == "MEGATRON":
        return lower_to_megatron(normalized, plan)
    if normalized.backend == "PYTORCH":
        return lower_to_pytorch(normalized, plan)
    raise ValueError(f"不支持的 backend: {normalized.backend}")
