"""多机多卡自动并行入口（双机 16 卡验证 -> 128 卡扩展）。

用法示例
--------

1. 双机 16 卡规划 + 生成多机启动脚本（不真正训练）::

    python3 examples/autoparallel_multinode_train.py \
        --num-nodes 2 --gpus-per-node 8 \
        --model-path models/glm45_air_105b_spec \
        --search-max-pp 2 --search-max-ep 2 \
        --memory-target-gb 64 \
        --run-mode LOWER

2. 128 卡（16 节点 × 8 卡）规划::

    python3 examples/autoparallel_multinode_train.py \
        --num-nodes 16 --gpus-per-node 8 \
        --model-path models/glm45_air_105b_spec \
        --search-max-pp 8 --search-max-ep 8 \
        --memory-target-gb 64 \
        --run-mode LOWER

3. 显式指定并行方案（跳过搜索）::

    python3 examples/autoparallel_multinode_train.py \
        --num-nodes 2 --gpus-per-node 8 \
        --model-path models/glm45_air_105b_spec \
        --explicit tp=2,dp=4,pp=2,ep=2 \
        --run-mode LOWER

4. 在真实集群上执行（先生成，再用 launch_cluster.py 拉起）::

    # 生成产物后，把 hostfile 里的 localhost 换成真实节点名，然后：
    python3 tools/launch_cluster.py \
        --hostfile generated/autoparallel/hostfile \
        --script generated/autoparallel/launch_megatron_plan.sh \
        --gpus-per-node 8

注意
----
- PLAN/LOWER 模式只需要当前机器能 import torch（用于环境探测），单卡开发机即可运行。
- LAUNCH 模式会在当前机器直接执行 smoke 脚本，多机验证请用第 4 步的 launch_cluster.py。
- PYTORCH backend 只支持 DP+TP（自动把 PP/EP 压回 1）；多机 PP/EP 请用 MEGATRON backend。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from autoparallel import Config
from autoparallel import ParallelPlan
from autoparallel import parallelize

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "glm45_air_105b_spec"


def parse_explicit_plan(raw: str) -> ParallelPlan:
    """解析 `tp=2,dp=4,pp=2,ep=2` 形式的显式方案。"""
    fields: dict[str, str] = {}
    for item in raw.split(","):
        key, _, value = item.partition("=")
        key = key.strip().lower()
        if not key or not value:
            raise ValueError(f"显式方案格式错误: {raw}（期望 tp=2,dp=4,pp=2,ep=2）")
        fields[key] = value.strip()

    tp = int(fields.get("tp", 1))
    dp = int(fields.get("dp", 1))
    pp = int(fields.get("pp", 1))
    ep = int(fields.get("ep", 1))

    if tp > 1:
        attention_policy = "head_shard"
        mlp_policy = "colrow_shard"
        embedding_policy = "vocab_shard"
        lm_head_policy = "vocab_shard"
        sequence_parallel = True
    else:
        attention_policy = "replicate"
        mlp_policy = "replicate"
        embedding_policy = "replicate"
        lm_head_policy = "replicate"
        sequence_parallel = False

    return ParallelPlan(
        tp=tp,
        dp=dp,
        pp=pp,
        ep=ep,
        attention_policy=attention_policy,
        mlp_policy=mlp_policy,
        embedding_policy=embedding_policy,
        lm_head_policy=lm_head_policy,
        sequence_parallel=sequence_parallel,
        micro_batch_size=1,
        global_batch_size=max(dp, 1),
    )


def build_config(args: argparse.Namespace) -> Config:
    """构造多机声明式配置。"""
    world_size = max(args.num_nodes, 1) * max(args.gpus_per_node, 1)
    explicit_plan = parse_explicit_plan(args.explicit) if args.explicit else None

    return Config(
        model_path=str(Path(args.model_path).expanduser().resolve()),
        # 规划视角：按目标集群总卡数规划（虚拟环境），不要求本机真有这些卡。
        search_target_devices=world_size,
        search_max_world_size=max(world_size, 128),
        search_max_pp=args.search_max_pp,
        search_max_ep=args.search_max_ep,
        # 集群拓扑：决定生成脚本里 GPUS_PER_NODE / NNODES 的默认值。
        num_nodes=args.num_nodes,
        gpus_per_node=args.gpus_per_node,
        master_addr=args.master_addr,
        master_port=args.master_port,
        # BW1000 单卡 64GB HBM2e。
        memory_target_gb=args.memory_target_gb,
        prefer="memory",
        backend=args.backend.upper(),
        run_mode=args.run_mode.upper(),
        sequence_length=args.sequence_length,
        train_iters=args.train_iters,
        use_mock_data=True,
        explicit_plan=explicit_plan,
        output_dir=str(Path(args.output_dir).expanduser().resolve()),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """构造命令行参数。"""
    parser = argparse.ArgumentParser(description="多机多卡自动并行入口")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--backend", default="MEGATRON", choices=["MEGATRON", "PYTORCH"])
    parser.add_argument("--run-mode", default="LOWER", choices=["PLAN", "LOWER", "LAUNCH"])
    parser.add_argument("--num-nodes", type=int, default=2, help="节点数（默认 2，双机验证）")
    parser.add_argument("--gpus-per-node", type=int, default=8, help="每节点卡数（默认 8）")
    parser.add_argument("--search-max-pp", type=int, default=1, help="PP 搜索上限（多机建议 2~16）")
    parser.add_argument("--search-max-ep", type=int, default=1, help="MoE EP 搜索上限")
    parser.add_argument("--memory-target-gb", type=float, default=64.0, help="单卡显存 GB（BW1000=64）")
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--train-iters", type=int, default=2)
    parser.add_argument("--master-addr", default=None, help="多机主节点地址（写入脚本默认值）")
    parser.add_argument("--master-port", type=int, default=None)
    parser.add_argument(
        "--explicit",
        default=None,
        help="显式并行方案，例如 tp=2,dp=4,pp=2,ep=2（提供时跳过自动搜索）",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "generated" / "autoparallel"),
    )
    return parser


def main() -> None:
    """命令行入口。"""
    args = build_arg_parser().parse_args()
    config = build_config(args)

    world_size = max(args.num_nodes, 1) * max(args.gpus_per_node, 1)
    print("=" * 60)
    print(f"多机自动并行: {args.num_nodes} 节点 × {args.gpus_per_node} 卡 = {world_size} 卡")
    print(f"模型: {config.model_path}")
    print(f"backend={config.backend} run_mode={config.run_mode}")
    print(f"搜索上限: world_size<={config.search_max_world_size}, "
          f"pp<={config.search_max_pp}, ep<={config.search_max_ep}")
    print(f"本机 GPU 数: {torch.cuda.device_count() if torch.cuda.is_available() else 0}（规划模式不要求匹配）")
    print("=" * 60)

    @parallelize(config)
    def train() -> None:
        """声明式训练入口：只描述意图，不手写切分/通信/多机编排。"""

    runtime = train.autoparallel_runtime
    plan = runtime.compile()
    selected = plan.parallel_plan

    print("-" * 60)
    print(
        "搜索结果: TP=%d, DP=%d, PP=%d, EP=%d, world_size=%d, "
        "mbs=%d, gbs=%d, seq_parallel=%s"
        % (
            selected.tp,
            selected.dp,
            selected.pp,
            selected.ep,
            selected.world_size,
            selected.micro_batch_size,
            selected.global_batch_size,
            selected.sequence_parallel,
        )
    )
    if selected.estimated_memory_bytes is not None:
        print(
            "预估单卡显存: %.2f GB / 上限 %.0f GB"
            % (selected.estimated_memory_bytes / 1024**3, config.memory_target_gb or 0)
        )
    print(f"策略: {plan.search_summary['selected_strategy']} / "
          f"{plan.search_summary['selected_policy']}，"
          f"耗时 {plan.search_summary['selected_search_time_seconds']:.4f}s")

    if config.run_mode == "PLAN":
        print("\n[PLAN 模式] 到此为止。生成产物见:", plan.output_dir)
        return

    lowered = runtime.lower()
    print("-" * 60)
    print("下放产物:")
    print(f"  smoke 脚本   : {lowered.launch_script_path}")
    print(f"  规划脚本     : {lowered.planning_launch_script_path}")
    if args.num_nodes > 1:
        hostfile = Path(lowered.output_dir) / "hostfile"
        cluster = Path(lowered.output_dir) / "launch_cluster.sh"
        print(f"  hostfile     : {hostfile}")
        print(f"  一键拉起脚本 : {cluster}")
        print(
            "\n多机执行步骤:\n"
            "  1. 编辑 hostfile，把 localhost 替换成真实节点名（第一行是主节点）。\n"
            "  2. 确认所有节点在相同路径部署了本仓库且 SSH 免密。\n"
            f"  3. 在主节点执行: python3 {PROJECT_ROOT}/tools/launch_cluster.py "
            f"--hostfile {hostfile} --script {lowered.planning_launch_script_path} "
            f"--gpus-per-node {args.gpus_per_node}"
        )
    for note in lowered.notes:
        print(f"  note: {note}")

    if config.run_mode == "LOWER":
        return

    # LAUNCH：在当前机器执行 smoke 脚本（真实多机请走 launch_cluster.py）。
    from autoparallel import launch_lowered_program

    print("-" * 60)
    print("[LAUNCH 模式] 在本机执行 smoke 脚本...")
    return_code = launch_lowered_program(lowered)
    raise SystemExit(return_code)


if __name__ == "__main__":
    main()
