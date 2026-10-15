"""一键运行自动并行全链路汇报脚本。

这个脚本专门服务于“汇报演示”场景，目标非常明确：
1. 用声明式 `Config + @parallelize` 入口展示自动并行前端。
2. 从搜索开始，一路打印到真实双卡训练结束。
3. 训练日志实时流式输出，便于直接截图或录屏。

默认配置偏向“稳定、快速、可复现”：
1. 默认跑 `MEGATRON` backend，因为它天然会打印每步训练 loss。
2. 默认使用 3B 级 `open_llama_3b_v2` 小模型。
3. 默认关闭 Megatron 的 validation/test，避免汇报时日志过长。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import torch

from autoparallel import Config
from autoparallel import parallelize


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "open_llama_3b_v2"
DEFAULT_WIKITEXT2_PREFIX = (
    PROJECT_ROOT / "datasets" / "wikitext2_megatron" / "wikitext2_text_document_text_document"
)
# Megatron 原生日志：iteration ... | lm loss: 1.23E+01
MEGATRON_NATIVE_STEP_LOSS_PATTERN = re.compile(
    r"iteration\s+(\d+)\s*/\s*(\d+)\s*\|.*?lm loss:\s*([0-9.Ee+-]+)"
)
# 统一 step loss 行：与 PyTorch runtime 输出格式一致
UNIFIED_STEP_LOSS_PATTERN = re.compile(
    r"\[(PYTORCH|MEGATRON)\]\[step\s+(\d+)/(\d+)\]\s+loss=([0-9.Ee+-]+)"
)


def _format_step_loss(backend: str, current_step: int, total_steps: int, loss_value: float) -> str:
    """统一双后端 step loss 打印格式，便于 PPT 截图对比。"""
    return "[%s][step %d/%d] loss=%.6f" % (
        backend.upper(),
        current_step,
        total_steps,
        loss_value,
    )


def _require_minimum_visible_gpus(minimum_gpu_count: int) -> int:
    """检查当前 shell 里是否真的暴露了足够多的 GPU。"""
    visible_gpu_count = torch.cuda.device_count()
    if not torch.cuda.is_available() or visible_gpu_count < minimum_gpu_count:
        raise RuntimeError(
            "当前会话无法执行真实多卡自动并行汇报。"
            f" 期望至少看到 {minimum_gpu_count} 张 GPU，"
            f" 实际只看到 {visible_gpu_count} 张。"
        )
    return visible_gpu_count


def _build_config(args: argparse.Namespace) -> Config:
    """构造汇报脚本使用的声明式自动并行配置。"""
    backend = args.backend.upper()
    use_mock_data = args.data_mode == "mock"
    # PyTorch / Megatron 共用同一份 Megatron indexed wikitext2 前缀。
    data_path = None if use_mock_data else str(DEFAULT_WIKITEXT2_PREFIX)

    num_devices = max(int(args.num_devices), 1)
    return Config(
        model_path=str(Path(args.model_path).expanduser().resolve()),
        num_devices=num_devices,
        search_target_devices=num_devices,
        runtime_test_devices=num_devices,
        backend=backend,
        parallel_mode="AUTO",
        search_strategy=args.search_strategy.upper(),
        search_policy=args.search_policy.upper(),
        sequence_length=args.sequence_length,
        train_iters=args.train_iters,
        log_interval=1,
        # 汇报模式下显式关闭 validation/test，只保留训练 loss。
        eval_interval=0,
        eval_iters=0,
        use_mock_data=use_mock_data,
        data_path=data_path,
        output_dir=str(Path(args.output_dir).expanduser().resolve()),
        run_mode="LAUNCH",
    )


def _print_stage(title: str) -> None:
    """打印简单阶段标题，方便汇报时快速定位。"""
    print(f"\n========== {title} ==========", flush=True)


def _try_extract_megatron_native_step_loss(line: str) -> tuple[int, int, float] | None:
    """从 Megatron 原生日志里提取 step/loss。"""
    matched = MEGATRON_NATIVE_STEP_LOSS_PATTERN.search(line)
    if matched is None:
        return None
    current_step = int(matched.group(1))
    total_steps = int(matched.group(2))
    loss_value = float(matched.group(3))
    return current_step, total_steps, loss_value


def _try_extract_unified_step_loss(line: str) -> tuple[str, int, int, float] | None:
    """从统一格式日志里提取 backend/step/loss。"""
    matched = UNIFIED_STEP_LOSS_PATTERN.search(line)
    if matched is None:
        return None
    backend_name = matched.group(1).upper()
    current_step = int(matched.group(2))
    total_steps = int(matched.group(3))
    loss_value = float(matched.group(4))
    return backend_name, current_step, total_steps, loss_value


def _record_step_loss(
    summary: dict[str, object],
    *,
    current_step: int,
    total_steps: int,
    loss_value: float,
) -> None:
    """把一步 loss 记入汇报摘要。"""
    step_losses = summary["step_losses"]
    assert isinstance(step_losses, list)
    step_losses.append(
        {
            "step": current_step,
            "total_steps": total_steps,
            "loss": loss_value,
        }
    )
    summary["observed_steps"] = current_step
    summary["train_iters"] = total_steps
    summary["final_loss"] = loss_value


def _stream_lowered_program(
    command: list[str],
    project_root: Path,
    env: dict[str, str],
    backend: str,
) -> tuple[int, dict[str, object]]:
    """流式执行下放后的程序，并在需要时抽取结构化摘要。"""
    merged_env = os.environ.copy()
    merged_env.update(env)
    merged_env["PYTHONUNBUFFERED"] = "1"
    backend = backend.upper()
    summary: dict[str, object] = {
        "backend": backend,
        "step_losses": [],
    }

    process = subprocess.Popen(
        command,
        cwd=str(project_root),
        env=merged_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None

    try:
        for line in process.stdout:
            # Megatron 原生日志很长且 loss 是科学计数法；
            # 这里把它改写成与 PyTorch 完全一致的统一格式，并抑制原生 iteration 行。
            if backend == "MEGATRON":
                native = _try_extract_megatron_native_step_loss(line)
                if native is not None:
                    current_step, total_steps, loss_value = native
                    _record_step_loss(
                        summary,
                        current_step=current_step,
                        total_steps=total_steps,
                        loss_value=loss_value,
                    )
                    print(
                        _format_step_loss("MEGATRON", current_step, total_steps, loss_value),
                        flush=True,
                    )
                    continue

            print(line, end="", flush=True)

            unified = _try_extract_unified_step_loss(line)
            if unified is None:
                continue
            _, current_step, total_steps, loss_value = unified
            # PyTorch runtime 已直接打印统一格式；Megatron 统一行若已记录则跳过重复。
            if backend == "MEGATRON":
                continue
            _record_step_loss(
                summary,
                current_step=current_step,
                total_steps=total_steps,
                loss_value=loss_value,
            )
    finally:
        process.stdout.close()

    return process.wait(), summary


def _build_arg_parser() -> argparse.ArgumentParser:
    """构造命令行参数。"""
    parser = argparse.ArgumentParser(description="自动并行全链路汇报脚本")
    parser.add_argument(
        "--backend",
        default="MEGATRON",
        choices=["MEGATRON", "PYTORCH", "megatron", "pytorch"],
        help="选择汇报时使用的 backend，默认使用 MEGATRON。",
    )
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help="本地 Hugging Face 模型目录。",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "generated" / "e2e_report"),
        help="自动并行产物输出目录。",
    )
    parser.add_argument(
        "--train-iters",
        type=int,
        default=3,
        help="训练步数。汇报模式默认取 3，便于展示逐 step loss。",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=16,
        help="汇报模式默认使用较短序列，避免日志等待过久。",
    )
    parser.add_argument(
        "--search-strategy",
        default="BLOCK_LEVEL_FAST",
        help="搜索策略名称。",
    )
    parser.add_argument(
        "--search-policy",
        default="TP_FIRST",
        help="搜索偏好策略。",
    )
    parser.add_argument(
        "--data-mode",
        default="wikitext2",
        choices=["mock", "wikitext2"],
        help=(
            "双后端数据模式。wikitext2 时 PyTorch/Megatron 都使用同一份 "
            "Megatron indexed dataset；mock 时回退到框架内置假数据。"
        ),
    )
    parser.add_argument(
        "--num-devices",
        type=int,
        default=2,
        help="规划与真实训练使用的卡数。当前机器可设为 4。",
    )
    return parser


def main() -> None:
    """命令行入口。"""
    parser = _build_arg_parser()
    args = parser.parse_args()

    _print_stage("环境检测")
    num_devices = max(int(args.num_devices), 1)
    visible_gpu_count = _require_minimum_visible_gpus(minimum_gpu_count=num_devices)
    print(f"检测到 GPU 数量: {visible_gpu_count}", flush=True)
    print(f"目标卡数: {num_devices}", flush=True)
    print(f"模型目录: {Path(args.model_path).expanduser().resolve()}", flush=True)
    print(f"后端: {args.backend.upper()}", flush=True)
    print(f"训练步数: {args.train_iters}", flush=True)
    print(f"数据模式: {args.data_mode}", flush=True)

    config = _build_config(args)

    @parallelize(config)
    def train() -> None:
        """声明式训练入口。

        这里故意不手写 TP/DP 通信细节。
        真正的并行切分、启动脚本生成和分布式执行都由框架自动完成。
        """

    runtime = train.autoparallel_runtime

    _print_stage("搜索结果")
    plan = runtime.compile()
    selected_plan = plan.parallel_plan
    print(
        "selected_plan: tp=%d, dp=%d, attention=%s, mlp=%s, embedding=%s, lm_head=%s"
        % (
            selected_plan.tp,
            selected_plan.dp,
            selected_plan.attention_policy,
            selected_plan.mlp_policy,
            selected_plan.embedding_policy,
            selected_plan.lm_head_policy,
        ),
        flush=True,
    )
    print(
        "search: strategy=%s, policy=%s, time=%.4fs"
        % (
            plan.search_summary["selected_strategy"],
            plan.search_summary["selected_policy"],
            plan.search_summary["selected_search_time_seconds"],
        ),
        flush=True,
    )
    print(
        "execution_mode=%s, world_size=%d"
        % (selected_plan.execution_mode, selected_plan.world_size),
        flush=True,
    )

    _print_stage("下放产物")
    lowered = runtime.lower()
    print(f"launch_script={lowered.launch_script_path}", flush=True)
    print(f"runtime_config={lowered.runtime_config_path}", flush=True)
    print(f"manual_example={lowered.manual_example_path}", flush=True)
    print(f"declarative_example={lowered.declarative_example_path}", flush=True)

    _print_stage("真实训练日志")
    return_code, runtime_summary = _stream_lowered_program(
        command=lowered.command,
        project_root=lowered.project_root,
        env=lowered.env_overrides,
        backend=lowered.backend,
    )
    if return_code != 0:
        raise SystemExit(return_code)

    _print_stage("汇报完成")
    report_summary = {
        "backend": lowered.backend,
        "world_size": selected_plan.world_size,
        "tp_size": selected_plan.tp,
        "dp_size": selected_plan.dp,
        "train_iters": runtime_summary.get("train_iters", args.train_iters),
        "observed_steps": runtime_summary.get("observed_steps", 0),
        "final_loss": runtime_summary.get("final_loss"),
        "step_losses": runtime_summary.get("step_losses", []),
        "data_mode": args.data_mode,
        "artifacts": str(lowered.output_dir),
    }
    print(json.dumps(report_summary, ensure_ascii=False, indent=2), flush=True)
    print("全链路状态: SUCCESS", flush=True)
    print(f"backend={lowered.backend}", flush=True)
    print(f"artifacts={lowered.output_dir}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n汇报脚本被用户中断。", file=sys.stderr, flush=True)
        raise
