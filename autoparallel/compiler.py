"""自动并行编译后端。"""

from __future__ import annotations

from autoparallel.config import Config
from autoparallel.config import coerce_config
from autoparallel.types import AutoParallelPlan
from megatron_mvp.detect import detect_hf_model
from megatron_mvp.detect import load_model_spec
from megatron_mvp.environment import detect_runtime_environment
from megatron_mvp.render import build_coarse_ir
from megatron_mvp.render import build_launch_script
from megatron_mvp.render import build_megatron_config
from megatron_mvp.search import search_parallel_candidates
from megatron_mvp.search import validate_standard_parallel_plan
from megatron_mvp.types import DeviceInfo
from megatron_mvp.types import ParallelPlan
from megatron_mvp.types import RuntimeEnvironment
from megatron_mvp.types import SearchResult


def _build_virtual_environment(
    detected_env: RuntimeEnvironment,
    target_device_count: int,
    memory_target_gb: float | None,
) -> RuntimeEnvironment:
    """按照目标卡数构造虚拟运行环境。"""
    resolved_device_count = max(target_device_count, 1)
    target_memory_bytes = detected_env.primary_device_memory_bytes
    if memory_target_gb is not None:
        target_memory_bytes = int(memory_target_gb * 1024**3)

    template = detected_env.devices[0]
    virtual_devices: list[DeviceInfo] = []
    for index in range(resolved_device_count):
        virtual_devices.append(
            DeviceInfo(
                index=index,
                name=template.name,
                total_memory_bytes=target_memory_bytes,
                multi_processor_count=template.multi_processor_count,
            )
        )

    return RuntimeEnvironment(
        accelerator_type=detected_env.accelerator_type,
        device_count=resolved_device_count,
        devices=virtual_devices,
        primary_device_name=template.name,
        primary_device_memory_bytes=target_memory_bytes,
        torch_version=detected_env.torch_version,
        hip_version=detected_env.hip_version,
    )


def _resolve_execution_mode(plan: ParallelPlan, env: RuntimeEnvironment) -> str:
    """根据目标方案和规划环境判断执行模式。"""
    if plan.world_size <= env.device_count:
        return "current_feasible"
    return "future_scalable"


def _validate_lowerable_parallel_plan(spec, plan: ParallelPlan) -> list[str]:
    """校验显式 ParallelPlan 是否属于当前标准 DP+TP 公共策略域。"""
    return validate_standard_parallel_plan(spec, plan)


def _build_explicit_search_result(
    explicit_plan: ParallelPlan,
    planning_env: RuntimeEnvironment,
) -> SearchResult:
    """把用户显式指定的 ParallelPlan 包装成统一的 SearchResult。"""
    normalized_plan = explicit_plan.normalized()
    execution_mode = _resolve_execution_mode(normalized_plan, planning_env)
    finalized_plan = normalized_plan.with_updates(
        execution_mode=execution_mode,
        strategy_name="EXPLICIT_PLAN",
        strategy_policy="USER_DEFINED",
        search_time_seconds=0.0,
        fits_current_environment=execution_mode == "current_feasible",
        requires_additional_devices=max(0, normalized_plan.world_size - planning_env.device_count),
        notes=(
            "该方案由用户显式指定，因此本轮编译直接跳过自动搜索。",
        ),
    )
    feasible_current_plans: list[ParallelPlan] = []
    scalable_future_plans: list[ParallelPlan] = []
    if execution_mode == "current_feasible":
        feasible_current_plans.append(finalized_plan)
    else:
        scalable_future_plans.append(finalized_plan)

    return SearchResult(
        strategy_name="EXPLICIT_PLAN",
        strategy_policy="USER_DEFINED",
        planning_device_count=planning_env.device_count,
        search_time_seconds=0.0,
        execution_mode=execution_mode,
        selected_plan=finalized_plan,
        selected_candidate=None,
        feasible_current_candidates=[],
        scalable_future_candidates=[],
        feasible_current_plans=feasible_current_plans,
        scalable_future_plans=scalable_future_plans,
        benchmarks=[],
        rejected_reasons=["显式 ParallelPlan 已生效，自动搜索阶段被跳过。"],
    )


def compile_plan(config: Config | ParallelPlan) -> AutoParallelPlan:
    """把声明式配置编译成自动并行执行计划。"""
    normalized = coerce_config(config).normalized()
    model_root = normalized.resolved_model_path()
    output_dir = normalized.resolved_output_dir()

    artifacts = detect_hf_model(model_root)
    spec = load_model_spec(artifacts)
    detected_env = detect_runtime_environment()

    planning_device_count = normalized.resolved_search_target_devices(
        detected_env.device_count
    )
    runtime_test_device_count = normalized.resolved_runtime_test_devices()
    planning_env = _build_virtual_environment(
        detected_env=detected_env,
        target_device_count=planning_device_count,
        memory_target_gb=normalized.memory_target_gb,
    )
    runtime_test_env = _build_virtual_environment(
        detected_env=detected_env,
        target_device_count=min(runtime_test_device_count, detected_env.device_count),
        memory_target_gb=normalized.memory_target_gb,
    )

    effective_search_policy = normalized.search_policy

    # PYTORCH backend 的自研运行时只实现了 DP+TP（无 PP/EP），
    # 这里提前把 PP 搜索上限压回 1，避免生成无法执行的方案。
    effective_max_pp = normalized.search_max_pp
    if normalized.backend == "PYTORCH" and effective_max_pp > 1:
        effective_max_pp = 1

    if normalized.explicit_plan is not None:
        if (
            normalized.backend == "PYTORCH"
            and (normalized.explicit_plan.pp > 1 or normalized.explicit_plan.ep > 1)
        ):
            raise ValueError(
                "PYTORCH backend 只支持 DP+TP（pp=1, ep=1）。"
                "如需 PP/EP，请使用 MEGATRON backend。"
            )
        search_result = _build_explicit_search_result(
            explicit_plan=normalized.explicit_plan,
            planning_env=planning_env,
        )
    else:
        search_result = search_parallel_candidates(
            spec=spec,
            env=planning_env,
            seq_length=normalized.sequence_length,
            max_world_size=normalized.search_max_world_size,
            strategy=normalized.search_strategy,
            policy=effective_search_policy,
            max_pp=effective_max_pp,
            max_ep=normalized.search_max_ep,
        )

    selected_plan = search_result.selected_plan
    if selected_plan is None:
        raise RuntimeError("粗搜索未找到任何可行 ParallelPlan，无法生成后续 IR 和模板。")

    if normalized.backend == "PYTORCH" and (selected_plan.pp > 1 or selected_plan.ep > 1):
        raise ValueError(
            "PYTORCH backend 只支持 DP+TP（pp=1, ep=1），"
            f"当前选中方案 pp={selected_plan.pp}, ep={selected_plan.ep}。"
            "请设置 search_max_pp=1 或改用 MEGATRON backend。"
        )

    selected_parallel_mode = normalized.parallel_mode
    if selected_parallel_mode == "AUTO":
        selected_parallel_mode = selected_plan.parallel_mode
    finalized_plan = selected_plan.with_updates(parallel_mode=selected_parallel_mode)
    lowerability_reasons = _validate_lowerable_parallel_plan(spec, finalized_plan)
    if lowerability_reasons:
        raise ValueError(
            "当前 ParallelPlan 不属于标准 DP+TP 公共策略域: "
            + "；".join(lowerability_reasons)
        )

    ir = build_coarse_ir(
        spec=spec,
        env=planning_env,
        plan=finalized_plan,
        search_result=search_result,
    )
    ir_dict = ir.to_dict()
    ir_dict["parallel_mode"] = selected_parallel_mode
    ir_dict["user_prefer"] = normalized.prefer
    ir_dict["requested_search_strategy"] = normalized.search_strategy
    ir_dict["effective_search_policy"] = effective_search_policy
    ir_dict["runtime_test_devices"] = runtime_test_device_count
    ir_dict["num_nodes"] = normalized.resolved_num_nodes()

    megatron_config = build_megatron_config(ir)
    launch_script = build_launch_script(
        ir,
        megatron_config,
        num_nodes=normalized.resolved_num_nodes(),
        gpus_per_node=normalized.resolved_gpus_per_node(),
        master_addr=normalized.master_addr,
        master_port=normalized.master_port,
    )

    generated_files: dict[str, str] = {}
    if normalized.generate_artifacts:
        from megatron_mvp.render import write_generation_outputs

        outputs = write_generation_outputs(
            artifacts=artifacts,
            spec=spec,
            env=planning_env,
            search_result=search_result,
            ir=ir,
            output_dir=output_dir,
            num_nodes=normalized.resolved_num_nodes(),
            gpus_per_node=normalized.resolved_gpus_per_node(),
            master_addr=normalized.master_addr,
            master_port=normalized.master_port,
        )
        generated_files = outputs.to_dict()

    search_summary = {
        "selected_strategy": search_result.strategy_name,
        "selected_policy": search_result.strategy_policy,
        "selected_search_time_seconds": search_result.search_time_seconds,
        "planning_device_count": search_result.planning_device_count,
        "selected_parallel_plan": finalized_plan.to_dict(),
        "benchmarks": [benchmark.to_dict() for benchmark in search_result.benchmarks],
    }

    notes = list(ir.search_notes)
    notes.append(
        "规划阶段按 %d 张卡搜索，当前 smoke test 按 %d 张卡执行。"
        % (planning_device_count, runtime_test_device_count)
    )
    if normalized.resolved_num_nodes() > 1:
        notes.append(
            "多机规划：%d 节点（每节点默认 %s 卡）；"
            "生成的启动脚本已按 NNODES/NODE_RANK/MASTER_ADDR 多机语义下放，"
            "可用 tools/launch_cluster.py 按 hostfile 一键拉起。"
            % (
                normalized.resolved_num_nodes(),
                normalized.resolved_gpus_per_node() or "world_size/num_nodes",
            )
        )
    notes.append(
        "最终采用 %s/%s，搜索耗时 %.4f 秒。"
        % (
            search_result.strategy_name,
            search_result.strategy_policy,
            search_result.search_time_seconds,
        )
    )
    if search_result.strategy_name == "BLOCK_LEVEL_FAST":
        notes.append(
            "BLOCK_LEVEL_FAST 当前直接搜索显式 ParallelPlan，并只下钻少量目标 world size。"
        )
    elif search_result.strategy_name == "HIERARCHICAL":
        notes.append(
            "HIERARCHICAL 当前使用两阶段 plan 搜索，先做 plan 预筛，再展开完整枚举。"
        )
    elif search_result.strategy_name == "RL_GUIDED_PPO":
        notes.append(
            "RL_GUIDED_PPO 当前在显式 plan 动作空间上做在线 rollout，并使用 clipped PPO 更新控制器。"
        )
    else:
        notes.append("当前使用用户显式指定的 ParallelPlan，未触发自动搜索。")
    notes.append("前端支持 Config + @parallelize(config)，也支持直接传入 ParallelPlan。")

    return AutoParallelPlan(
        model_path=model_root,
        output_dir=output_dir,
        parallel_plan=finalized_plan,
        model_summary={
            "model_type": spec.model_type,
            "architecture": spec.architecture,
            "hidden_size": spec.hidden_size,
            "intermediate_size": spec.intermediate_size,
            "num_layers": spec.num_layers,
            "num_heads": spec.num_heads,
            "num_kv_heads": spec.num_kv_heads,
            "estimated_total_params": spec.estimated_total_params,
        },
        runtime_summary={
            "detected_device_count": detected_env.device_count,
            "planning_device_count": planning_env.device_count,
            "runtime_test_device_count": runtime_test_env.device_count,
            "detected_device_memory_bytes": detected_env.primary_device_memory_bytes,
            "planning_device_memory_bytes": planning_env.primary_device_memory_bytes,
            "runtime_test_device_memory_bytes": runtime_test_env.primary_device_memory_bytes,
            "device_name": planning_env.primary_device_name,
            "accelerator_type": planning_env.accelerator_type,
        },
        search_summary=search_summary,
        coarse_ir=ir_dict,
        megatron_config=megatron_config,
        launch_script=launch_script,
        generated_files=generated_files,
        notes=notes,
    )
