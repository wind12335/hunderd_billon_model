"""多厂商档位回归驱动（multi-vendor 分支）。

四部分：
1. hygon   —— 海光基线不回归变：跑 tools/regress_ep.py 子进程 +
              glm64_g2 全链路重渲染，megatron_config 与已提交版本逐键相等
              （仅剔除计时字段 search_time_seconds）；
2. vendors —— 三厂商档（AUTOPLAN_VENDOR=maca/musa/corex）：
              替身探测断言、GLM MoE 百亿线 EP 必须被拒（ep==1 且留拒绝理由）、
              稠密线渲染断言（activate/wrapper 路径、--transformer-impl local、
              musa/corex 无 --sequence-parallel）；
3. frontend—— 声明式前端 LOWER 冒烟（AUTOPLAN_VENDOR=musa + glm45_air_smoke），
              产物落盘断言（厂商脚本、torch_musa 后端注册、musa:0 设备）；
4. unit    —— vendor_profile 纯函数断言（回落、覆盖、非法值报错）。

用法：
    python3 tools/regress_vendor.py             # 全部
    python3 tools/regress_vendor.py hygon vendors frontend unit   # 任选子集
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from megatron_mvp import detect, render, search  # noqa: E402
from megatron_mvp.environment import detect_runtime_environment as detect_environment  # noqa: E402
from megatron_mvp.types import ModelSpec  # noqa: E402
from megatron_mvp.vendor_profile import PROFILES, current_profile, pick_profile  # noqa: E402

_SECTIONS = ("hygon", "vendors", "frontend", "unit")
_TIMING_KEYS = {"search_time_seconds"}


def _scrub(obj):
    """递归剔除计时类字段（重跑必然变化，与正确性无关）。"""
    if isinstance(obj, dict):
        return {k: _scrub(v) for k, v in obj.items() if k not in _TIMING_KEYS}
    if isinstance(obj, list):
        return [_scrub(v) for v in obj]
    return obj


def _with_vendor(key: str | None):
    """设置/清除厂商覆盖环境变量。"""
    if key:
        os.environ["AUTOPLAN_VENDOR"] = key
    else:
        os.environ.pop("AUTOPLAN_VENDOR", None)


def glm_base_spec() -> ModelSpec:
    """GLM-4.5-Air 105B（MoE）真权重 spec——百亿/千亿共用的检测入口。"""
    return detect.load_model_spec(detect.detect_hf_model(REPO / "models"))


def llama_dense_spec() -> ModelSpec:
    """稠密 Llama-3.1-8B spec：权重不在本机，从已提交 detection_summary 重建。"""
    d = json.loads(
        (REPO / "generated/dual16_llama8b/detection_summary.json").read_text()
    )["model_spec"]
    d["model_dir"] = Path(d["model_dir"])
    return ModelSpec(**d)


def _search(spec, devices, world, pp, ep, seq=2048):
    os.environ["AUTOPLAN_STANDIN_GPUS"] = str(devices)
    env = detect_environment()
    return env, search.search_parallel_candidates(
        spec=spec, env=env, seq_length=seq, max_world_size=world,
        strategy="BLOCK_LEVEL_FAST", policy="BALANCED",
        max_pp=pp, max_ep=ep, benchmark_all_builtin=False,
    )


# ---------------------------------------------------------------- hygon 基线
def check_hygon_parity() -> None:
    print("== [1/4] 海光基线等价 ==")
    sub_env = {k: v for k, v in os.environ.items() if k != "AUTOPLAN_VENDOR"}
    proc = subprocess.run(
        [sys.executable, str(REPO / "tools/regress_ep.py")],
        env=sub_env, capture_output=True, text=True,
    )
    tail = "\n".join(proc.stdout.strip().splitlines()[-4:])
    print(f"[regress_ep 子进程] rc={proc.returncode}\n{tail}")
    assert proc.returncode == 0, f"regress_ep.py 失败:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"

    # glm64_g2 全链路重渲染：搜索→IR→megatron_config，与已提交产物逐键比对。
    _with_vendor(None)
    spec = glm_base_spec()
    env, res = _search(spec, devices=64, world=128, pp=2, ep=8, seq=4096)
    plan = res.selected_plan
    assert (plan.tp, plan.dp, plan.pp, plan.ep) == (8, 8, 1, 8), f"glm64 选型漂移: {plan.mesh}"
    ir = render.build_coarse_ir(spec, env, plan, res)
    config = render.build_megatron_config(ir)
    committed = json.loads((REPO / "generated/glm64_g2/megatron_config.json").read_text())
    # notes 是人读日志行：含搜索耗时（每次必变）与每步墙钟估算（10 月生成
    # 后 Phase6 常量回填引起的既有漂移，与本合并无关——已在未打补丁的 main
    # 上复现同样数值）。结构键必须逐键相等；notes 剔除这两类行后逐句相等。
    def _drop_volatile_notes(cfg: dict) -> dict:
        out = _scrub(cfg)
        out["notes"] = [
            n for n in cfg["notes"] if ("耗时" not in n and "墙钟估算" not in n)
        ]
        return out

    assert _drop_volatile_notes(config) == _drop_volatile_notes(committed), (
        "glm64_g2 megatron_config 与已提交版本不一致（海光渲染回归失败）"
    )
    print("[assert] glm64_g2 megatron_config 结构键逐键相等；notes 除计时/墙钟行外逐句相等")

    # coarse_ir：新版本仅在 environment 多了 vendor_profile=="hygon" 一个键。
    # 下列键为已提交 glm64_g2（10 月生成）与当前 main 之间【既有】漂移，
    # 已在未打补丁的工作树上复现同样数值，与本合并无关：
    #   - communication.flops_per_device / estimated_step_seconds：
    #     Phase6 实测回填 250→300 TFLOPS 及其派生墙钟秒；
    #   - search.benchmarks：benchmark 列表 10 月主干演化（1→2 条）。
    new_ir = _scrub(ir.to_dict())
    old_ir = _scrub(json.loads((REPO / "generated/glm64_g2/coarse_ir.json").read_text()))
    new_ir["environment"].pop("vendor_profile")
    for doc in (new_ir, old_ir):
        doc["communication"].pop("flops_per_device", None)
        doc["communication"].pop("estimated_step_seconds", None)
        doc["search"].pop("benchmarks", None)
        doc["search_notes"] = [
            n for n in doc["search_notes"] if ("耗时" not in n and "墙钟估算" not in n)
        ]
    assert new_ir == old_ir, "glm64_g2 coarse_ir 除 vendor_profile 与既有漂移外仍有差异"
    print("[assert] glm64_g2 coarse_ir 除新增 vendor_profile 与既有漂移键外逐键相等")


# ---------------------------------------------------------------- 三厂商档
def check_vendor_profiles() -> None:
    print("== [2/4] 三厂商档（maca/musa/corex）==")
    spec_smoke = detect.load_model_spec(
        detect.detect_hf_model(REPO / "models/glm45_air_smoke")
    )
    assert spec_smoke.is_moe, "smoke 切片应为 MoE"
    spec_dense = llama_dense_spec()

    # 海光对照：同一 MoE spec 海光档会选 EP>1，下面三厂商档必须 EP=1——
    # 这样 ep==1 才能归因于档位门控，而非模型/拓扑本身不适用 EP。
    _with_vendor(None)
    _, res_h = _search(spec_smoke, devices=8, world=8, pp=2, ep=8)
    plan_h = res_h.selected_plan
    assert plan_h is not None and plan_h.ep > 1, f"海光对照未选 EP>1: {plan_h}"
    print(f"[hygon 对照] smoke MoE 选 TP{plan_h.tp}/DP{plan_h.dp}/PP{plan_h.pp}/EP{plan_h.ep}")

    for key in ("maca", "musa", "corex"):
        profile = PROFILES[key]
        _with_vendor(key)

        # (a) 替身探测：厂商档生效，替身卡规格随档切换。
        os.environ["AUTOPLAN_STANDIN_GPUS"] = "8"
        env = detect_environment()
        assert env.accelerator_type == profile.accelerator_types[0], (
            f"{key}: accelerator_type={env.accelerator_type}"
        )
        assert env.primary_device_name == profile.standin_device_name, key
        assert env.primary_device_memory_bytes == profile.standin_memory_bytes, key
        assert env.device_count == 8, key

        # (b) EP 拒绝（百亿/千亿分账）：MoE spec 开 max_ep=8 也必须选 ep=1，
        #     且拒绝理由留痕（千亿 EP/MoE 仅海光档）。corex 32G 替身在无 EP
        #     的 MoE 上可能整体无可行方案（真实行为：天数 32G 卡放不下无 EP
        #     的 MoE 权重），此时只断言 EP 拒绝留痕。
        env, res = _search(spec_smoke, devices=8, world=8, pp=2, ep=8)
        plan = res.selected_plan
        if plan is None:
            assert key == "corex", f"{key}: 意外整体不可行"
            print(f"[{key}] MoE 无 EP 下无可行方案（32G 替身真实行为）——查拒绝留痕")
        else:
            assert plan.ep == 1, f"{key}: 选出 EP={plan.ep}，应被档位拒绝"
            assert plan.megatron_lowerable, f"{key}: 稠密路径应保持可下放"
        ep_refused = any("专家并行" in r for r in res.rejected_reasons)
        assert ep_refused, f"{key}: EP 拒绝理由未留痕"

        # (c) 稠密线渲染：厂商脚本路径 + transformer-impl local + SP 开关。
        env, res = _search(spec_dense, devices=16, world=128, pp=2, ep=1)
        dplan = res.selected_plan
        ir = render.build_coarse_ir(spec_dense, env, dplan, res)
        assert ir.environment["vendor_profile"] == key, f"{key}: IR 未记录厂商档"
        config = render.build_megatron_config(ir)
        assert config["model_args"].get("transformer_impl") == "local", (
            f"{key}: transformer_impl={config['model_args'].get('transformer_impl')}"
        )
        sp = config["parallelism"].get("sequence_parallel")
        if not profile.allow_sequence_parallel:
            assert sp is False, f"{key}: 无厂商 TE 却渲染 sequence_parallel={sp}"
        script = render.build_launch_script(ir, config)
        assert profile.activate_script in script, f"{key}: 启动脚本缺 {profile.activate_script}"
        assert profile.run_wrapper in script, f"{key}: 启动脚本缺 {profile.run_wrapper}"
        assert "--transformer-impl local" in script, f"{key}: 启动脚本缺 --transformer-impl local"
        if not profile.allow_sequence_parallel:
            assert "--sequence-parallel" not in script, f"{key}: 启动脚本不应含 --sequence-parallel"
        # activate 脚本与 wrapper 真实在位（渲染引用的路径必须存在）。
        assert (REPO / profile.activate_script).is_file(), f"{key}: {profile.activate_script} 缺失"
        assert (REPO / profile.run_wrapper).is_file(), f"{key}: {profile.run_wrapper} 缺失"
        print(
            f"[{key}] 替身OK | MoE EP=1(拒EP留痕) | 稠密 TP{dplan.tp}/DP{dplan.dp}/"
            f"PP{dplan.pp} SP={sp} | 脚本断言OK"
        )
    _with_vendor(None)


# ---------------------------------------------------------------- 前端 LOWER
def check_musa_frontend_lower() -> None:
    print("== [3/4] 声明式前端 LOWER 冒烟（musa 档 + glm45_air_smoke）==")
    from autoparallel import Config, parallelize

    out_dir = Path("/tmp/autoplan_vendor_regress/musa_frontend")
    if out_dir.exists():
        subprocess.run(["rm", "-rf", str(out_dir)], check=True)
    _with_vendor("musa")

    config = Config(
        model_path=str(REPO / "models/glm45_air_smoke"),
        num_devices=8,
        search_target_devices=8,
        backend="MEGATRON",
        run_mode="LOWER",
        sequence_length=2048,
        search_max_pp=2,
        search_max_ep=8,  # 前端也开着 EP 上限，验证 musa 档在声明式链路同样拒 EP
        use_mock_data=True,
        output_dir=str(out_dir),
    )

    @parallelize(config)
    def train() -> None:
        """声明式训练入口（仅规划下放，不执行）。"""

    runtime = train.autoparallel_runtime
    plan = runtime.compile()
    sel = plan.parallel_plan
    assert sel.ep == 1, f"前端 musa 档选出 EP={sel.ep}"
    runtime.lower()

    smoke = (out_dir / "launch_megatron_smoke.sh").read_text()
    assert "vendors/musa/scripts/activate_megatron_musa.sh" in smoke
    assert "vendors/musa/tools/run_megatron_musa.py" in smoke
    assert "--transformer-impl local" in smoke
    assert "--sequence-parallel" not in smoke
    manual = (out_dir / "examples/manual_megatron_train.py").read_text()
    assert "import torch_musa" in manual, "PyTorch 后端示例未注册 MUSA 后端"
    assert '"musa:0"' in manual, "PyTorch 后端示例设备名未切到 musa:0"
    print(
        f"[musa 前端] TP{sel.tp}/DP{sel.dp}/PP{sel.pp}/EP{sel.ep} → {out_dir} "
        f"产物断言全过（厂商脚本/local/无SP/torch_musa/musa:0）"
    )
    _with_vendor(None)


# ---------------------------------------------------------------- 纯函数断言
def check_vendor_unit() -> None:
    print("== [4/4] vendor_profile 纯函数 ==")
    _with_vendor(None)
    assert pick_profile("hip") is PROFILES["hygon"]
    assert pick_profile("cuda") is PROFILES["hygon"], "通用 cuda 应回落海光（合并前行为）"
    assert pick_profile(None) is PROFILES["hygon"]
    assert pick_profile("MUSA") is PROFILES["musa"], "大小写不敏感"
    assert pick_profile("corex") is PROFILES["corex"]
    assert PROFILES["hygon"].supports_ep_moe and not PROFILES["maca"].supports_ep_moe
    assert not PROFILES["musa"].supports_ep_moe and not PROFILES["corex"].supports_ep_moe

    os.environ["AUTOPLAN_VENDOR"] = "bogus"
    try:
        current_profile()
    except ValueError as exc:
        assert "maca" in str(exc) and "musa" in str(exc), "报错应列出可选档位"
        print("[assert] AUTOPLAN_VENDOR=bogus 报错并列出可选档位")
    else:
        raise AssertionError("非法 AUTOPLAN_VENDOR 未报错")
    _with_vendor(None)
    print("[assert] 回落/大小写/能力位断言全过")


def main() -> None:
    sections = sys.argv[1:] or list(_SECTIONS)
    unknown = [s for s in sections if s not in _SECTIONS]
    assert not unknown, f"未知小节 {unknown}，可选 {_SECTIONS}"
    if "hygon" in sections:
        check_hygon_parity()
    if "vendors" in sections:
        check_vendor_profiles()
    if "frontend" in sections:
        check_musa_frontend_lower()
    if "unit" in sections:
        check_vendor_unit()
    print("\n[regress_vendor] 全部所选小节通过 ✅")


if __name__ == "__main__":
    main()
