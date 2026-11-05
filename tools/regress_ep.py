"""EP 适配回归驱动：新 spec 探测 + 三组回归 + PPO 冒烟。"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from megatron_mvp import detect, search
from megatron_mvp.environment import detect_runtime_environment as detect_environment
from megatron_mvp.types import ModelSpec

REPO = Path(__file__).resolve().parents[1]


def glm_spec():
    spec = detect.load_model_spec(detect.detect_hf_model(REPO / "models"))
    print(f"[spec] GLM: is_moe={spec.is_moe} experts={spec.num_experts} "
          f"attn_bias={spec.attention_bias} router={spec.moe_router_score_function} "
          f"rotary={spec.partial_rotary_factor}")
    assert spec.attention_bias, "attention_bias 未探测到"
    assert spec.moe_router_score_function == "sigmoid", "sigmoid 路由未探测到"
    return spec


def llama_spec():
    d = json.loads((REPO / "generated/dual16_llama8b/detection_summary.json").read_text())["model_spec"]
    d["model_dir"] = Path(d["model_dir"])
    return ModelSpec(**d)


def run(tag, spec, devices, world, pp, ep, strategy="BLOCK_LEVEL_FAST", policy="BALANCED", seq=2048):
    os.environ["AUTOPLAN_STANDIN_GPUS"] = str(devices)
    env = detect_environment()
    res = search.search_parallel_candidates(
        spec=spec, env=env, seq_length=seq, max_world_size=world,
        strategy=strategy, policy=policy, max_pp=pp, max_ep=ep,
        benchmark_all_builtin=False,
    )
    p = res.selected_plan
    print(f"[{tag}] TP={p.tp} DP={p.dp} PP={p.pp} EP={p.ep} mbs={p.micro_batch_size} "
          f"gbs={p.global_batch_size} est={p.estimated_memory_bytes/2**30:.2f}G "
          f"margin={p.memory_margin_bytes/2**30:.2f}G "
          f"({res.strategy_name}, {res.search_time_seconds:.2f}s)")
    return p


glm = glm_spec()
p64 = run("glm64", glm, 64, 128, 2, 8)
p128 = run("glm128", glm, 128, 128, 2, 8)
dense = llama_spec()
print(f"[spec] Llama dense: is_moe={dense.is_moe} attn_bias={dense.attention_bias} "
      f"router={dense.moe_router_score_function}")
p16 = run("llama16", dense, 16, 128, 2, 8)

# 期望断言
assert (p64.tp, p64.dp, p64.pp, p64.ep) == (8, 8, 1, 8), f"glm64 偏离旧基线: {p64.mesh}"
assert (p128.tp, p128.dp, p128.pp, p128.ep) == (8, 8, 2, 8), f"glm128 偏离旧基线: {p128.mesh}"
p16_old = json.loads((REPO / "generated/dual16_llama8b/search_result.json").read_text())["selected_plan"]
assert (p16.tp, p16.dp, p16.pp, p16.ep) == (p16_old["tp"], p16_old["dp"], p16_old["pp"], p16_old["ep"]), \
    f"稠密回归失败: {p16.mesh} vs 旧 {p16_old['mesh']}"
print("[assert] 三组回归全部通过")

# PPO 冒烟（ep 进决策节点）
os.environ["AUTOPLAN_STANDIN_GPUS"] = "64"
env = detect_environment()
nodes = search._build_decision_nodes(
    search._build_mesh_choices(spec=glm, env=env, max_world_size=128, max_pp=2),
    spec=glm, max_ep=8)
print(f"[ppo] 决策节点: {[(n.node_name, n.choices) for n in nodes]}")
res = search.search_parallel_candidates(
    spec=glm, env=env, seq_length=2048, max_world_size=128,
    strategy="RL_GUIDED_PPO", policy="BALANCED", max_pp=2, max_ep=8,
    benchmark_all_builtin=False)
p = res.selected_plan
print(f"[ppo] TP={p.tp} DP={p.dp} PP={p.pp} EP={p.ep} margin={p.memory_margin_bytes/2**30:.2f}G "
      f"({res.strategy_name}, {res.search_time_seconds:.2f}s)")
assert p.ep > 1, "PPO 未选出 EP>1"
print("[assert] PPO 冒烟通过")
