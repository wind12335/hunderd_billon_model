"""G2/G3/G7/MLA 适配验证驱动：通信模型单元 + MLA 全链路 + 存档策略。"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["AUTOPLAN_STANDIN_GPUS"] = "64"

from megatron_mvp import detect, search, render
from megatron_mvp.comm_model import estimate_communication, estimate_active_params
from megatron_mvp.environment import detect_runtime_environment
from megatron_mvp.types import ModelSpec, RuntimeEnvironment

REPO = Path(__file__).resolve().parents[1]

# ---------- 1. 通信模型单元 ----------
glm = detect.load_model_spec(detect.detect_hf_model(REPO / "models"))
env = detect_runtime_environment()

def comm(tp, dp, pp, ep, mbs=1, seq=4096, grads=10 * 1024**3, e=None):
    return estimate_communication(glm, e or env, tp, dp, pp, ep, mbs, seq, grads)

c_ep1 = comm(8, 8, 1, 1)
c_ep8 = comm(8, 8, 1, 8)
c_ep4 = comm(8, 8, 1, 4)
assert c_ep1.ep_bytes == 0 and c_ep1.ep_seconds == 0.0, "EP=1 不应有 a2a"
assert c_ep8.ep_bytes > 0 and c_ep8.ep_seconds > 0
ratio = c_ep8.ep_bytes / c_ep4.ep_bytes
assert abs(ratio - (7 / 8) / (3 / 4)) < 0.01, f"a2a (ep-1)/ep 比例错: {ratio}"
print(f"[comm] a2a 比例 OK: ep8/ep4 = {ratio:.3f} (期望 {7/6:.3f})")

c_tp_intra = comm(8, 4, 1, 1)
small_node = RuntimeEnvironment(
    accelerator_type="hip", device_count=64, devices=[], primary_device_name="x",
    primary_device_memory_bytes=64 * 1024**3, torch_version="x", hip_version=None,
    gpus_per_node=4,
)
c_tp_inter = comm(8, 4, 1, 1, e=small_node)
assert not c_tp_intra.cross_node["tp"] and c_tp_inter.cross_node["tp"]
assert c_tp_inter.tp_seconds > c_tp_intra.tp_seconds, "跨节点 TP 应更慢"
print(f"[comm] 拓扑翻转 OK: tp8 机内 {c_tp_intra.tp_seconds*1e3:.1f}ms vs 4卡节点 {c_tp_inter.tp_seconds*1e3:.1f}ms")

dp_expect = int(10 * 1024**3 * 2 * 7 / 8)  # grads × 2×(dp-1)/dp, dp=8
assert comm(8, 8, 1, 1).dp_bytes == dp_expect, "DP 梯度同步字节公式错"
print(f"[comm] DP 字节公式 OK: {dp_bytes_check if (dp_bytes_check := comm(8,8,1,1).dp_bytes) == dp_expect else ''}")

# ---------- 2. MLA：探测 + 参数公式 + 搜索 + 渲染 ----------
ling_cfg = {
    "hidden_size": 4096, "num_attention_heads": 48, "num_key_value_heads": 48,
    "q_lora_rank": 1536, "kv_lora_rank": 512,
    "qk_nope_head_dim": 128, "qk_rope_head_dim": 64, "v_head_dim": 128,
}
mla = detect._extract_mla_fields(ling_cfg)
assert mla["kv_lora_rank"] == 512 and mla["qk_nope_head_dim"] == 128
assert detect._extract_mla_fields({"hidden_size": 8}) == {}, "非 MLA 应返回空"
print(f"[mla] 探测 OK: {mla}")

ling = ModelSpec(
    model_dir=REPO, model_type="ling", architecture="LingMoeForCausalLM",
    hidden_size=4096, intermediate_size=8192, num_layers=45, num_heads=48, num_kv_heads=48,
    vocab_size=160000, max_seq_len=8192, rope_theta=5000000.0, rms_norm_eps=1e-6,
    tie_word_embeddings=False, source_torch_dtype="bfloat16",
    estimated_total_params=1, estimated_params_per_layer=1,
    num_experts=288, num_experts_per_tok=16, moe_intermediate_size=768,
    shared_expert_intermediate_size=1536, first_k_dense_replace=1,
    use_mla=True, q_lora_rank=1536, kv_lora_rank=512,
    qk_nope_head_dim=128, qk_rope_head_dim=64, mla_v_head_dim=128,
)
# 用 detect 的 MLA 感知估算器回填总参（顺带验证该估算路径）。
tp_ling, avg_ling = detect._estimate_moe_model_params(
    hidden_size=4096, intermediate_size=8192, num_layers=45, num_heads=48, num_kv_heads=48,
    head_dim=None, num_experts=288, moe_intermediate_size=768,
    shared_expert_intermediate_size=1536, first_k_dense_replace=1, mla_fields=mla)
ling = ModelSpec(
    **{**ling.to_dict(), "model_dir": REPO,
       "estimated_total_params": tp_ling + 160000 * 4096 * 2 + 4096,
       "estimated_params_per_layer": int(avg_ling)})
print(f"[mla] 总参估算(MLA 感知): {(tp_ling + 160000*4096*2 + 4096)/1e9:.1f}B")
comps = search._estimate_transformer_component_params(ling)
# 手算 MLA attention: h*qr + qr*H*(nr+rr) + h*(kr+rr) + kr*H*(nr+v) + H*v*h
h, H, qr, kr, nr, rr, v = 4096, 48, 1536, 512, 128, 64, 128
mla_attn = h*qr + qr*H*(nr+rr) + h*(kr+rr) + kr*H*(nr+v) + H*v*h
assert comps["attention"] == mla_attn, f"MLA attention 公式错: {comps['attention']} vs {mla_attn}"
print(f"[mla] attention 参数公式 OK: {mla_attn/1e6:.1f}M/层 (GQA 版本应为 ~{h*H*(nr+rr)*2//1e6}M 量级对比)")

res = search.search_parallel_candidates(
    spec=ling, env=env, seq_length=4096, max_world_size=128,
    strategy="BLOCK_LEVEL_FAST", policy="BALANCED", max_pp=2, max_ep=16,
    benchmark_all_builtin=False)
p = res.selected_plan
sc = res.selected_candidate
assert sc.communication is not None
print(f"[mla] 搜索 OK: TP={p.tp} DP={p.dp} PP={p.pp} EP={p.ep} est={p.estimated_memory_bytes/2**30:.2f}G "
      f"comm={sc.communication.total_bytes/1024**2:.0f}MB/{sc.communication.total_seconds*1e3:.0f}ms")

ir = render.build_coarse_ir(ling, env, p, res)
cfg = render.build_megatron_config(ir)
ma = cfg["model_args"]
for flag in ("multi_latent_attention", "q_lora_rank", "kv_lora_rank",
             "qk_head_dim", "qk_pos_emb_head_dim", "v_head_dim", "num_experts"):
    assert flag in ma, f"缺 MLA/MoE 标志 {flag}"
assert "num_query_groups" not in ma, "MLA 不应渲染 GQA 标志"
assert ma["qk_head_dim"] == 128 and ma["v_head_dim"] == 128
print(f"[mla] 渲染 OK: {ma['multi_latent_attention']=} {ma['kv_lora_rank']=} 无GQA标志")

# ---------- 3. G7：存档策略 ----------
ra = cfg["runtime_args"]
si, sri = ra["save_interval"], ra["save_retain_interval"]
assert 50 <= si <= 2000 and sri == si * 10 and sri % si == 0
assert ra["use_dist_ckpt"] is True and ra["async_save"] is True
assert ir.communication["estimated_checkpoint_bytes"] > 1e12, "千亿级存档体积估算缺失"
print(f"[g7] GLM 侧: 每步≈{ir.communication['estimated_step_seconds']:.2f}s → save={si} retain={sri} "
      f"单份≈{ir.communication['estimated_checkpoint_bytes']/1024**4:.1f}TB")

# GLM(真实权重 spec) 全链路确认通信块存在
res2 = search.search_parallel_candidates(
    spec=glm, env=env, seq_length=4096, max_world_size=128,
    strategy="BLOCK_LEVEL_FAST", policy="BALANCED", max_pp=2, max_ep=8,
    benchmark_all_builtin=False)
p2 = res2.selected_plan
ir2 = render.build_coarse_ir(glm, env, p2, res2)
cfg2 = render.build_megatron_config(ir2)
assert ir2.communication is not None and "save_interval_recommendation" in ir2.communication
assert cfg2["runtime_args"]["save_retain_interval"] % cfg2["runtime_args"]["save_interval"] == 0
assert "num_query_groups" in cfg2["model_args"], "GLM 的 GQA 标志不应受影响"
assert "multi_latent_attention" not in cfg2["model_args"], "GLM 非 MLA 不应渲染 MLA 标志"
print(f"[g7] GLM 全链路 OK: save={cfg2['runtime_args']['save_interval']} "
      f"retain={cfg2['runtime_args']['save_retain_interval']} "
      f"comm={ir2.communication['total_bytes']/1024**2:.0f}MB")
print("[assert] 全部通过")
