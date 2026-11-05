#!/usr/bin/env bash
# =============================================================================
# 双机 16 卡 MoE（专家并行 EP）冒烟一键启动脚本
#
# 用法（SCNet vcjob）：作业"启动脚本"直接填本脚本路径即可，不必手工配
# 环境变量——下面这组默认值就是说明书 Run A；要改的用作业环境变量覆盖
# （全部取 XXX="${XXX:-默认}" 形式，同名作业环境变量优先）。
#
#   Run A（默认提交）      : EP 冒烟 + 通信微基准（BENCH_COMM=1）
#   Run B（跨机 a2a 冒烟） : 作业环境变量加两个
#                             FORCE_EP=4  RUN_TAG=glm16_ep_smoke_xnode
#                            FORCE_EP 由入口脚本在计划生成后覆写
#                            --expert-model-parallel-size；重提交会重新生成
#                            计划，手工改生成文件会被覆盖，所以走环境变量
#
# 本脚本只是 scripts/run_dual_node_test.sh 的"参数皮"：编排/RDMA/日志判据
# 全在入口脚本里。说明书：docs/双机16卡_EP校准与冒烟打包_20261024.md
# =============================================================================
# --- POSIX 前奏：平台可能用 /bin/sh 调起，先切换到 bash ---
if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi
printf '[moe-smoke] 启动脚本已执行 hostname=%s bash=%s\n' "$(hostname 2>/dev/null || echo '?')" "${BASH_VERSION}"
set -euo pipefail

# ── 本试验参数（默认 = 说明书 Run A） ────────────────────────────────────────
export NUM_NODES="${NUM_NODES:-2}"                         # 双机
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"                 # 每节点 8 卡 → world=16
export MODEL_PATH="${MODEL_PATH:-models/glm45_air_smoke}"  # 8 层切片 MoE stub
export RUN_TAG="${RUN_TAG:-glm16_ep_smoke}"                # 产物目录 generated/<RUN_TAG>
export TRAIN_ITERS="${TRAIN_ITERS:-20}"                    # 冒烟 20 步
export SEQ_LEN="${SEQ_LEN:-4096}"
export SEARCH_MAX_PP="${SEARCH_MAX_PP:-2}"
export SEARCH_MAX_EP="${SEARCH_MAX_EP:-8}"                 # EP 搜索上限必须放开
export BENCH_COMM="${BENCH_COMM:-1}"                       # 训练前跑通信微基准
if [[ -n "${FORCE_EP:-}" ]]; then
    [[ "$FORCE_EP" =~ ^[1-9][0-9]*$ ]] || { printf '[moe-smoke][失败] FORCE_EP 必须是正整数: %s\n' "$FORCE_EP" | tee /dev/stderr; exit 1; }
    export FORCE_EP
fi

printf '[moe-smoke] 参数: world=%s卡  model=%s  tag=%s  FORCE_EP=%s\n' \
    "$((NUM_NODES * GPUS_PER_NODE))" "$MODEL_PATH" "$RUN_TAG" "${FORCE_EP:-无(走规划选型)}"
printf '[moe-smoke] 预期计划: TP4/DP4/PP1/EP%s(约38.1G/64G)  判据: iteration %s/%s ... loss= 且 NCCL 日志含 NET/IBext_v8\n' \
    "${FORCE_EP:-2}" "$TRAIN_ITERS" "$TRAIN_ITERS"

# ── 定位入口脚本（平台可能把启动脚本 staging 到别处，不能只依赖自身路径） ────
self_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd || true)"
entry=""
for c in "${THOUSAND_MODEL_ROOT:-}/scripts" "$self_dir" \
         "/root/private_data/hunderd_billon_model/scripts" \
         "${HOME:-/root}/private_data/hunderd_billon_model/scripts"; do
    if [ -n "$c" ] && [ -f "$c/run_dual_node_test.sh" ]; then
        entry="$c/run_dual_node_test.sh"
        break
    fi
done
if [ -z "$entry" ]; then
    printf '[moe-smoke][失败] 找不到 run_dual_node_test.sh；可用环境变量 THOUSAND_MODEL_ROOT=<仓库根> 指定\n' | tee /dev/stderr
    exit 1
fi

exec bash "$entry" "$@"
