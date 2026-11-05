#!/usr/bin/env bash
set -euo pipefail
# 全量微调 colossal-llama2-13B（TP=4/DP=2, gbs=256, ~1250 iters = 1 epoch）
#
# 执行方式（用户亲自跑）：
#   从头微调:   bash /root/workspace/run_ft8_full.sh
#   断点续训:   RESUME=1 bash /root/workspace/run_ft8_full.sh
#     （从中断处的最新 checkpoint 继续；前提是已有 iter_200 及以上 checkpoint）
#
# 续训时日志用追加模式(tee -a)：新段接在旧日志后面，run_loss_report.sh
# 重跑即可得到 0→1250 完整曲线；从头微调则覆盖日志（旧段作废）。
#
# 日志实时查看（在宿主机另开终端）：
#   tail -f /root/workspace/ft8_full.log
#   （容器内 tee /ws/ft8_full.log → 挂载到宿主机 /root/workspace/ft8_full.log）
#
# checkpoint 策略（bootstrap 补丁 #6 实现）：
#   --save-interval 50   每 50 步保存一次（50/100/.../1250，末步 1250 额外保存）
#   （默认 50，用户过夜决定不重启；可用 SAVE_INTERVAL=200 覆盖）
#   滚动保留最近 3 个（iter_* 目录），更旧的自动删除，仅 rank0 删，日志可见
#
# 时长估算（实测 66s/步）：1250 步 ≈ 23 小时

# 续训(RESUME=1)追加日志，从头则覆盖
if [[ "${RESUME:-0}" == "1" ]]; then TEE_FLAG="-a"; else TEE_FLAG=""; fi

docker run --rm \
    --network host \
    --shm-size=64g \
    -e RESUME="${RESUME:-0}" \
    --device=/dev/mtgpu.0 --device=/dev/mtgpu.1 --device=/dev/mtgpu.2 --device=/dev/mtgpu.3 \
    --device=/dev/mtgpu.4 --device=/dev/mtgpu.5 --device=/dev/mtgpu.6 --device=/dev/mtgpu.7 \
    -v /usr/lib64/libmusa.so.4.3.5:/lib/x86_64-linux-gnu/libmusa.so.1 \
    -v /root/workspace:/ws \
    sh-harbor.mthreads.com/sets/sqt-vllm-train-musa-bench:musa-sdk-4.3.5_torch2.7.1-intel \
    bash -c "bash /ws/musa_framework/generated/run_13b/launch_ft8.sh 2>&1 | tee ${TEE_FLAG} /ws/ft8_full.log"
