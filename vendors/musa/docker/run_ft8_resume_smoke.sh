#!/usr/bin/env bash
set -euo pipefail
# 续训验证：RESUME=1 + 真实配置（warmup=200, iters=1250，与原始训练一致——
# 本版本 scheduler 校验 checkpoint 的 warmup/decay，压缩参数必崩）。
# 跑 2 步确认从 iter_400 续上且 loss 衔接后由外部 docker kill 停止。
# 日志独立文件 /ws/ft8_resume_smoke.log，不碰 ft8_full.log；
# 2 步不在 50 步保存边界，不写新 checkpoint，iter_400 保持最新。
docker run --rm \
    --network host \
    --shm-size=64g \
    -e RESUME=1 \
    --device=/dev/mtgpu.0 --device=/dev/mtgpu.1 --device=/dev/mtgpu.2 --device=/dev/mtgpu.3 \
    --device=/dev/mtgpu.4 --device=/dev/mtgpu.5 --device=/dev/mtgpu.6 --device=/dev/mtgpu.7 \
    -v /usr/lib64/libmusa.so.4.3.5:/lib/x86_64-linux-gnu/libmusa.so.1 \
    -v /root/workspace:/ws \
    sh-harbor.mthreads.com/sets/sqt-vllm-train-musa-bench:musa-sdk-4.3.5_torch2.7.1-intel \
    bash -c "bash /ws/musa_framework/generated/run_13b/launch_ft8.sh 2>&1 | tee /ws/ft8_resume_smoke.log"
