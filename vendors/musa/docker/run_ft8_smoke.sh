#!/usr/bin/env bash
set -euo pipefail
# TP=4/DP=2 冒烟：小 gbs=16 → grad_accum=8，3 步快速验证
#  ckpt(TP=4 分片)加载 / distributed optimizer(DP=2) / MCCL TP+DP / loss / save
docker run --rm \
    --network host \
    --shm-size=64g \
    --device=/dev/mtgpu.0 --device=/dev/mtgpu.1 --device=/dev/mtgpu.2 --device=/dev/mtgpu.3 \
    --device=/dev/mtgpu.4 --device=/dev/mtgpu.5 --device=/dev/mtgpu.6 --device=/dev/mtgpu.7 \
    -v /usr/lib64/libmusa.so.4.3.5:/lib/x86_64-linux-gnu/libmusa.so.1 \
    -v /root/workspace:/ws \
    sh-harbor.mthreads.com/sets/sqt-vllm-train-musa-bench:musa-sdk-4.3.5_torch2.7.1-intel \
    bash -c 'export TRAIN_ITERS=3 LR_WARMUP_ITERS=1 GLOBAL_BATCH_SIZE=16; bash /ws/musa_framework/generated/run_13b/launch_ft8.sh 2>&1 | tee /ws/ft8_smoke_tp4.log'
