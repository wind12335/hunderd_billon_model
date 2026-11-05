#!/usr/bin/env bash
set -euo pipefail

# 该脚本由 Megatron 自动并行 MVP 生成（支持多机）。
# 当前推荐模式: current_feasible
# 搜索策略: BLOCK_LEVEL_FAST / BALANCED
# 推荐并行: TP=8, DP=32, PP=4, EP=8, WORLD_SIZE=1024
# 通信估算(G2/G3 占位): 每卡每步 9701.5 MB / 0.197 s (tp 0.001, dp 0.061, pp 0.004, ep 0.130)，跨节点=['dp', 'pp', 'ep']
# 每步墙钟估算 0.235 s → 存档间隔 2000 步(约1h)/保留 20000 步，单份存档约 1.4 TB；--load 自动续训
# 显式策略: attention=head_shard, mlp=colrow_shard, embedding=vocab_shard, lm_head=vocab_shard
# 三段式策略: attn=(head_shard,head_shard,head_shard), mlp=(colrow_shard,colrow_shard,colrow_shard)
# 注记: 搜索域覆盖 DP + TP + PP（MoE 模型额外包含 EP 候选）。
# 注记: 显存过滤使用 block-level 粗估，并假设开启激活重计算。
# 注记: 选定搜索策略 BLOCK_LEVEL_FAST，策略偏好 BALANCED，耗时 0.0067 秒。
# 注记: 最终显式 ParallelPlan: TP=8, DP=32, PP=4, EP=8, attention=head_shard, mlp=colrow_shard, embedding=vocab_shard, lm_head=vocab_shard。
# 注记: 三段式策略: attn(front/mid/back)=(head_shard,head_shard,head_shard), mlp(front/mid/back)=(colrow_shard,colrow_shard,colrow_shard)。
# 注记: 通信估算(G2/G3 占位模型): 每卡每步 9701.5 MB / 0.197 s，跨节点维度=['dp', 'pp', 'ep']。
# 注记: 每步墙钟估算 0.235 s → 存档间隔 2000 步（约 1h），保留间隔 20000 步；单份存档体积约 1.4 TB。
#
# 多机用法：在每台机器上以不同 NODE_RANK 启动同一份脚本：
#   node0: NNODES=128 NODE_RANK=0 MASTER_ADDR=<主节点IP> bash launch_megatron.sh
#   node1: NNODES=128 NODE_RANK=1 MASTER_ADDR=<主节点IP> bash launch_megatron.sh
# 也可以直接使用 tools/launch_cluster.py 按 hostfile 一键拉起全部节点。

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

source "${ROOT_DIR}/scripts/activate_megatron_hygon.sh"

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export NNODES="${NNODES:-128}"
export NODE_RANK="${NODE_RANK:-0}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
export WORLD_SIZE="$((NNODES * GPUS_PER_NODE))"
export DIST_TIMEOUT_MIN="${DIST_TIMEOUT_MIN:-60}"

export CHECKPOINT_DIR="${CHECKPOINT_DIR:-${ROOT_DIR}/generated/megatron_mvp/checkpoints}"
export TENSORBOARD_DIR="${TENSORBOARD_DIR:-${ROOT_DIR}/generated/megatron_mvp/tensorboard}"
export DATA_CACHE_PATH="${DATA_CACHE_PATH:-${ROOT_DIR}/generated/megatron_mvp/data_cache}"
export DATA_PATH="${DATA_PATH:-MOCK}"
mkdir -p "${CHECKPOINT_DIR}" "${TENSORBOARD_DIR}" "${DATA_CACHE_PATH}"

DATA_ARGS=()
if [[ "${DATA_PATH}" == "MOCK" ]]; then
  DATA_ARGS+=(
    --mock-data
    --tokenizer-type NullTokenizer
    --vocab-size 151552
  )
else
  DATA_ARGS+=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model /root/private_data/hunderd_billon_model/models/glm45_air_base
    --vocab-size 151552
    --data-path ${DATA_PATH}
    --split 949,50,1
    --data-cache-path ${DATA_CACHE_PATH}
  )
fi

python3 -m torch.distributed.run     --nproc_per_node "${GPUS_PER_NODE}"     --nnodes "${NNODES}"     --node_rank "${NODE_RANK}"     --master_addr "${MASTER_ADDR}"     --master_port "${MASTER_PORT}"     "${ROOT_DIR}/tools/run_megatron_hygon.py" Megatron-LM/pretrain_gpt.py     --rank "${RANK:-0}"     --world-size "${WORLD_SIZE}"     --dist-url "tcp://${MASTER_ADDR}:${MASTER_PORT}"     --distributed-timeout-minutes "${DIST_TIMEOUT_MIN}"     --use-mcore-models \
    --num-layers 46 \
    --hidden-size 4096 \
    --ffn-hidden-size 10944 \
    --num-attention-heads 96 \
    --seq-length 4096 \
    --max-position-embeddings 131072 \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --swiglu \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --rotary-base 1000000 \
    --num-query-groups 8 \
    --kv-channels 128 \
    --rotary-percent 0.5 \
    --add-qkv-bias \
    --num-experts 128 \
    --moe-router-topk 8 \
    --moe-ffn-hidden-size 1408 \
    --moe-shared-expert-intermediate-size 1408 \
    --moe-layer-freq [0]*1+[1]*45 \
    --moe-aux-loss-coeff 0.0001 \
    --moe-router-load-balancing-type aux_loss \
    --moe-router-score-function sigmoid \
    --moe-router-enable-expert-bias \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 4 \
    --sequence-parallel \
    --expert-model-parallel-size 8 \
    --micro-batch-size 1 \
    --global-batch-size 32 \
    --bf16 \
    --use-distributed-optimizer \
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1 \
    --save ${CHECKPOINT_DIR} \
    --load ${CHECKPOINT_DIR} \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --log-interval 1 \
    --eval-interval 100 \
    --eval-iters 10 \
    --save-interval 2000 \
    --save-retain-interval 20000 \
    --use-dist-ckpt \
    --async-save \
    "${DATA_ARGS[@]}"
