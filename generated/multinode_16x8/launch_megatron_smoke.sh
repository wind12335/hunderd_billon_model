#!/usr/bin/env bash
set -euo pipefail

# 该脚本由 autoparallel 声明式前端生成，并下放到现有 Megatron 执行（支持多机）。
# 脚本角色: smoke_test
# 搜索策略: BLOCK_LEVEL_FAST / BALANCED
# 搜索耗时: 0.0038 秒
# 规划推荐: TP=8, DP=8, PP=2, WORLD_SIZE=128
# 显式策略: attention=head_shard, mlp=colrow_shard, embedding=vocab_shard, lm_head=vocab_shard
# 三段式策略: attn=(head_shard,head_shard,head_shard), mlp=(colrow_shard,colrow_shard,colrow_shard)
# 当前执行: 当前 smoke test 默认回退到单卡串行配置，只验证数据链路和 Megatron 启动链路。
#
# 单机用法:  bash launch_megatron_smoke.sh
# 多机用法:  在每台节点上用不同 NODE_RANK 启动同一份脚本:
#   node0: NNODES=16 NODE_RANK=0 MASTER_ADDR=<主节点IP> bash launch_megatron_smoke.sh
#   node1: NNODES=16 NODE_RANK=1 MASTER_ADDR=<主节点IP> bash launch_megatron_smoke.sh
# 或在主节点直接: bash launch_cluster.sh (由 hostfile 驱动 tools/launch_cluster.py)

ROOT_DIR="/root/private_data/thousand_model"
cd "${ROOT_DIR}"

source "${ROOT_DIR}/scripts/activate_megatron_hygon.sh"

export AUTOPARALLEL_ROLE="smoke_test"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export NNODES="${NNODES:-16}"
export NODE_RANK="${NODE_RANK:-0}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
export WORLD_SIZE="$((NNODES * GPUS_PER_NODE))"
export DIST_TIMEOUT_MIN="${DIST_TIMEOUT_MIN:-60}"

export CHECKPOINT_DIR="${CHECKPOINT_DIR:-${ROOT_DIR}/generated/autoparallel/checkpoints}"
export TENSORBOARD_DIR="${TENSORBOARD_DIR:-${ROOT_DIR}/generated/autoparallel/tensorboard}"
export DATA_CACHE_PATH="${DATA_CACHE_PATH:-/root/private_data/thousand_model/generated/multinode_16x8/data_cache}"
mkdir -p "${CHECKPOINT_DIR}" "${TENSORBOARD_DIR}" "${DATA_CACHE_PATH}"

DATA_ARGS=(
    --mock-data
    --tokenizer-type NullTokenizer
    --vocab-size 32000
)

python3 -m torch.distributed.run     --nproc_per_node "${GPUS_PER_NODE}"     --nnodes "${NNODES}"     --node_rank "${NODE_RANK}"     --master_addr "${MASTER_ADDR}"     --master_port "${MASTER_PORT}"     "${ROOT_DIR}/tools/run_megatron_hygon.py" Megatron-LM/pretrain_gpt.py     --rank "${RANK:-0}"     --world-size "${WORLD_SIZE}"     --dist-url "tcp://${MASTER_ADDR}:${MASTER_PORT}"     --distributed-timeout-minutes "${DIST_TIMEOUT_MIN}"     --use-mcore-models \
    --num-layers 46 \
    --hidden-size 4096 \
    --ffn-hidden-size 10944 \
    --num-attention-heads 96 \
    --seq-length 4096 \
    --max-position-embeddings 40960 \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --swiglu \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --rotary-base 1000000 \
    --num-query-groups 8 \
    --kv-channels 128 \
    --num-experts 128 \
    --moe-router-topk 8 \
    --moe-ffn-hidden-size 1408 \
    --moe-shared-expert-intermediate-size 10944 \
    --moe-layer-freq [0]*1+[1]*45 \
    --moe-aux-loss-coeff 0.0001 \
    --moe-router-load-balancing-type aux_loss \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --expert-model-parallel-size 1 \
    --micro-batch-size 1 \
    --global-batch-size 1 \
    --bf16 \
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1 \
    --train-iters 2 \
    --lr 0.0001 \
    --min-lr 1e-05 \
    --lr-decay-style cosine \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --init-method-std 0.02 \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --log-interval 1 \
    --eval-interval 1 \
    --eval-iters 0 \
    --save-interval 1000 \
    "${DATA_ARGS[@]}"
