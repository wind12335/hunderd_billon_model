#!/usr/bin/env bash
set -euo pipefail

# 该脚本由 Megatron 自动并行 MVP 生成（支持多机）。
# 当前推荐模式: current_feasible
# 搜索策略: BLOCK_LEVEL_FAST / BALANCED
# 推荐并行: TP=4, DP=4, PP=1, EP=1, WORLD_SIZE=16
# 显式策略: attention=head_shard, mlp=colrow_shard, embedding=vocab_shard, lm_head=vocab_shard
# 三段式策略: attn=(head_shard,head_shard,head_shard), mlp=(colrow_shard,colrow_shard,colrow_shard)
#
# 多机用法：在每台机器上以不同 NODE_RANK 启动同一份脚本：
#   node0: NNODES=2 NODE_RANK=0 MASTER_ADDR=<主节点IP> bash launch_megatron.sh
#   node1: NNODES=2 NODE_RANK=1 MASTER_ADDR=<主节点IP> bash launch_megatron.sh
# 也可以直接使用 tools/launch_cluster.py 按 hostfile 一键拉起全部节点。

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

source "${ROOT_DIR}/scripts/activate_megatron_hygon.sh"

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export NNODES="${NNODES:-2}"
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
    --vocab-size 128256
  )
else
  DATA_ARGS+=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model /root/private_data/thousand_model/model/Llama-3.1-8B
    --vocab-size 128256
    --data-path ${DATA_PATH}
    --split 949,50,1
    --data-cache-path ${DATA_CACHE_PATH}
  )
fi

python3 -m torch.distributed.run     --nproc_per_node "${GPUS_PER_NODE}"     --nnodes "${NNODES}"     --node_rank "${NODE_RANK}"     --master_addr "${MASTER_ADDR}"     --master_port "${MASTER_PORT}"     "${ROOT_DIR}/tools/run_megatron_hygon.py" Megatron-LM/pretrain_gpt.py     --rank "${RANK:-0}"     --world-size "${WORLD_SIZE}"     --dist-url "tcp://${MASTER_ADDR}:${MASTER_PORT}"     --distributed-timeout-minutes "${DIST_TIMEOUT_MIN}"     --use-mcore-models \
    --num-layers 32 \
    --hidden-size 4096 \
    --ffn-hidden-size 14336 \
    --num-attention-heads 32 \
    --seq-length 4096 \
    --max-position-embeddings 131072 \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --swiglu \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --rotary-base 500000 \
    --num-query-groups 8 \
    --tensor-model-parallel-size 4 \
    --pipeline-model-parallel-size 1 \
    --sequence-parallel \
    --micro-batch-size 1 \
    --global-batch-size 4 \
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
    --save-interval 100 \
    "${DATA_ARGS[@]}"
