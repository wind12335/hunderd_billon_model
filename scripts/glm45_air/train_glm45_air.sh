#!/bin/bash

# GLM-4.5-Air (105B, glm4_moe) 全量微调 per-node 训练脚本（BW1000 / 128 卡）。
# 结构参考 Megatron-LM-das/examples/GLM/train_glm5_300B.sh（海光双启动模式）。
#
# GLM-4.5-Air 是 GQA + partial-RoPE + MoE 结构（Glm4MoeForCausalLM），
# Megatron 原生 GPT 参数无法直接表达 partial rotary，
# 因此走 Megatron-Bridge（--use-bridge --bridge-hf-model）由 GLM45Bridge
# 自动完成 HF Glm4MoeForCausalLM <-> Megatron-Core 的结构与权重映射。
#
# 并行基线（128 卡 = 16 节点 × 8 卡，TP×PP×DP = 2×4×16，EP 在 DP 内切 8 份）:
#   每卡显存估算（bf16 + Adam fp32 状态 + 分布式优化器）:
#     权重 ~3GB + 梯度 ~3GB + 优化器分片 ~10GB + 激活(重计算) ~5GB ≈ 21GB / 64GB，余量充足。
#
# 双机 16 卡冒烟：把下面的 TP/PP/EP/DP 相应调小（例如 TP=2, PP=2, EP=2，
# GBS 同步减小），并建议先冻结/少步数验证链路。

for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=${para#*=}
    elif [[ $para == --tokenizer_path* ]];then
        tokenizer_path=${para#*=}
    elif [[ $para == --checkpoint_path* ]];then
        checkpoint_path=${para#*=}
    elif [[ $para == --launch_with_binding* ]];then
        launch_with_binding=${para#*=}
    elif [[ $para == --launch_backend* ]];then
        launch_backend=${para#*=}
    elif [[ $para == --hf_model_path* ]];then
        hf_model_path=${para#*=}
    fi
done

# data path
DATA_PATH=${data_path}
TOKENIZER_MODEL_PATH=${tokenizer_path}
CHECKPOINT_PATH=${checkpoint_path}
HF_MODEL_PATH=${hf_model_path}   # GLM-4.5-Air HF 权重目录（bridge 从这里读结构与权重）

# default env（由 run_multinode.sh 传入或在 torchrun 模式下取环境变量）
DIST_URL=${1}
DIST_PORT=${2}
RANK=$OMPI_COMM_WORLD_RANK
LOCAL_RANK=$OMPI_COMM_WORLD_LOCAL_RANK
WORLD_SIZE=$OMPI_COMM_WORLD_SIZE
export MEGATRON_LAUNCH_BACKEND=${launch_backend:-"mpirun"}
MASTER_ADDR=${MASTER_ADDR:-loadlhost}
MASTER_PORT=${MASTER_PORT:-6000}
NNODES=${NNODES:-16}
NODE_RANK=${NODE_RANK:-${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-0}}}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
MEGATRON_PATH=$(dirname "$(dirname "${CURRENT_DIR}")")/Megatron-LM-das

export GPU_MAX_HW_QUEUES=4
export NVTE_USE_HIPBLASLT_GROUPEDGEMM=1
export NVTE_OVERLAP_GRAD_REDUCE=1

# ===== 并行切分（128 卡基线；双机 16 卡冒烟请改这里）=====
TP=2
PP=4
CP=1
EP=8
ETP=2

# ===== batch =====
MBS=2
GBS=256

# ===== 序列 =====
SEQ_LEN=4096
MAX_POSITION_EMBEDDINGS=40960

# ===== 训练 =====
TRAIN_ITERS=1000

MPI_DISTRIBUTED_ARGS=(
    --rank ${RANK}
    --world-size ${WORLD_SIZE}
    --local-rank ${LOCAL_RANK}
    --dist-url tcp://${DIST_URL}:${DIST_PORT}
    --distributed-timeout-minutes 60
    --distributed-backend nccl
)

TORCH_DISTRIBUTED_ARGS=(
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
    --nproc_per_node $GPUS_PER_NODE
)

# GLM-4.5-Air 结构参数由 bridge 从 HF config 读取；
# 这里只声明 Megatron 侧与结构强相关的少量参数（层数/微调超参）。
MODEL_ARGS=(
    --use-mcore-models
    --use-bridge
    --bridge-hf-model ${HF_MODEL_PATH}
    --seq-length ${SEQ_LEN}
    --max-position-embeddings ${MAX_POSITION_EMBEDDINGS}
    --num-layers 46
    --disable-bias-linear
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --norm-epsilon 1e-5
    --position-embedding-type rope
    --swiglu
    --untie-embeddings-and-output-weights
    --use-flash-attn
    --no-check-for-nan-in-loss-and-grad
    --no-create-attention-mask-in-dataloader
    --make-vocab-size-divisible-by 128
    --manual-gc
    --manual-gc-interval 20
)

# MoE 行为参数（专家个数/topk/宽度都来自 HF config，这里只调路由行为）。
MOE_ARGS=(
    --moe-aux-loss-coeff 1e-4
    --moe-token-dispatcher-type alltoall
    --moe-router-dtype fp32
    --moe-router-pre-softmax
    --moe-router-load-balancing-type aux_loss
)

DATA_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model ${TOKENIZER_MODEL_PATH}
    --data-path ${DATA_PATH}
    --split 949,50,1
    --num-workers 6
    --no-mmap-bin-files
)

# 全量微调超参（小 lr + cosine）
TRAINING_ARGS=(
    --train-iters ${TRAIN_ITERS}
    --micro-batch-size ${MBS}
    --global-batch-size ${GBS}
    --lr 1.0e-5
    --min-lr 1.0e-6
    --lr-decay-style cosine
    --lr-warmup-iters 20
    --weight-decay 0.1
    --clip-grad 1.0
    --bf16
    --adam-beta1 0.9
    --adam-beta2 0.95
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size ${TP}
    --pipeline-model-parallel-size ${PP}
    --expert-model-parallel-size ${EP}
    --expert-tensor-parallel-size ${ETP}
    --context-parallel-size ${CP}
    --num-layers-per-virtual-pipeline-stage 2
    --use-distributed-optimizer
    --sequence-parallel
    --overlap-param-gather
    --overlap-grad-reduce
)

LOGGING_ARGS=(
    --log-throughput
    --log-interval 1
    --log-memory-to-tensorboard
    --save-interval 500
    --eval-interval 200
    --eval-iters 10
    --save $CHECKPOINT_PATH
    --load $CHECKPOINT_PATH
    --tensorboard-dir "${CHECKPOINT_PATH}/tensorboard"
    --no-load-optim
    --no-load-rng
    --auto-detect-ckpt-format
    --dist-ckpt-strictness log_all
)

if [[ "$MEGATRON_LAUNCH_BACKEND" == "mpirun" ]]; then
    APP="python3 -u ${MEGATRON_PATH}/pretrain_gpt.py \
        ${MPI_DISTRIBUTED_ARGS[@]} \
        ${MODEL_ARGS[@]} \
        ${MOE_ARGS[@]} \
        ${DATA_ARGS[@]} \
        ${TRAINING_ARGS[@]} \
        ${MODEL_PARALLEL_ARGS[@]} \
        ${LOGGING_ARGS[@]} \
        "
elif [[ "$MEGATRON_LAUNCH_BACKEND" == "torchrun" ]]; then
    APP="torchrun ${TORCH_DISTRIBUTED_ARGS[@]} \
        ${MEGATRON_PATH}/pretrain_gpt.py \
        ${MODEL_ARGS[@]} \
        ${MOE_ARGS[@]} \
        ${DATA_ARGS[@]} \
        ${TRAINING_ARGS[@]} \
        ${MODEL_PARALLEL_ARGS[@]} \
        ${LOGGING_ARGS[@]} \
        "
else
    echo "Only mpirun and torchrun are supported as launch methods"
    exit 1
fi

# for hygon cpu
if [[ "$MEGATRON_LAUNCH_BACKEND" == "mpirun" ]]; then
    ${launch_with_binding} ${LOCAL_RANK} ${APP}
else
    echo ${APP}
    ${APP}
fi
