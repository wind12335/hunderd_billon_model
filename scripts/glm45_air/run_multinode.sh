#!/bin/bash

# GLM-4.5-Air 多机拉起器（hostfile 驱动）。
# 结构参考 Megatron-LM-das/examples/GLM/run_glm5_300B.sh 与 examples/glm4.5/run_glm.sh。
#
# 用法:
#   bash run_multinode.sh <hostfile> <node_num>
# 示例（双机 16 卡）:
#   bash run_multinode.sh hostfile.example 2
# 示例（16 节点 128 卡）:
#   bash run_multinode.sh hostfile.128cards 16
#
# hostfile 每行一个节点（与 das 仓库一致的纯主机名列表，第一行是主节点）。
# 环境变量:
#   LAUNCHER=mpirun|torchrun   （默认 mpirun，容器内 SSH 端口用 MPI_PORT）
#   HF_MODEL_PATH=...          GLM-4.5-Air HF 权重目录
#   DATA_PATH=...              Megatron indexed dataset 前缀
#   TOKENIZER_MODEL_PATH=...   tokenizer 目录（可复用 HF_MODEL_PATH）
#   CHECKPOINT_PATH=...        ckpt 目录

hostfile_input=${1}
node_num=${2}

CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
MEGATRON_PATH=$(dirname "$(dirname "${CURRENT_DIR}")")/Megatron-LM-das

# Those variables need to modify
DTK_ENV="/opt/dtk/env.sh"                                                    # where env.sh of dtk
DATA_PATH=${DATA_PATH:-"./mycorpus_text_document"}                           # Megatron indexed 数据前缀
TOKENIZER_MODEL_PATH=${TOKENIZER_MODEL_PATH:-${HF_MODEL_PATH:-""}}           # tokenizer 路径
HF_MODEL_PATH=${HF_MODEL_PATH:-""}                                           # GLM-4.5-Air HF 权重目录
LAUNCHER=${LAUNCHER:-"mpirun"}                                               # mpirun or torchrun
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"./ckpt_glm45_air"}                       # ckpt 目录
TRAIN_SCRIPT=${TRAIN_SCRIPT:-train_glm45_air.sh}                             # per-node 训练脚本
NCCL_ENV=${MEGATRON_PATH}/requirements/env.sh                                # RCCL/IB 环境变量
LAUNCH_WITH_BINDING=${MEGATRON_PATH}/requirements/launch_with_binding.sh     # 海光 CPU 绑核
MPI_PORT=${MPI_PORT:-11451}

if [[ "${LAUNCHER}" != "mpirun" && "${LAUNCHER}" != "torchrun" ]]; then
    echo "Only mpirun and torchrun are supported as launch methods"
    exit 1
fi

HOSTFILE="${hostfile_input}_slots"
rm -f "${HOSTFILE}"
if [[ "${LAUNCHER}" == "mpirun" ]]; then
    sed -n "1,${node_num}p" "${hostfile_input}" | sed 's/$/ slots=8/' > "${HOSTFILE}"
else
    sed -n "1,${node_num}p" "${hostfile_input}" | sed 's/$/ slots=1/' > "${HOSTFILE}"
fi

HOST="$(sed -n "1p" "${HOSTFILE}" | awk -F ' ' '{print $1}')"

if [[ "${LAUNCHER}" == "mpirun" ]]; then
    MPIRUN_NP=$((node_num * 8))
    PORT=${PORT:-25900}
else
    MPIRUN_NP=${node_num}
    GPUS_PER_NODE=${GPUS_PER_NODE:-8}
    MASTER_ADDR=${MASTER_ADDR:-${HOST}}
    MASTER_PORT=${MASTER_PORT:-25900}
fi

torchrun_args=""
if [[ "${LAUNCHER}" == "torchrun" ]]; then
    torchrun_args+="
        -x PATH \
        -x LD_LIBRARY_PATH \
        -x PYTHONPATH \
        -x MASTER_ADDR=${MASTER_ADDR} \
        -x MASTER_PORT=${MASTER_PORT} \
        -x NNODES=${node_num} \
        -x GPUS_PER_NODE=${GPUS_PER_NODE} \
    "
fi

source "${NCCL_ENV}"

CMD="mpirun -np ${MPIRUN_NP} --hostfile ${HOSTFILE} \
    --allow-run-as-root \
    --bind-to none \
    --mca plm_rsh_no_tree_spawn 1 \
    --mca plm_rsh_args '-p ${MPI_PORT}' \
    ${torchrun_args} \
    bash -c '
    source ${DTK_ENV} && \
    source ${NCCL_ENV} && \
    cd ${CURRENT_DIR} && \
    bash ${TRAIN_SCRIPT} \
    ${HOST} \
    ${PORT:-${MASTER_PORT}} \
    --data_path=${DATA_PATH} \
    --launch_backend=${LAUNCHER} \
    --tokenizer_path=${TOKENIZER_MODEL_PATH} \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --hf_model_path=${HF_MODEL_PATH} \
    --launch_with_binding=${LAUNCH_WITH_BINDING}' 2>&1 | tee log-glm45air-${node_num}nodes-$(date +%F-%H%M).log
"

eval ${CMD}
wait
