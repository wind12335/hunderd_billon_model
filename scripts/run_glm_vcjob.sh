#!/usr/bin/env bash
# =============================================================================
# GLM-4.5-Air 微调 —— SCNet 模型训练模块（vcjob）专用入口
#
# 平台用法：把本文件路径填进作业的"启动脚本"栏，无需任何参数/环境变量。
#   平台会在每个 worker 上以 /bin/sh 各执行一次本文件；协商机制与
#   run_dual_node_test.sh 相同（worker-0 体检→生成方案→发布地址→起训，
#   worker-1..N-1 等地址标记→起训），跨节点自动启用 shca RDMA。
#
# 排练/正式切换：只改下面 PROFILE 区的行（共享存储上改完，重新提交即生效）。
# 当前配置 = 64 卡 20 步排练（现阶段为随机初始化+mock 数据，验证多 worker
# 编排与 GLM MoE 前向/反向链路；权重转换与真数据接入后同一入口即为真排练）。
# =============================================================================
# ── PROFILE（按作业改这里）───────────────────────────────────────────────────
export NUM_NODES=8               # worker 总数：64 卡=8，128 卡=16
export MODEL_PATH=models/glm45_air_base
export SEQ_LEN=2048
export TRAIN_ITERS=20            # 排练 20~50；正式 1250
export RUN_TAG=glm64_rehearsal   # 生成目录名；128 卡正式改 glm128_formal
export SEARCH_MAX_PP=2           # GLM 需要的规划搜索上限，一般不动
export SEARCH_MAX_EP=8
# ── 引擎：通用多机脚本（绝对路径，防平台把启动脚本 staging 到别处后失效）────
exec bash /root/private_data/hunderd_billon_model/scripts/run_dual_node_test.sh "$@"
