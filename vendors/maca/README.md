# autoparallel —— 沐曦(MetaX/MACA)移植版

海光(hygon)版 autoparallel 的沐曦移植。**沐曦上跑这个包;海光上继续用老的 autoparallel。**

## 目录结构

```
autoparallel_maca/
├── autoparallel/        # autoparallel 核心包
├── megatron_mvp/        # 并行方案搜索(mvp)
├── scripts/             # 环境激活 / 启动脚本(仅沐曦)
│   ├── activate_megatron_maca.sh   # 一键激活环境(相对路径,解压后任意位置可用)
│   └── run_megatron_maca.sh
├── tools/               # 运行包装器 / 自检(仅沐曦)
│   ├── run_megatron_maca.py        # 沐曦版 Megatron 运行包装器
│   ├── megatron_maca_bootstrap.py  # bootstrap:把沐曦 Megatron-LM 挂进 import 链路
│   ├── check_megatron_maca_env.py  # 环境自检
│   └── run_pytorch_llama_tp_dp.py  # 纯 torch TP/DP 训练脚本
├── Megatron-LM/         # 沐曦官方 Megatron-LM 0.12.0 源码(maca-megatron-lm-3.2.1.5)
└── generated/           # 生成的启动配置 / 权重转换脚本(llama2、qwen3)
```

## 前置依赖(需自行安装,不在包内)

1. conda 环境 `mxc`:沐曦 MetaX torch 2.6.0+metax3.1.0.4(含 cu-bridge)等整套软件栈
2. MACA 运行时:`/opt/maca`(含 mxgpu_llvm、tools/cu-bridge)
3. 模型权重、数据集、训练 checkpoint:按需放置

## 使用

```bash
cd autoparallel_maca
source scripts/activate_megatron_maca.sh      # 激活环境(conda mxc + MACA 环境变量 + PYTHONPATH)
python3 tools/check_megatron_maca_env.py      # 自检(可选)
```

跑 Megatron 训练(以 pretrain_gpt 为例,参数透传):

```bash
source scripts/activate_megatron_maca.sh
python3 -m torch.distributed.run --nproc_per_node 8 \
    tools/run_megatron_maca.py Megatron-LM/pretrain_gpt.py \
    --use-mcore-models --transformer-impl local ... \
    --data-path <DATA> --tokenizer-type Llama2Tokenizer --tokenizer-model <MODEL>
```

或直接使用 autoparallel 生成启动脚本(参考 `generated/` 里的实际配置)。

## 与海光版差异

| 项 | 海光(hygon,老) | 沐曦(maca,本包) |
|----|----------------|------------------|
| bootstrap | `megatron_hygon_bootstrap.py`(拼接 dcu_megatron 补丁) | `megatron_maca_bootstrap.py`(直接用沐曦 Megatron-LM 源码) |
| 运行包装器 | `run_megatron_hygon.py` | `run_megatron_maca.py`(剔除 rank/world-size/dist-url,用 torchrun 环境变量) |
| 环境变量 | DTK | MACA_PATH / MACA_CLANG / cu-bridge 等 |
| Megatron | dcu_megatron whl | maca-megatron-lm-0.12.0 源码 |

注意:沐曦 0.12 是官方参数体系,分布式信息由 torchrun 通过 RANK/WORLD_SIZE/LOCAL_RANK 注入,不再支持 `--rank/--world-size/--dist-url` 旧参数。
