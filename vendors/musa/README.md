# autoparallel —— 摩尔线程(Moore Threads / MUSA)适配版

海光(hygon)版 autoparallel 的摩尔线程 MUSA 移植，已完成 **colossal-llama2-13B 在 8× MTT S5000 上的全量微调验证**（TP=4/DP=2, 1 epoch, 690M tokens, lm loss 12.49→2.34）。**本包在摩尔线程 MUSA 上运行；海光/沐曦继续用各自的移植版。**

---

## 1. 目录结构

```
autoparallel_musa/
├── README.md              # 本文档
├── autoparallel/          # autoparallel 核心包（声明式并行框架）
├── megatron_mvp/          # 并行方案搜索(mvp)
├── musa_patch/            # 摩尔线程官方 megatron-lm-musa-patch + autoparallel 定制
├── te_stub/               # TransformerEngine 屏蔽 stub（见 §6.2）
├── scripts/               # 环境激活 / 启动脚本
│   ├── activate_megatron_musa.sh              # 一键激活环境（相对路径，解压后任意位置可用）
│   ├── run_megatron_musa.sh                   # 便捷启动（转发到 tools/run_megatron_musa.py）
│   └── run_autoparallel_dual_gpu_small_train.sh
├── tools/                 # 运行包装器 / bootstrap / 转换 / 自检 / loss 工具
│   ├── megatron_musa_bootstrap.py   # bootstrap：te_stub + 源码补丁(7处) + musa_patch
│   ├── run_megatron_musa.py         # MUSA 版 Megatron 运行包装器（参数透传）
│   ├── check_megatron_musa_env.py   # 环境自检
│   ├── convert_hf_to_mcore.py       # HF 权重 → mcore checkpoint（含 TP 分片）
│   ├── preprocess_musa.py           # 数据预处理包装器（屏蔽 TE 后跑 preprocess_data.py）
│   ├── prepare_infinity_sft.py      # Infinity-Instruct parquet → jsonl
│   ├── log_to_csv.py                # 训练日志 → loss CSV
│   ├── plot_loss_csv.py             # CSV → 曲线 PNG
│   └── verify_*.py / debug_*.py     # 验证与排障脚本
├── models/                # 冒烟用微型模型（tiny_llama_smoke）
├── generated/             # 生成的实际运行配置（不含权重/数据/checkpoint）
│   ├── run_13b/launch_ft8.sh        # 13B TP=4 全量微调启动脚本（核心）
│   └── e2e_report/                  # e2e 报告配置示例
├── Megatron-LM/           # 摩尔线程适配版 Megatron-LM 源码（干净版，bootstrap 运行时打补丁）
│                          #   git 版本: NVLM-1.0-1368-gea57327f
└── docker/                # 宿主机侧一键容器脚本（原始路径，复用需按 §4 调整）
    ├── run_ft8_full.sh              # 8卡全量微调 / RESUME 续训入口
    ├── run_ft8_smoke.sh             # 冒烟
    ├── run_ft8_resume_smoke.sh      # 续训验证
    └── run_loss_report.sh           # 日志→CSV→曲线 一步生成
```

---

## 2. Docker 环境（本文档验证环境）

本包的所有运行路径都基于下面的容器与宿主机环境。**复用本包时请先核对这里的环境。**

### 2.1 宿主机

| 项 | 值 |
|----|----|
| GPU | 8× Moore Threads MTT S5000（80GB HBM / 卡） |
| 驱动 / 运行时 | MUSA SDK 4.3.5（`libmusa.so.4.3.5`，映射进容器为 `.so.1`） |
| CPU | 海光 C86-4G（**无 AVX-512**，这是 te_stub 存在的原因，见 §6.2） |
| 目录挂载 | 宿主 `/root/workspace` ↔ 容器 `/ws` |

### 2.2 容器镜像

```
sh-harbor.mthreads.com/sets/sqt-vllm-train-musa-bench:musa-sdk-4.3.5_torch2.7.1-intel
```

| 项 | 值 |
|----|----|
| torch | 2.7.1（MUSA backend 以 `privateuse1` 挂载，torch.cuda→torch.musa 由 musa_patch 映射） |
| torch_musa | 2.7.1 |
| MCCL | 2.11.4（摩尔线程集合通信库） |
| Megatron-LM | `/home/Megatron-LM`（干净源码，本包 bootstrap 运行时打补丁；也可用包内 `Megatron-LM/`） |

### 2.3 容器启动参数（关键）

```bash
docker run --rm \
    --network host \
    --shm-size=64g \
    --device=/dev/mtgpu.0 --device=/dev/mtgpu.1 ... --device=/dev/mtgpu.7 \
    -v /usr/lib64/libmusa.so.4.3.5:/lib/x86_64-linux-gnu/libmusa.so.1 \
    -v /root/workspace:/ws \
    sh-harbor.mthreads.com/sets/sqt-vllm-train-musa-bench:musa-sdk-4.3.5_torch2.7.1-intel \
    bash -c "bash /ws/musa_framework/.../launch_ft8.sh 2>&1 | tee /ws/ft8_full.log"
```

- `--network host`：MCCL 走共享内存/主机网络，8 卡通信必须
- `--shm-size=64g`：数据加载 / 集合通信共享内存充足
- `--device=/dev/mtgpu.0..7`：8 卡透传
- `libmusa.so` 映射：容器内 `/lib/x86_64-linux-gnu/libmusa.so.1` 指向宿主 4.3.5 版本
- `/root/workspace` ↔ `/ws`：本包解压后放在 `/ws/musa_framework`（即包根目录）

> 本包内路径约定：`/ws/musa_framework` = 包根目录，`/ws/model` = HF 模型权重，`/ws/dataset` = 数据集，`/ws/ft8_full.log` = 训练日志。

---

## 3. 前置依赖（不在包内，需自行准备）

| 依赖 | 说明 | 位置示例 |
|------|------|----------|
| Docker + MUSA 驱动/SDK 4.3.5 | 见 §2 | 宿主 |
| 容器镜像 | `sh-harbor...:musa-sdk-4.3.5_torch2.7.1-intel` | 需 pull 到宿主 |
| 8× MTT S5000 | 80GB/卡 | 透传 |
| HF 模型权重 | colossal-llama2-13B（`69104` vocab，40 层，hidden 5120） | `/ws/model` |
| 数据集 | Infinity-Instruct-7M-core（parquet） | `/ws/dataset/7M_core` |
| 训练数据/权重产物 | Megatron 格式 bin/idx + TP=4 分片 ckpt | 见 §5 生成 |

---

## 4. 使用与再配置

### 4.1 激活环境（容器内）

```bash
cd /ws/musa_framework
source scripts/activate_megatron_musa.sh   # 设置 PYTHONPATH / MUSA 环境变量 / MEGATRON_UPSTREAM_ROOT
python3 tools/check_megatron_musa_env.py   # 自检（可选）
```

激活脚本自动完成：
- `MEGATRON_UPSTREAM_ROOT`：优先容器内 `/home/Megatron-LM`，否则包内 `Megatron-LM/`
- PYTHONPATH 顺序：`tools : Megatron-LM : te_stub : 包根`（te_stub 必须在 Megatron 之前，才能屏蔽 TE）
- MUSA 环境变量：`ACCELERATOR_BACKEND=musa`、`MUSA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`、`MUSA_KERNEL_TIMEOUT`、`MCCL_PROTOS=2`、`MCCL_CHECK_POINTERS=0`
- `TORCHDYNAMO_DISABLE=1`、`NVTE_TORCH_COMPILE=0`（MUSA 下 torch.compile 的 dynamo 追踪会崩）
- 固定 `CC/CXX` 为系统 gcc（避免 Megatron 编译 C++ 扩展用错工具链）

### 4.2 快速冒烟（验证 8 卡链路）

```bash
# 容器内，激活后
TRAIN_ITERS=3 GLOBAL_BATCH_SIZE=16 \
python3 -m torch.distributed.run --nproc_per_node 8 \
    tools/run_megatron_musa.py Megatron-LM/pretrain_gpt.py \
    --use-mcore-models --transformer-impl local --num-layers 40 --hidden-size 5120 \
    --ffn-hidden-size 13824 --num-attention-heads 40 --num-query-groups 40 \
    --seq-length 2048 --bf16 --tensor-model-parallel-size 4 --micro-batch-size 1 \
    --global-batch-size 16 --train-iters 3 --lr-warmup-iters 1 \
    --tokenizer-type HuggingFaceTokenizer --tokenizer-model /ws/model \
    --data-path /ws/dataset/infinity_7M_core_text_document --split 949,50,1
```

或直接用包内 `generated/run_13b/launch_ft8.sh`（TRAIN_ITERS 环境变量覆盖为小值）。

### 4.3 全量微调 colossal-llama2-13B

宿主机一键脚本（`docker/run_ft8_full.sh`，原始路径为 `/root/workspace`，复用请改）：

```bash
# 从头微调
bash run_ft8_full.sh
# 断点续训（从最新 checkpoint）
RESUME=1 bash run_ft8_full.sh
```

等价地在容器内直接跑：

```bash
cd /ws/musa_framework && source scripts/activate_megatron_musa.sh
bash generated/run_13b/launch_ft8.sh           # 默认 TRAIN_ITERS=1250
```

关键运行参数（launch_ft8.sh 内，均可环境变量覆盖）：

| 参数 | 值 | 说明 |
|------|----|------|
| `--tensor-model-parallel-size` | 4 | TP=4（DP 自动推导 = 8/4 = 2） |
| `--global-batch-size` | 256 | 全局 batch（grad_accum = 256/(2×1) = 128） |
| `--micro-batch-size` | 1 | |
| `--bf16` | | 混合精度（用户指定，不绕开官方 bf16） |
| `--use-distributed-optimizer` | | 优化器状态 8 卡分片 |
| `--recompute-granularity full --recompute-method uniform --recompute-num-layers 40` | | 全重算，压峰值显存 |
| `--save-interval` | 50 | 可 `SAVE_INTERVAL=200` 覆盖（§7 滚动保留） |
| `--lr 1e-5 --min-lr 1e-6 --lr-decay-style cosine --lr-warmup-iters 200` | | 余弦 + 200 步 warmup |

### 4.4 日志与 loss 曲线

```bash
tail -f /ws/ft8_full.log                      # 实时日志（训练用 tee 写）
bash docker/run_loss_report.sh                # 一步: 日志→CSV→曲线 PNG
# 产物: /ws/ft8_loss.csv (iteration,lm_loss)  +  /ws/loss_curve.png
# 手动:
#   python3 tools/log_to_csv.py  --log /ws/ft8_full.log --csv /ws/ft8_loss.csv
#   python3 tools/plot_loss_csv.py --csv /ws/ft8_loss.csv --out /ws/loss_curve.png
```

### 4.5 再配置要点（换机器/换数据/换模型）

1. **换机器**：核对 §2 的镜像 / libmusa.so 版本 / GPU 数量；8 卡 → 其他卡数需同步改 `TP`、`DP` 与 `--save-interval` 等。
2. **换数据**：`tools/prepare_infinity_sft.py` 把 parquet 转 jsonl → `tools/preprocess_musa.py` 转 Megatron bin/idx → 改 `--data-path` 前缀。
3. **换模型**：用 `tools/convert_hf_to_mcore.py` 重新转换权重（见 §5.2），并按新模型改 `--num-layers/--hidden-size/...`。
4. **续训**：`RESUME=1` 会自动从 `--save` 目录最新 checkpoint 续。**注意**：本版本 scheduler 把 warmup/decay 按 `iters×gbs` 换算并校验 checkpoint 一致性，续训时 `TRAIN_ITERS / LR_WARMUP_ITERS` 必须与原训练一致（见 §6.5）。

---

## 5. 复现本文档的 13B 微调（端到端）

### 5.1 数据准备

```bash
# 1) parquet → jsonl（Colossal-LLaMA 风格对话文档，<s>Human:...Assistant:...</s>）
python3 tools/prepare_infinity_sft.py --input-dir /ws/dataset/7M_core \
    --output /ws/dataset/infinity_7M_core.jsonl

# 2) jsonl → Megatron bin/idx（包装器先 bootstrap 屏蔽 TE 再跑 preprocess_data.py）
python3 tools/preprocess_musa.py /home/Megatron-LM/tools/preprocess_data.py \
    --input /ws/dataset/infinity_7M_core.jsonl \
    --output-prefix /ws/dataset/infinity_7M_core \
    --tokenizer-type LlamaTokenizer --vocab-file /ws/model/tokenizer.model \
    --append-eod --workers 32
```

### 5.2 权重转换（HF → mcore，TP=4 分片）

```bash
python3 tools/convert_hf_to_mcore.py \
    --model-dir /ws/model \
    --output-dir /ws/musa_framework/generated/ckpt_13b_tp4 \
    --tensor-model-parallel-size 4 \
    --num-layers 40 --num-heads 40
```

输出 `iter_0000001/mp_rank_00..03/model_optim_rng.pt`（每 rank 6.7GB，含正确 TP=4 分片：vocab 17276、qkv 按 head-group 3840、fc1 6912、fc2 3456）。

### 5.3 启动训练

```bash
RESUME=0 bash docker/run_ft8_full.sh    # 或容器内 bash generated/run_13b/launch_ft8.sh
```

**实测结果**（本文档验证）：

| 项 | 值 |
|----|----|
| 显存 | 38.4GB allocated / 45GB max / 80GB 卡 |
| 每步耗时 | ~66s（mbs=1, full recompute, seq=2048） |
| 吞吐 | ~7450 tokens/s（8 卡合计） |
| 总时长 | 1250 iters ≈ 23 小时 |
| loss | 12.49 → **2.34**（1 epoch，全程单调下降，无 NaN） |
| checkpoint | 每 50 步保存 + 末步，滚动保留 3 个 |

---

## 6. 关键适配点（迁移做了什么）

### 6.1 `musa_patch/` —— 官方补丁 + 定制

`musa_patch` 源自摩尔线程官方 `megatron-lm-musa-patch`，做两件事：
- `patch_after_import_torch`：把 `torch.cuda.*` 映射到 `torch.musa.*`（MUSA backend 是 `privateuse1`，不走 CUDA）
- `patch_before_import_megatron`：给 megatron 内部打 MUSA 适配补丁

本仓库在此基础上定制了 `arguments.py` / `training.py` 等（bootstrap 在 `import musa_patch` 之前先完成源码补丁）。

### 6.2 `te_stub/` —— 屏蔽 MT-TransformerEngine

MT-TransformerEngine 的 `.so` 用 **AVX-512** 编译，而本机 CPU（海光 C86-4G）**无 AVX-512**，`import transformer_engine` 即 SIGILL（硬崩，try/except 捕不到）。`te_stub` 是一个假 `transformer_engine` 包，置于 PYTHONPATH 最前，让 `import transformer_engine` 抛 `ModuleNotFoundError` → Megatron 全局 `HAVE_TE=False`，回退到本地实现。

### 6.3 `megatron_musa_bootstrap.py` —— 幂等源码补丁（7 处）

bootstrap 在 `import musa_patch` 前，对上游 Megatron 源码做**按 patch 粒度幂等**的修改（8 rank 并发有文件锁，不重复打）：

| # | 文件 | 作用 |
|---|------|------|
| 1 | `core/pipeline_parallel/schedules.py` | 屏蔽 TE 顶层硬导入 |
| 2 | `training/training.py` | 模型分配 `.cuda()` → `.to(torch.device('musa'))`（MUSA 不识别 cuda） |
| 3 | `core/transformer/transformer_block.py` | TE 不可用时给 `get_fine_grained_offload_handler` no-op 兜底 |
| 4 | `core/jit.py` | `torch.compile` 退化为普通函数（dynamo 在 MUSA 下崩） |
| 5 | `training/checkpointing.py` | `torch.load` 到 CPU 再拷 GPU（防加载期显存超限） |
| 6 | `training/checkpointing.py` | 滚动保留最近 3 个持久 checkpoint |
| 7 | `core/optimizer/distrib_optimizer.py` | 修 dp_zero 续训 `KeyError:'step'`（跳过 tensors 缺失键） |

### 6.4 续训机制

`RESUME=1` 时 `launch_ft8.sh` 变为：`--load ${SAVE_DIR}`（最新 checkpoint）+ **去掉** `--finetune --no-load-optim --no-load-rng`。日志用 `tee -a` 追加，曲线可接着画。

### 6.5 续训已知约束

- **scheduler 参数必须与原训练一致**：本版本 scheduler 把 warmup/decay 按 `iters × gbs` 换算（`optimizer_param_scheduler.py`），加载时 `_check_and_set` 断言 class 值与 checkpoint 值一致。续训若改 `TRAIN_ITERS` 或 `LR_WARMUP_ITERS`（如为了快速验证而压缩），会报 `AssertionError: class input value ... and checkpoint value ... do not match`。
- **不要重复启动**：`MASTER_PORT=29500` 被正在运行的训练占用时，二次启动报 `EADDRINUSE`。这是"训练已经在跑"的正常信号，不是故障。

---

## 7. Checkpoint 策略

- 保存：每 `--save-interval`（默认 50，可 `SAVE_INTERVAL=200` 覆盖）步 + 末步 1250 额外保存
- 滚动保留：**最近 3 个** `iter_%07d` 目录（bootstrap 补丁 #6，仅 rank0 删除，日志可见）
- 目录：`generated/run_13b/checkpoints/`（每目录 ~175GB：bf16 模型 + fp32 优化器主参/动量/方差，TP=4 分片）
- 断点续训：`RESUME=1` 自动定位 `latest_checkpointed_iteration.txt`

---

## 8. 与海光 / 沐曦版差异

| 项 | 海光(hygon) | 沐曦(maca) | 摩尔线程(musa, 本包) |
|----|-------------|------------|----------------------|
| bootstrap | `megatron_hygon_bootstrap.py`（拼接 dcu_megatron 补丁） | `megatron_maca_bootstrap.py` | `megatron_musa_bootstrap.py`（te_stub + 7 处幂等补丁 + musa_patch） |
| 运行包装器 | `run_megatron_hygon.py` | `run_megatron_maca.py` | `run_megatron_musa.py` |
| 环境 | DTK | MACA_PATH / cu-bridge | MUSA SDK / `torch_musa` / MCCL |
| Megatron | dcu_megatron whl | maca-megatron-lm-0.12.0 源码 | 摩尔线程适配版（`NVLM-1.0-1368-gea57327f`） |
| 主要障碍 | — | — | TE .so AVX-512 SIGILL（te_stub）、dynamo 崩溃（jit 退化）、dp_zero 续训 bug（补丁 #7） |

---

## 9. 已知限制与注意事项

1. **CPU 无 AVX-512 时必须用 te_stub**；若宿主机 CPU 支持 AVX-512，可去掉 te_stub 并安装真实 MT-TransformerEngine。
2. `torch.compile`/dynamo 在 MUSA 下不可用（`TORCHDYNAMO_DISABLE=1` 已默认）。
3. TP=1/DP=8 时 13B 单卡峰值 ~76GB/80GB 会 OOM；本包默认 **TP=4/DP=2**（~45GB 峰值）更稳。
4. checkpoint 加载必须走 CPU（补丁 #5），不要在 GPU 上 `torch.load` 大权重。
5. `docker/` 下脚本为原始宿主机路径（`/root/workspace`），复用按 §4.5 调整。
