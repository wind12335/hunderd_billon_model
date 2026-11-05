# autoparallel_corex — 自研自动并行框架 · 天数智芯天垓150(BI-V150)版

由摩尔线程版(autoparallel_musa)适配而来的天数智芯 CoreX 迁移，是继海光 DCU、
沐曦 MACA、摩尔线程 MUSA 之后的第四个国产卡迁移目标。

- 硬件：8× Iluvatar BI-V150（天垓150，32GB，ivcore11 架构；本机共 16 张，8 卡训练默认取前 8 张）
- 软件栈：CoreX 4.4.0（CUDA 兼容层，驱动报告 CUDA 10.2）
- PyTorch：torch 2.7.1+corex.4.4.0（`torch.cuda.*` 原生可用）
- Megatron：上游干净源码 core_v0.14.0rc2（`Megatron-LM/`，无厂商魔改）

## 一、与 MUSA 版的差异（为什么 CoreX 版简单得多）

| 迁移项 | MUSA(摩尔线程) | CoreX(天数智芯) |
|---|---|---|
| 设备访问 | musa_patch 整层 torch.cuda→torch.musa 垫片（~2500 行） | **无需任何垫片**，CUDA 兼容层原生 |
| NCCL | 重映射 nccl→mccl | **libnccl 原生**（2 卡 all_reduce 实测通过） |
| apex | 屏蔽 amp_C/FusedAdam（海光 CPU 无 AVX-512 会 SIGILL） | **全量启用**（corex 编译版，Xeon 6330 有 AVX-512） |
| TransformerEngine | te_stub + 2 处源码补丁兜底 | te_stub 即可（上游 0.14.0rc2 的 TE 导入全部有保护） |
| Megatron 源码 | 厂商改版 + 7 处运行时补丁 | 上游干净版 + **4 处**补丁（见下） |
| optimizer | 强制 torch AdamW | **apex FusedAdam**（自动选择，AdamW 等价实现） |
| flash_attn | 不可用 | **可用**（2.6.3+corex，bf16 实测通过） |

### bootstrap 源码补丁清单（tools/megatron_corex_bootstrap.py，幂等 + 文件锁）

1. `fusions/fused_layer_norm.py`：FusedLayerNorm.__new__ 按 normalization 分发 ——
   上游装了 apex 就选 FusedLayerNorm（只支持 LayerNorm），RMSNorm(llama2) 直接崩；
   补丁让 RMSNorm 回退 torch 原生 RMSNorm（torch≥2.4）。
2. `core/optimizer/distrib_optimizer.py`：dp_zero 断点续训 KeyError:'step' 修复
   （上游 bug，跳过 tensors 缺失键）。
3. `core/jit.py`：jit_fuser 退化为普通函数（torch.compile/dynamo 在非 NVIDIA
   后端追踪兼容性未验证，首发稳态）。
4. `training/checkpointing.py`：滚动保留最近 N 个持久 checkpoint
   （环境变量 `COREX_KEEP_LAST_CKPT`，默认 3；仅 rank0 删除）。

运行时猴子补丁：`legacy/fused_kernels.load` 尽力加载，失败降级 no-op
（corex 的 CUDA_HOME JIT 编译不保证成功；本配方未启用 legacy 融合核特性）。

### 启动参数层面的适配（非源码补丁）

- `--no-rope-fusion`：TE 被 stub 后融合 RoPE 不可用，走纯 torch RoPE
  （corex 版 rotary_emb 只导出 `apply_rotary`，缺 `apply_rotary_emb_func`）。
- `--no-persist-layer-norm`：上游默认 persist=True 是为 NVIDIA apex 设计。
- `--transformer-impl local` + te_stub：无天数版 TE，走 local 实现。

## 二、目录结构

```
autoparallel_corex/
├── autoparallel/          # 声明式自动并行核心（平台无关，lowering.py 生成 corex 启动脚本）
├── megatron_mvp/          # 搜索/检测/渲染层（environment.py 增加 corex 探测）
├── Megatron-LM/           # 上游 core_v0.14.0rc2 干净源码
├── te_stub/               # TransformerEngine 屏蔽桩
├── tools/
│   ├── megatron_corex_bootstrap.py   # 环境引导（te_stub + 4 补丁 + 猴子补丁）
│   ├── run_megatron_corex.py         # Megatron 运行包装器
│   ├── check_megatron_corex_env.py   # 环境自检（apex/flash_attn/NCCL/TE 全验证）
│   ├── convert_hf_to_mcore.py        # HF colossal-llama2-13B → mcore TP=4 分片
│   ├── prepare_infinity_sft.py       # Infinity-Instruct-7M parquet → Megatron jsonl
│   ├── log_to_csv.py                 # 训练日志 → loss+lr CSV（比 MUSA 版多 lr 列）
│   ├── plot_loss_csv.py              # CSV → loss/lr 双曲线 PNG
│   ├── dump_mcore_state_dict.py / verify_ckpt_sanity.py   # 权重转换核对
│   └── run_autoparallel_e2e_report.py  # 声明式前端 E2E 汇报脚本
├── scripts/
│   ├── activate_megatron_corex.sh    # 环境激活（PYTHONPATH/CUDA_VISIBLE_DEVICES）
│   └── run_megatron_corex.sh         # 统一运行入口
├── generated/run_13b/launch_ft8.sh   # 8 卡微调启动脚本（TP=4/DP=2）
└── run_ft8_full.sh                   # 全量微调节奏脚本（日志 tee 到 /ws/ft8_full.log）
```

## 三、微调配方（来源：/private/国产卡百亿参数微调.pdf，与 MUSA 版口径一致）

- 模型：colossal-llama2-13B（40 层 / hidden 5120 / ffn 13824 / 40 heads MHA /
  vocab 69104 / rope base 10000 / RMSNorm / swiglu / untied）
- 数据：BAAI/Infinity-Instruct-7M-Core ≈140 万条 ≈15 亿 tokens，单 epoch
- 并行：TP=4, DP=2, PP=1（与 MUSA 版一致，保证跨平台 loss 曲线可比）
- batch：micro-batch 1 × 全局 256（DP=2 时 grad-accum=128 microbatch/步）
- 精度：bf16 + 全激活重计算(40 层) + DistributedOptimizer + grad-reduce-in-bf16
- 优化器：apex FusedAdam（AdamW 等价），lr=1e-5 → min 1e-6 cosine，
  warmup 200 步，weight-decay 0，grad-clip 1.0，β=(0.9, 0.95)
- 步数：**TRAIN_ITERS=1250 ≈ 1 epoch**（注：断点续训/末步日志常见 1249/1250，
  即此前各平台汇报中的"1249 轮"）
- 时长参考（PDF 口径 MFU=30%）：8 卡 ≈ 70h / 2.9 天；实际以日志
  `elapsed time per iteration (ms)` 为准

### 显存测算（32GB 卡，为何 TP=4 刚好能放下）

| 项 | 大小/卡 |
|---|---|
| bf16 参数（26GB ÷ TP4） | 6.5 GB |
| fp32 master + Adam m/v（13B×12B ÷ 8 rank） | 19.5 GB |
| bf16 梯度缓冲（26GB ÷ 8） | 3.25 GB |
| CUDA context + workspace + 激活（full recompute） | ≈2 GB |
| **合计** | **≈31.7 / 34.4 GB** |

余量 ~2.7GB，已加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 抗碎片。
若仍 OOM：`TP_SIZE=8 bash generated/run_13b/launch_ft8.sh`（TP=8/DP=1，
≈28.5GB/卡；vocab 69104÷8、40 heads÷8 均整除；注意需先转换 TP=8 基座权重）。

## 四、完整执行流程（作业机上）

```bash
# 0) 环境自检（每次开作业先跑）
python3 tools/check_megatron_corex_env.py
#    期望全部通过：corex/apex/flash_attn/nccl/te_stub
#    若报 pybind11 缺失：pip3 install pybind11（数据集 C++ 索引器编译需要）

# 1) 数据准备（Infinity-Instruct-7M-Core parquet → jsonl → mmap）
#    1.1 parquet → jsonl（Colossal-LLaMA 对话格式，全序列 next-token loss）
python3 tools/prepare_infinity_sft.py \
    --input-dir /ws/dataset/7M_core \
    --output /ws/dataset/infinity_7M_core.jsonl
#    1.2 jsonl → Megatron mmap（用 Megatron 预处理工具，tokenizer=/ws/model）
python3 Megatron-LM/tools/preprocess_data.py \
    --input /ws/dataset/infinity_7M_core.jsonl \
    --output-prefix /ws/dataset/infinity_7M_core_text_document \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model /ws/model \
    --workers 16 --append-eod

# 2) 基座权重转换（HF colossal-llama2-13B → mcore TP=4）
python3 tools/convert_hf_to_mcore.py \
    --model-dir /ws/model \
    --output-dir generated/ckpt_13b_tp4 \
    --tensor-model-parallel-size 4
#    （转换正确性可用 tools/verify_ckpt_sanity.py 复核）

# 3) 8 卡全量微调（1250 步 ≈ 1 epoch；日志实时 tee 到 /ws/ft8_full.log）
bash run_ft8_full.sh
#    短冒烟（3 步验证链路）: TRAIN_ITERS=3 bash generated/run_13b/launch_ft8.sh
#    断点续训:              RESUME=1 bash run_ft8_full.sh
#    （RESUME 时 TRAIN_ITERS/WARMUP 必须保持 1250/200，scheduler 会校验）

# 4) 记录并绘制 loss + learning rate 曲线（训练中/结束后任意时刻可跑，幂等）
python3 tools/log_to_csv.py    --log /ws/ft8_full.log --csv /ws/ft8_loss.csv
python3 tools/plot_loss_csv.py --csv /ws/ft8_loss.csv  --out /ws/loss_lr_curve.png
#    TensorBoard 同步在记录（generated/run_13b/tensorboard/，含 loss/lr 全量曲线）:
#    tensorboard --logdir generated/run_13b/tensorboard --port 6006
```

## 五、本机验证记录（2026-09-22，16×BI-V150 开发容器）

- [x] `check_megatron_corex_env.py` 全项通过（apex/flash_attn/NCCL/te_stub）
- [x] NCCL 2 进程 all_reduce 数值正确（1+2=3）
- [x] 单卡 mock-data 冒烟：3 步收敛行为正常（loss≈9.06 ≈ ln(8192)）
- [x] **8 卡 TP=4/DP=2 冒烟（与正式微调同拓扑）**：DistributedOptimizer +
      bf16 + full recompute + checkpoint 保存全部正常
- [x] log_to_csv / plot_loss_csv 双曲线（loss + lr）端到端出图
- [x] autoparallel 声明式前端 + megatron_mvp 环境检测（accelerator=corex）
- [ ] 全量 1250 步微调：待作业机挂载 /ws/model（HF 权重+tokenizer）与
      /ws/dataset（7M_core parquet）后按第四节执行

## 六、已知边界与排障

1. **/ws 未挂载**：开发容器无 /ws；作业机上按约定挂载（/ws/model、/ws/dataset）。
2. **pybind11**：容器缺该包会导致数据集 C++ 索引器编译失败，`pip3 install pybind11`。
3. **rotary_emb（corex 版）只导出 apply_rotary**：因此必须 `--no-rope-fusion`，
   RoPE 走纯 torch 路径（launch 脚本已带）。
4. **fused_kernels JIT**：bootstrap 已做"尽力加载"，失败自动降级 no-op 并打印
   警告，属预期行为，不影响本配方正确性。
5. **续训校验**：RESUME=1 时 TRAIN_ITERS/LR_WARMUP_ITERS 必须与首次一致
   （1250/200），否则 scheduler AssertionError。
6. **多机扩展**：NNODES/NODE_RANK/MASTER_ADDR 环境变量已参数化，单机 8 卡
   之外可扩多机（PDF 口径 16 卡=2 集群 35h）。
