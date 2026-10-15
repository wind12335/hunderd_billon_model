# 自研自动并行训练框架详解（多机版）

> 适用仓库：`/root/private_data/thousand_model`
> 面向读者：需要接手/扩展该框架的科研同学
> 更新说明：本文档包含两大部分——**第一部分讲解框架原有设计**，**第二部分讲解本次"多机多卡"升级**（搜到 128 卡、PP/EP 并行、多机启动编排），并给出 GLM-4.5-Air 105B 在 128×BW1000 上全量微调的完整路线。

---

## 目录

- [一、一句话定位：这是什么框架](#一一句话定位这是什么框架)
- [二、总体架构：一个"并行编译器"](#二总体架构一个并行编译器)
- [三、目录结构与模块详解](#三目录结构与模块详解)
- [四、核心设计思想](#四核心设计思想)
- [五、两条执行后端](#五两条执行后端)
- [六、三大搜索策略](#六三大搜索策略)
- [七、海光（Hygon）适配链](#七海光hygon适配链)
- [八、多机升级详解（本次改动）](#八多机升级详解本次改动)
- [九、GLM-4.5-Air 128 卡实战路线](#九glm-45-air-128-卡实战路线)
- [十、当前限制与后续工作](#十当前限制与后续工作)

---

## 一、一句话定位：这是什么框架

这不是又一个训练内核，而是一个 **声明式自动并行编译器（auto-parallel compiler）**：

```
用户只写"我要训这个模型、用多少卡"
        │
        ▼
框架自动完成：模型结构探测 → 并行策略搜索(TP/DP/PP/EP) → 显存可行性验证
        │     → 生成 Megatron 配置与启动脚本 → 多机编排拉起真实训练
```

### 与主流框架的差异

| 维度 | Megatron-LM | DeepSpeed | Alpa/Unity 等自动并行 | **本框架** |
|---|---|---|---|---|
| 并行策略 | 手工指定 TP/PP/DP/EP | 手工配置 ZeRO 分片 | 编译器自动切分算子图，生成全新执行计划 | **自动搜索"标准策略域内"的切分方案，下放到 Megatron 执行** |
| 执行内核 | 自带 | 包装 PyTorch | 自研 runtime | **不自建内核**：MEGATRON 后端复用 Megatron-LM；PYTORCH 后端是自带的手写 DP+TP 参考实现 |
| 搜索代价 | 无 | 无 | 编译开销大、通用图切分 | 秒级启发式搜索（BLOCK_LEVEL_FAST 实测 <10ms） |
| 可控性 | 全手工 | 半手工 | 黑盒 | **ParallelPlan 是一等公民**：搜索结果和用户显式指定共用同一数据结构，随时可锁定复现 |
| 海光适配 | 需自己打补丁 | 需适配 | 无 | **原生接入 dcu_megatron 补丁链 + DTK/RCCL 环境** |

**核心卖点**：把"自动并行"的不可控性收敛到一个**可枚举、可校验、可复现的标准策略域**（Megatron 有成熟实现的那部分），搜索层负责在显存约束下选最优组合，执行层 100% 复用工业级内核。这与 Alpa 式"重编译器"路线是刻意的差异化：牺牲理论最优性，换取工程可靠性——对科研/汇报场景非常合适。

---

## 二、总体架构：一个"并行编译器"

### 编译流水线

```
@parallelize(Config(...))                    ← 声明式前端（decorator.py 分发）
        │
        ├─ run_mode=PLAN    → 只输出方案与产物（规划）
        ├─ run_mode=LOWER   → PLAN + 生成可执行脚本（下放）
        └─ run_mode=LAUNCH  → LOWER + 真实启动训练（launcher.py 子进程执行）
        │
        ▼  compiler.compile_plan()
┌─────────────────────────────────────────────────────────────┐
│ 1. detect_hf_model     扫描 HF 模型目录（config.json+权重）   │
│ 2. load_model_spec     解析结构规格（含 MoE 字段）             │
│ 3. detect_runtime_environment  探测 GPU/DCU 数量与显存        │
│ 4. 构造虚拟规划环境      按目标卡数虚拟出 N 卡环境（关键！）    │
│ 5. search_parallel_candidates  搜索 TP/DP/PP/EP 组合          │
│ 6. build_coarse_ir     生成粗中间表示                         │
│ 7. build_megatron_config → Megatron 完整参数                  │
│ 8. build_launch_script → torchrun 启动脚本（多机语义）         │
│ 9. write_generation_outputs  落盘全部产物                     │
└─────────────────────────────────────────────────────────────┘
        │
        ▼  lowering.lower_to_backend()
┌──────────────────────────────────────────────┐
│ MEGATRON 后端: launch_megatron_{plan,smoke}.sh│→ tools/run_megatron_hygon.py
│              + hostfile + launch_cluster.sh  │   → Megatron-LM-das + dcu_megatron
│ PYTORCH 后端:  launch_pytorch_*.sh            │→ tools/run_pytorch_llama_tp_dp.py
│              (纯自研 DP+TP 参考实现)           │   （手写 ColumnParallel/RowParallel…）
└──────────────────────────────────────────────┘
```

### 关键数据流：`ParallelPlan` 贯穿始终

```
搜索候选 SearchCandidate ──┐
                          ├─→ ParallelPlan（用户可见）──→ 粗 IR ──→ Megatron 配置 ──→ 启动脚本
用户显式 ParallelPlan ─────┘
```

`ParallelPlan`（`megatron_mvp/types.py`）是整个框架的"合同"：

```python
ParallelPlan(
    tp=8, dp=8, pp=2, ep=8,          # 四个并行维度（ep 细分 dp，world = tp*dp*pp）
    attention_policy="head_shard",    # 三段式策略（front/mid/back 可分别指定）
    mlp_policy="colrow_shard",
    embedding_policy="vocab_shard",
    lm_head_policy="vocab_shard",
    sequence_parallel=True,
    micro_batch_size=1, global_batch_size=8,
    estimated_memory_bytes=...,       # 搜索时的显存预估
)
```

自动搜索的输出和用户手写的输入是**同一个类型**——你可以先 AUTO 搜一遍，把结果抄下来作为 `explicit_plan` 锁定，保证实验可复现。

---

## 三、目录结构与模块详解

```
thousand_model/
├── autoparallel/                  # 前端：声明式 API + 编译 + 下放
│   ├── config.py                  #   Config 声明式配置（含多机字段）
│   ├── compiler.py                #   compile_plan() 编译主流程
│   ├── lowering.py                #   生成启动脚本/示例代码（多机感知）
│   ├── runtime.py                 #   AutoParallelRuntime 三段式 API
│   ├── decorator.py               #   @parallelize 装饰器
│   ├── launcher.py                #   LAUNCH 模式子进程执行
│   └── types.py                   #   AutoParallelPlan / LoweredProgram
│
├── megatron_mvp/                  # 中端：探测/搜索/渲染
│   ├── detect.py                  #   HF 模型探测（稠密 + MoE）
│   ├── search.py                  #   TP×DP×PP(+EP) 搜索 + 策略注册表
│   ├── ppo_search.py              #   RL_GUIDED_PPO 策略（clipped PPO 控制器）
│   ├── render.py                  #   粗 IR → Megatron 配置 → 启动脚本
│   ├── environment.py             #   GPU/DCU 环境探测
│   └── types.py                   #   全部数据结构（ModelSpec/ParallelPlan/…）
│
├── tools/                         # 执行层
│   ├── run_megatron_hygon.py      #   MEGATRON 后端入口包装（rank 注入）
│   ├── megatron_hygon_bootstrap.py#   上游源码 + dcu_megatron 补丁装载
│   ├── run_pytorch_llama_tp_dp.py #   PYTORCH 后端：手写 DP+TP 运行时（1400 行）
│   ├── launch_cluster.py          #   ★新增：SSH+hostfile 多机一键拉起
│   └── run_autoparallel_e2e_report.py  # 汇报演示脚本（统一双后端 loss 格式）
│
├── scripts/
│   ├── activate_megatron_hygon.sh #   DTK/海光环境激活（多布局兼容）
│   └── glm45_air/                 #   ★新增：GLM-4.5-Air 128 卡参考脚本
│
├── examples/
│   └── autoparallel_multinode_train.py  # ★新增：多机声明式入口
│
├── models/glm45_air_105b_spec/    # ★新增：GLM-4.5-Air 规格 stub（无权重）
└── Megatron-LM-das/               # 海光 hcu-megatron 参考仓库（含 GLM 示例）
```

### 用户可见的三种入口

```python
# 1. 纯声明式（推荐）
@parallelize(Config(model_path=..., num_nodes=16, gpus_per_node=8, ...))
def train(): pass

# 2. 显式方案（可复现）
@parallelize(Config(explicit_plan=ParallelPlan(tp=8, dp=8, pp=2, ep=8, ...)))
def train(): pass

# 3. 直接传 ParallelPlan（最省事）
@parallelize(ParallelPlan(tp=2, dp=2))
def train(): pass
```

---

## 四、核心设计思想

### 4.1 两种设备语义（规划 vs 冒烟）

`Config` 显式区分两个视角，这是本框架最巧妙的设计之一：

| 字段 | 含义 | 用途 |
|---|---|---|
| `search_target_devices` / `num_devices` | **未来目标卡数**（可以是 128，本机不需要真有） | 搜索规划用——框架会"虚拟"出 N 卡环境做显存估算 |
| `runtime_test_devices` | **当前实际可用的卡数** | smoke test 用——卡数不够时自动回退单卡验证链路 |

这让你**今天就能在 1 张卡的开发机上规划 128 卡集群的方案**，产物（`launch_megatron_plan.sh`）拿到集群上直接可用。

### 4.2 显存估算模型（block-level 粗估）

搜索的硬约束来自 `search.py::_estimate_memory`，每卡显存：

```
参数量/卡 = transformer 层参数(÷TP分片 × PP层占比 × EP专家切分)
          + embedding/lm_head (÷vocab分片)
显存 = 参数×2B(bf16权重) + 参数×2B(梯度) + 参数×12B(Adam fp32状态)
     + 激活( ∝ 层数×mbs×seq×hidden×10×0.35重计算 / TP分片, SP再×0.82 )
```

常数含义：`10` 是激活放大系数、`0.35` 是开重计算后的保留比、`0.92` 是可用显存占比。**PP 的引入是 105B 模型可行性的关键**：`layers_per_stage = ceil(L/pp)`，每卡只持有所在 stage 的层。

### 4.3 标准 DP+TP 公共策略域（"可控的自动并行"）

搜索空间被刻意收敛到 Megatron 有成熟实现的"公共域"，`_common_parallel_reasons()` 强制校验：

| 条件 | 约束 |
|---|---|
| TP=1 | 所有策略必须 `replicate` |
| TP>1 | attention 三段统一 `head_shard`，MLP 统一 `colrow_shard`，lm_head=`vocab_shard`，embedding∈{replicate, vocab_shard} |
| tie_word_embeddings | embedding 与 lm_head 策略必须一致 |

越界方案会被标记"不可下放"并在排序中惩罚。**好处**：任何搜索结果都保证能被 Megatron 正确执行，不会出现"自动搜出来的方案跑不起来"。

### 4.4 三段式策略（front/mid/back）

每个策略字段都有三段变体（如 `attention_policy_front/mid/back`），支持"模型前几层/中间层/后几层用不同切分"的实验——这是为研究非均匀切分预留的接口，当前标准域内三段退化为统一值。

---

## 五、两条执行后端

### MEGATRON 后端（生产路径）

```
launch_megatron_plan.sh
  → torch.distributed.run --nnodes --node_rank --master_addr ...
    → tools/run_megatron_hygon.py
      → megatron_hygon_bootstrap.bootstrap()     # 挂上游源码 + dcu_megatron 补丁
        → runpy 执行 Megatron-LM pretrain_gpt.py # 真实训练
```

- `run_megatron_hygon.py` 的职责：剥掉静态 `--rank/--world-size/--local-rank/--dist-url`，按 torchrun 注入的真实环境变量重新注入（多机下 rank 由 torchrun 分配，静态值会错）。
- `bootstrap()` 的职责：按 `环境变量 → <repo>/Megatron-LM → Megatron-LM-das/3rdparty/Megatron-LM` 顺序找上游源码挂进 `sys.path`，再导入 `dcu_megatron` 打海光补丁，最后校验 `parse_args` 确实来自补丁层。

### PYTORCH 后端（参考实现/对照组）

`tools/run_pytorch_llama_tp_dp.py`（1400 行）是**不依赖 Megatron 的手写 DP+TP 运行时**：

- `ParallelContext`: `tp_rank = rank % tp_size`，`dp_rank = rank // tp_size`，手工 `dist.new_group`
- 手写 Megatron 风格算子：`ColumnParallelLinear` / `RowParallelLinear` / `_CopyToTensorParallelRegion` / `_VocabParallelCrossEntropy`（按 vocab 分片的交叉熵，避免 logits 全聚合）
- DDP 只挂在 DP 组上；`env://` 初始化（天然多机兼容）
- 支持 `--inspect-only` 静态检查切分布局（单卡开发机可用）

**限制**：只实现了 DP+TP（无 PP/EP）——编译器会在 PYTORCH 后端下自动把 PP/EP 压回 1，显式给 pp>1 会直接报错。

两个后端共用同一份 Megatron indexed dataset（wikitext2），输出统一格式 `[BACKEND][step i/N] loss=x`，方便横向对比与汇报截图。

---

## 六、三大搜索策略

注册表模式（`register_search_strategy`），三内置策略共享同一套候选枚举/打分：

| 策略 | 机制 | 耗时 | 适用 |
|---|---|---|---|
| `BLOCK_LEVEL_FAST` | 直接枚举标准域全部 mesh×mbs 组合 | ~5ms | 默认；候选空间小时最优 |
| `HIERARCHICAL` | 两阶段：先用代理分（结构惩罚+偏好）预筛 proposal，再全量展开 | ~10ms | 候选空间大时 |
| `RL_GUIDED_PPO` | `PlacementController`（Linear 骨干+policy/value 头）在线 rollout（12 episode），clipped PPO + GAE 更新 | ~秒级 | 研究搜索策略本身 |

策略偏好（`search_policy`）影响打分排序：`BALANCED / MEMORY_FIRST / THROUGHPUT_FIRST / TP_FIRST / DP_FIRST`。默认还会顺带 benchmark 两个启发式策略，方便对比。

---

## 七、海光（Hygon）适配链

```
应用层     autoparallel / megatron_mvp          （纯 Python，硬件无关）
             │
补丁层     dcu_megatron (pip)                   动态替换 megatron.* 模块
             │  （MegatronAdaptation.register 机制：函数/类替换）
上游层     Megatron-LM core_v0.18.2             ← Megatron-LM-das/3rdparty 子模块
             │                                    （本仓库当前未 init，见 §10）
框架层     hcu_megatron (das 仓库)               数据集/启动等增强
             │
运行时     torch 2.7.1+das / RCCL(NCCL 兼容) / transformer_engine 2.10
             │
硬件层     DTK (/opt/dtk) → BW1000 (64GB HBM2e, HIP/ROCm 生态)
```

多机通信要点（来自 `Megatron-LM-das/requirements/env.sh`）：
- `NCCL_ALGO=Ring`（海光推荐环算法）
- `NCCL_IB_HCA` 自动探测 mlx/shca 两套 IB 网卡 profile
- `GLOO_SOCKET_IFNAME`/`NCCL_SOCKET_IFNAME=eth0`（容器内网络）
- 参考仓库自带 `multi/rccltest`（多机带宽测试）与 `multi/check_nodes`（节点体检）脚本，扩集群前建议先跑

---

## 八、多机升级详解（本次改动）

### 8.1 原框架的 7 个单机假设（升级前）

| # | 问题 | 位置 |
|---|---|---|
| 1 | 搜索上限 16/64 卡 | `search.py` / `config.py` |
| 2 | **PP 硬编码 =1**（mesh 只有 tp×dp） | `search.py::_build_parallel_plan` |
| 3 | 只识别稠密 LLaMA 配置，无 MoE 字段 | `detect.py` |
| 4 | **`GPUS_PER_NODE` 默认=world_size**：128 卡会把 128 进程堆到 1 台机器 | `lowering.py` / `render.py` |
| 5 | 无集群编排：每台机器要手工设 NODE_RANK | — |
| 6 | bootstrap 只认 `<repo>/Megatron-LM` 布局，当前仓库该目录不存在 | `megatron_hygon_bootstrap.py` |
| 7 | 无 EP（专家并行）维度 | `types.py` |

骨架其实已具备多机能力（NNODES/NODE_RANK/MASTER_ADDR 透传、`env://` 初始化、`--nnodes` 参数），缺的是上面这些"最后一公里"。

### 8.2 升级后的搜索空间

```
MeshChoice(tp, dp, pp)                    # 三维 mesh
  PP 候选 = num_layers 的因子 ∪ {2,4,8,16}   # Megatron 支持不均匀切分
  EP 候选 = 同时整除 dp 和 num_experts 的最大值 ≤ max_ep（仅 MoE）
world = tp × dp × pp（不变）；EP 在 DP 组内部细分专家，不改变 world
```

新增 `Config` 字段（都有安全默认值，**不传任何新字段时行为与旧版完全一致**——已回归验证）：

```python
Config(
    num_nodes=16, gpus_per_node=8,       # 集群拓扑 → 脚本默认值
    master_addr=None, master_port=None,  # rendezvous 默认值
    search_max_pp=8,                     # PP 搜索上限（默认 1）
    search_max_ep=8,                     # EP 搜索上限（默认 1，仅 MoE 生效）
    search_max_world_size=128,           # 搜索上限（默认从 64 提升）
)
```

### 8.3 显存模型的 PP/EP 感知

- 每卡参数 = 层参数(÷TP) × `ceil(L/pp)/L`，路由专家额外 ÷EP
- 激活同步按 stage 比例缩减
- MoE 层与稠密层（`first_k_dense_replace`）分开估算

### 8.4 多机启动编排（三层产物）

`num_nodes > 1` 时 lowering 额外生成：

1. **hostfile 模板**（第一行=主节点；`host[:ssh_port] slots=N` 格式，与 das 仓库习惯一致）
2. **launch_cluster.sh** → 调用 `tools/launch_cluster.py`
3. 生成的训练脚本内部：`NNODES` 默认=节点数、`GPUS_PER_NODE` 默认=world/节点数、`--distributed-timeout-minutes`（多机建链超时）、多机用法注释

`tools/launch_cluster.py`（纯标准库，不依赖 mpirun）：

```bash
# hostfile 每行一个节点，第一行是主节点
python3 tools/launch_cluster.py \
    --hostfile generated/autoparallel/hostfile \
    --script  generated/autoparallel/launch_megatron_plan.sh \
    --gpus-per-node 8            # [--dry-run] [--log-dir logs/] [--master-addr x.x.x.x]
```

它做的事：解析 hostfile → 逐节点 SSH 注入 `NODE_RANK/NNODES/MASTER_ADDR/MASTER_PORT/GPUS_PER_NODE` → 并发拉起同一份脚本 → 日志按 `[node0:host]` 前缀聚合（可选分文件）。等价于 das 仓库 `run_glm5_300B.sh` 的 mpirun+hostfile 流程，但只要求 SSH 免密。

### 8.5 升级验证记录（本机 1×BW 64GB 实测）

| 验证项 | 结果 |
|---|---|
| 双机16卡规划 GLM-4.5-Air | 搜索正确判定 16 卡放不下（需 ≥32 卡），输出 32 卡方案 ✅ |
| 128 卡规划 GLM-4.5-Air | **TP=8, DP=8, PP=2, EP=8，预估 24.6GB/64GB** ✅ |
| 生成脚本多机语义 | NNODES=16 / GPUS_PER_NODE=8 / PP/EP/MoE 参数全部正确 ✅ |
| launch_cluster dry-run | 双节点 SSH 命令正确（含非默认端口）✅ |
| 旧用法回归（稠密 LLaMA，不传新字段） | TP=2/DP=1/PP=1/EP=1，与旧版一致 ✅ |
| PYTORCH 后端守卫 | pp>1 显式传入→明确报错；max_pp=8 自动压回 1 ✅ |
| bootstrap 路径 | 三候选解析 + 缺源码时给出 submodule 修复指引 ✅ |

---

## 九、GLM-4.5-Air 128 卡实战路线

### 9.1 模型与显存账

GLM-4.5-Air（`glm4_moe` / Glm4MoeForCausalLM，结构见 `models/glm45_air_105b_spec/config.json`）：

```
46 层 = 1 稠密层 + 45 MoE 层
每 MoE 层 = 128 路由专家(top-8) × 3×4096×1408 + 1 共享专家(10944)
attention: GQA 96头/8KV头，head_dim=128（q_proj 从 4096 上采样），partial rope 0.5
总参数 ≈ 105B，激活 ≈ 12B/token，vocab 151552（untied）
```

**128×BW1000(64GB) 显存账**（全量微调，bf16 + fp32 Adam 状态）：

```
静态合计 ≈ 105B × 16B/参数 ≈ 1.7TB  →  128 卡均摊 ~13.5GB/卡（开分布式优化器前上限）
搜索器实选 TP=8/PP=2/EP=8：预估 24.6GB/卡（含激活，保守估算）→ 余量健康
```

**重要结论（搜索器自动发现）**：双机 16 卡（共 1TB 显存）**放不下** 105B 全量微调（静态需求 1.7TB）。16 卡阶段请用以下之一：① 小模型（如 GLM-4.1VL-9B 级别）验证多机链路；② LoRA/部分参数微调；③ 直接跳过 16 卡冒烟、在 32+ 卡上做首次全量验证。

### 9.2 推荐的三阶段计划

**阶段 0：环境准备（每台节点）**

```bash
cd Megatron-LM-das && git submodule update --init --recursive   # 关键！补齐上游 Megatron-LM + Megatron-Bridge
bash multi/rccltest/test_rccl.sh <hostfile> <n>                 # 多机 RCCL 带宽体检
```

**阶段 1：双机 16 卡链路验证（用小模型）**

```bash
# 规划+生成（本机即可）
python3 examples/autoparallel_multinode_train.py \
    --num-nodes 2 --gpus-per-node 8 --backend PYTORCH \
    --model-path <小模型目录> --run-mode LOWER --output-dir generated/dual_node
# PYTORCH 后端: world=16 全 DP+TP，验证多机 rendezvous/通信/数据链路

# 集群执行
vim generated/dual_node/hostfile        # localhost → 真实节点名
python3 tools/launch_cluster.py \
    --hostfile generated/dual_node/hostfile \
    --script  generated/dual_node/launch_pytorch_plan.sh --gpus-per-node 8
```

**阶段 2：MEGATRON 后端 16 卡小规模验证**（同上，`--backend MEGATRON`，需阶段 0 完成）

**阶段 3：128 卡 GLM-4.5-Air**

```bash
python3 examples/autoparallel_multinode_train.py \
    --num-nodes 16 --gpus-per-node 8 \
    --model-path models/glm45_air_105b_spec \
    --search-max-pp 8 --search-max-ep 8 --memory-target-gb 64 \
    --run-mode LOWER --output-dir generated/glm45_air_128
# → 自动方案 TP=8/DP=8/PP=2/EP=8，24.6GB/卡
```

或直接用**保底参考脚本**（绕过 autoparallel，走 Megatron-LM-das 原生链路，`--use-bridge` 由 Megatron-Bridge 的 `GLM45Bridge` 自动完成 HF glm4_moe ↔ Megatron 映射，解决 partial-rope 等 GPT 参数表达不了的问题）：

```bash
cd scripts/glm45_air
HF_MODEL_PATH=/data/models/glm-4.5-air \
DATA_PATH=/data/corpus_text_document \
LAUNCHER=torchrun bash run_multinode.sh hostfile.128cards 16
```

> 两条路线的关系：autoparallel 生成的是**通用 GPT 路径**参数（MoE 参数可正确渲染，但 partial rotary 需要模型 spec 层支持）；bridge 路径结构映射最完整。**128 卡正式跑建议用 bridge 路径**，autoparallel 负责前期的方案搜索与可行性验证。

### 9.3 并行配置速查

| 规模 | 推荐 | 说明 |
|---|---|---|
| 双机 16 卡（链路验证） | TP=2, DP=8（小模型）或 PYTORCH 后端 | 验证 rendezvous/RCCL/数据 |
| 32 卡（105B 最低可行） | TP=8, PP=4, DP=1 | 搜索器给出的最小可行域 |
| **128 卡（正式）** | **TP=8, DP=8, PP=2, EP=8**（自动）或 TP=2, PP=4, EP=8, ETP=2（bridge 参考） | 显存余量大，GBS 建议 256+ |

---

## 十、当前限制与后续工作

1. **上游子模块未初始化**（本仓库现状）：`Megatron-LM-das/3rdparty/{Megatron-LM,Megatron-Bridge}` 为空 → MEGATRON 后端在本机只能生成脚本不能执行。修复：`git submodule update --init --recursive`。PYTORCH 后端不受影响。
2. **autoparallel 的 MEGATRON 下放走通用 GPT spec**：GLM-4.5 的 partial rotary / MTP 层需要 spec 层（或 bridge）支持；MoE 参数已可正确渲染，结构级映射建议走 `--use-bridge`。
3. **GBS 搜索域较小**：`global_batch_size = mbs × dp`，mbs∈{1,2,4}。128 卡正式训练建议手工把 GBS 调到 256+（直接改生成脚本或用 explicit_plan）。
4. **EP 搜索是启发式**（每 mesh 只试 `{1, 最大合法 ep}`），没有全枚举；RL 策略的动作空间也还是 mesh+embedding 粒度。
5. **数据侧**：MoE/长序列场景建议 `--moe-token-dispatcher-type alltoall`、`--overlap-param-gather` 等已在参考脚本中，autoparallel 生成的保守配置可按需叠加。
6. **海光已知坑**（继承自原框架）：训练完成后的 checkpoint 保存阶段可能段错误 → smoke 脚本默认关保存（`lowering.py` 有注释）；需存档时单独验证。

---

## 附：快速上手命令

```bash
export PYTHONPATH=$(pwd)

# 单卡开发机上规划 128 卡方案（无需真集群）
python3 examples/autoparallel_multinode_train.py --num-nodes 16 --gpus-per-node 8 \
    --model-path models/glm45_air_105b_spec --search-max-pp 8 --search-max-ep 8 \
    --memory-target-gb 64 --run-mode LOWER --output-dir generated/glm45_air_128

# 锁定方案（复现实验）
python3 examples/autoparallel_multinode_train.py --explicit tp=8,dp=8,pp=2,ep=8 ...

# 集群拉起（改好 hostfile 后）
python3 tools/launch_cluster.py --hostfile generated/glm45_air_128/hostfile \
    --script generated/glm45_air_128/launch_megatron_plan.sh --gpus-per-node 8

# 产物清单
# generated/<name>/
#   ├── search_result.json          搜索全量候选与打分
#   ├── coarse_ir.json              粗 IR
#   ├── megatron_plan_config.json   Megatron 完整参数
#   ├── launch_megatron_plan.sh     多机训练脚本（planning）
#   ├── launch_megatron_smoke.sh    冒烟脚本（本机/少卡）
#   ├── hostfile                    多机节点清单模板
#   └── launch_cluster.sh           一键拉起
```
