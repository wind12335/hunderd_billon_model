# 单机 vs 多机：大模型分布式训练完全指南

> 仓库：`/root/private_data/thousand_model`（自研 autoparallel 框架 + Megatron-LM-das + 海光 DCU/BW1000 栈）
> 面向读者：需要**从原理到实操彻底掌握**多机多卡训练的同学
> 配套文档：`FRAMEWORK_GUIDE.md`（框架架构详解）；本文专注"单机→多机到底差在哪、指令怎么写、坏了怎么查"
>
> 本文分三个层次讲解同一件事：
> **L1 原生 PyTorch/torchrun 层**（学原理，零框架依赖）→ **L2 Megatron 层**（学工业级工程）→ **L3 本框架 autoparallel 层**（学封装与编排）

---

## 目录

- [一、地基：分布式训练的核心概念](#一地基分布式训练的核心概念)
- [二、单机多卡：完整过程与指令](#二单机多卡完整过程与指令)
- [三、多机多卡与单机的六层差异（原理）](#三多机多卡与单机的六层差异原理)
- [四、多机实操指令（三层，从手搓到框架）](#四多机实操指令三层从手搓到框架)
- [五、环境变量与参数总表（背下来）](#五环境变量与参数总表背下来)
- [六、上集群前的体检清单](#六上集群前的体检清单)
- [七、故障排查手册（症状→原因→修法）](#七故障排查手册症状原因修法)
- [八、单机→多机迁移检查表](#八单机多机迁移检查表)
- [九、一次双机 16 卡训练的完整时序](#九一次双机-16-卡训练的完整时序)
- [十、本仓库当前状态与已知雷区](#十本仓库当前状态与已知雷区)
- [十一、推荐学习路径（六级闯关）](#十一推荐学习路径六级闯关)

---

## 一、地基：分布式训练的核心概念

### 1.1 五个必须张口就来的词

| 术语 | 含义 | 单机 8 卡示例 | 双机 16 卡示例 |
|---|---|---|---|
| **rank** | 全局进程编号（一个 GPU 对应一个进程） | 0~7 | 0~15，**跨节点连续编号** |
| **local_rank** | 进程在**本节点内**的编号 | 0~7 | 两台各自 0~7 |
| **world_size** | 总进程数 | 8 | 16 = 节点数 × 每节点卡数 |
| **process group（进程组）** | 一组需要互相通信的进程（TP 组/DP 组/PP 组…） | — | **跨节点也能成组**（如 TP=8 横跨两台机器） |
| **rendezvous（会合）** | 所有进程向一个地址报到、凑齐 world_size 后开工的过程 | 本机自动完成 | **必须显式约定 MASTER_ADDR/PORT** |

### 1.2 四种并行维度（本框架搜索的就是它们的组合）

| 维度 | 切什么 | 通信量级 | 对显存的作用 |
|---|---|---|---|
| **DP** 数据并行 | 每张卡放**完整模型**，喂不同数据，梯度 all-reduce | 每步参数量级（大） | 不省单卡显存（ZeRO/分布式优化器除外） |
| **TP** 张量并行 | 把每层权重矩阵**竖着/横着切成条**，一张卡拿一条 | 每层 2 次 all-reduce（极大，**必须走高速互联**） | 权重+激活÷TP |
| **PP** 流水线并行 | 把**层**切成段，一台机器（组）拿一段 | 只传层间激活（小） | 层数÷PP，是百亿级模型可行性的关键 |
| **EP** 专家并行 | 把 **MoE 的不同专家**分到不同卡 | all-to-all（中） | 专家权重÷EP |

以 GLM-4.5-Air 105B 在 128 卡为例（本框架自动搜索结果）：
`TP=8, DP=8, PP=2, EP=8`，即 world = 8×8×2 = 128，EP 在 DP 组内部细分专家。
显存账：静态约 105B×16B/参数 ≈ 1.7TB → 每卡摊 ~13.5GB（分布式优化器进一步分片），加激活后预估 24.6GB/64GB。

**关键直觉**：TP 通信最重（每层两次全归约），所以 TP 一般限制在单机 8 卡的片间互联内；PP/DP/EP 通信轻，适合跨机走网卡。这就是"单机内总线、跨机走网"的并行布局原则。

### 1.3 单机与多机的通信介质（本质差异之源）

```
单机 8 卡：  GPU0 ⇄ GPU1 ⇄ ... ⇄ GPU7     走主板：HBM/片间互联/PCIe
                                        RCCL 自动发现，零配置

双机 16 卡： node0: GPU0..7 ⇄ node1: GPU0..8   跨机部分必须走网卡（IB/RoCE/以太网）
                                        需要人工指定：用哪块网卡、什么算法、什么超时
```

所有"多机适配"问题，归根结底都是**跨机这一段网络**引出的。

---

## 二、单机多卡：完整过程与指令

### 2.1 L1：原生 PyTorch（先掌握这个，其他全是它的封装）

**训练代码三要素**（任何分布式训练脚本都是这三件事）：

```python
import os, torch
import torch.distributed as dist

# ① 初始化进程组：env:// 表示从环境变量读 RANK/WORLD_SIZE/MASTER_ADDR...
dist.init_process_group(backend="nccl", init_method="env://")   # DCU 上 nccl 映射到 RCCL

# ② 绑定本进程专属的卡（torchrun 会把 LOCAL_RANK 注入环境）
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

# ③ 数据按 rank 切片，保证每个 rank 喂不同数据
sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
# 每个 epoch 前 sampler.set_epoch(epoch)，否则 shuffle 相同

model = ... .to(local_rank)
model = DDP(model, device_ids=[local_rank])
# ... 正常训练循环；DDP 在 backward 时自动 all-reduce 梯度
```

**启动指令（单机 8 卡）**：

```bash
torchrun --nproc_per_node=8 train.py
# 等价旧写法：python -m torch.distributed.launch --nproc_per_node=8 train.py（已淘汰）
```

torchrun 单机时自动完成：拉起 8 个进程、注入 `RANK/LOCAL_RANK/WORLD_SIZE=8`、
在本机起一个 rendezvous 端口、设 `MASTER_ADDR=127.0.0.1`。**你什么都不用管**。

### 2.2 L2：Megatron-LM 单机（das 仓库）

Megatron 的单机脚本本质上就是"torchrun + 一大坨模型/并行参数"：

```bash
cd Megatron-LM-das
source /opt/dtk/env.sh                      # DTK 运行时
source requirements/env.sh                  # RCCL 网络参数（单机时基本无感）

torchrun --nproc_per_node 8 pretrain_gpt.py \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 1 \
    --num-layers 32 --hidden-size 4096 --num-attention-heads 32 \
    --train-iters 20 --lr 1e-4 \
    --data-path XXX_text_document \
    --tokenizer-type HuggingFaceTokenizer --tokenizer-model XXX
```

Megatron 内部自己完成 rank→(TP,DP,PP,EP) 的映射与进程组创建，使用者只管给并行度。

### 2.3 L3：本框架单机用法

```bash
cd /root/private_data/thousand_model
export PYTHONPATH=$(pwd)

# 声明式：只说"1 台机器 8 张卡训这个模型"
python3 examples/autoparallel_multinode_train.py \
    --num-nodes 1 --gpus-per-node 8 \
    --model-path model/Llama-3.1-8B \
    --backend MEGATRON --run-mode LOWER \
    --sequence-length 4096 \
    --output-dir generated/single_node_llama8b
```

产物 `generated/single_node_llama8b/launch_megatron_plan.sh` 内部即 2.2 的完整形态，
且 `NNODES=1` 时 torchrun 不需要 `--node_rank/--master_addr`（单机默认值即可直接跑）：

```bash
bash generated/single_node_llama8b/launch_megatron_plan.sh
```

`--run-mode LAUNCH` 则在本机直接执行 smoke 脚本（注意：**多机配置下不要用 LAUNCH**，
smoke 会按多机 NNODES 等待不存在的节点而挂死，见第七节故障 1）。

---

## 三、多机多卡与单机的六层差异（原理）

> 简答版：**① 要先"碰头"（rendezvous）；② 跨机通信要走对网卡（NIC）；③ 要有人去每台机器上点火（编排）；④ 数据要全局分片；⑤ 两边环境必须一致；⑥ 挂的方式变多了。**

### 3.1 第一层：rendezvous——所有进程先碰头

单机进程在一个 shell 里同时拉起，彼此可见。多机的进程分布在两台机器上，
**互相不知道对方存在**，必须约定集合点：

```
                    MASTER_ADDR=node0 的 IP，MASTER_PORT=29500
                                │
   node0: torchrun --nnodes=2 --node_rank=0 --master_addr=node0IP --master_port=29500
      └─ 拉起 8 进程（RANK=0..7，LOCAL_RANK=0..7），rank0 在 29500 端口监听
   node1: torchrun --nnodes=2 --node_rank=1 --master_addr=node0IP --master_port=29500
      └─ 拉起 8 进程（RANK=8..15，LOCAL_RANK=0..7），向 node0:29500 报到
                                │
        16 个进程全部报到 → world_size=16 凑齐 → 各自建 TP/DP/PP 进程组 → 开始训练
```

要点：
- `MASTER_ADDR` 必须是**各节点都能访问到的地址**（主节点的对外 IP/主机名），绝不能是 127.0.0.1
- `MASTER_PORT` 是一个 TCP 端口，**防火墙必须放行**
- 建链慢（节点开机有先后），所以要 `--distributed-timeout-minutes 60` 这类超时
- `node_rank` 决定本机的 rank 偏移：node0 占 0-7，node1 占 8-15

### 3.2 第二层：通信后端与网卡选择（你问的"NIC 适配"）

跨机的梯度同步由 **RCCL（海光的 NCCL 兼容层）** 完成，它需要知道**走哪块网卡**。
服务器常有"管理网（千兆）+ 计算网（IB/百兆以太）"多块网卡，选错=慢几十倍。

| 环境变量 | 作用 | 海光集群典型值 |
|---|---|---|
| `NCCL_SOCKET_IFNAME` | 控制面/回退数据面用的网卡名 | `eth0`（容器内网络） |
| `GLOO_SOCKET_IFNAME` | gloo 后端（元数据通信）网卡 | `eth0` |
| `NCCL_IB_HCA` | 数据面用哪块 IB 网卡 | mlx5_* / shca（das 的 env.sh 自动探测两套 profile） |
| `NCCL_ALGO` | 集合通信算法 | `Ring`（海光推荐） |
| `NCCL_DEBUG` | 排障开关 | `INFO`（排障时用） |
| `NCCL_TIMEOUT`/`TORCH_NCCL_BLOCKING_WAIT` | 通信超时行为 | 排查挂死时用 |

这些变量**不是改代码，是改环境**。das 仓库 `Megatron-LM-das/requirements/env.sh` 已含自动探测；
上集群第一件事是验证它探测的结果和实际计算网一致（方法见第六节 rccltest）。

### 3.3 第三层：集群编排——谁去每台机器上点火

单机一条 bash；多机需要"在每台机器上以不同 NODE_RANK 执行同一份脚本"。三种等价方式：

| 方式 | 指令形态 | 依赖 | 本仓库对应 |
|---|---|---|---|
| 手工 SSH | 每台机器自己 `ssh` 上去敲 `NODE_RANK=k bash train.sh` | 无 | 调试期用，最透明 |
| **mpirun + hostfile** | `mpirun -np 16 --hostfile hf bash train.sh`（train.sh 里从 `OMPI_COMM_WORLD_RANK` 取节点号） | MPI + 统一 SSH 端口 | das 官方 `examples/GLM/run_glm5_300B.sh`、`scripts/glm45_air/run_multinode.sh` |
| **torchrun 原生多机** | 每节点跑一次 `torchrun --nnodes --node_rank ...`（由外部工具并发触发） | SSH | 本框架 `tools/launch_cluster.py` |

本框架的编排器 `tools/launch_cluster.py` 实际生成的远端命令长这样（dry-run 可见）：

```
ssh -p 22 -o StrictHostKeyChecking=no -o BatchMode=yes node0 \
    "cd /root/private_data/thousand_model && NODE_RANK=0 NNODES=2 MASTER_ADDR=node0 MASTER_PORT=29500 GPUS_PER_NODE=8 bash .../launch_megatron_plan.sh"
ssh ... node1 "cd ... && NODE_RANK=1 NNODES=2 MASTER_ADDR=node0 ... bash .../launch_megatron_plan.sh"
```

注意三点：① 训练脚本从环境变量读拓扑（`${NNODES:-2}` 这种带默认值的写法）；
② 两台机器**仓库路径必须一致**；③ SSH 是非交互 shell，**不加载 conda**，`python3` 必须在系统 PATH。

### 3.4 第四层：数据分片

- 单机：DistributedSampler 在 8 个 rank 间切数据。
- 多机：DP rank 是**全局编号**（0~15），每个 rank 必须拿到不同分片，否则等价于 batch 缩水 16 倍。
  - 原生 PyTorch：`DistributedSampler` 天然按全局 world_size 切，无需改动——**这是 env:// 初始化的好处**
  - 本框架 PYTORCH 后端：按 `(iteration×dp_size + dp_rank)×tokens_per_step` 错峰切片
  - Megatron：内部按 data-parallel group 自动切
- 数据文件必须**两台都能读到**（NFS 共享，或同步到相同路径）。

### 3.5 第五层：环境一致性

两台机器必须：相同 torch/DTK/megatron 版本、相同代码、相同绝对路径、相同模型/数据路径。
任何一台版本漂移，轻则 hang（协议不匹配），重则静默数值错误。

### 3.6 第六层：故障面

单机挂=全挂；多机要考虑：一个节点 SSH 不通（编排层要能报错而不是无限等），
一个节点掉线（NCCL 超时后全队报错退出），节点起来有先后（建链超时要够长），
时钟漂移（影响日志与监控对齐，建议 chrony 同步）。

---

## 四、多机实操指令（三层，从手搓到框架）

### 4.1 L1 手搓版：20 行代码验证多机链路（强烈推荐第一步）

把下面存成 `/tmp/dist_hello.py`：

```python
import os, torch
import torch.distributed as dist

dist.init_process_group(backend="nccl", init_method="env://", timeout=__import__('datetime').timedelta(minutes=10))
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
rank, world = dist.get_rank(), dist.get_world_size()

x = torch.ones(1024, device=f"cuda:{local_rank}") * (rank + 1)
dist.all_reduce(x)                    # 跨机 all-reduce：验证 RCCL 数据面
dist.barrier()                        # 跨机同步：验证 rendezvous/控制面

if rank == 0:
    print(f"[OK] world_size={world}, all_reduce 求和={x[0].item()} (期望 {sum(range(1, world+1))})")
dist.destroy_process_group()
```

在**两台机器**分别执行（只有 master 地址和 node_rank 不同——这就是多机的全部秘密）：

```bash
# node0（主节点，IP 记为 10.0.0.1）
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \
    --master_addr=10.0.0.1 --master_port=29500 /tmp/dist_hello.py

# node1
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=1 \
    --master_addr=10.0.0.1 --master_port=29500 /tmp/dist_hello.py
```

打印 `[OK] world_size=16, all_reduce 求和=136` 即多机 rendezvous + RCCL 全通。
**这一步不通，后面所有 Megatron/框架都不可能通；这一步通了，问题就只剩模型侧。**

### 4.2 L2 Megatron 官方方式（mpirun + hostfile，das 仓库范式）

```bash
cd Megatron-LM-das
source /opt/dtk/env.sh
source requirements/env.sh

# hostfile：每行一个主机名（das 惯例不带 slots，脚本自动补）
cat > /tmp/hf <<EOF
node0
node1
EOF

mpirun -np 16 --hostfile /tmp/hf \
    --mca plm_rsh_args '-p 11451' \      # 集群 SSH 端口统一时才需要
    bash my_train.sh                       # 脚本内从 $OMPI_COMM_WORLD_RANK 推 NODE_RANK
```

单节点训练脚本 `my_train.sh` 的骨架（多机差异只有开头四个变量）：

```bash
#!/usr/bin/env bash
# ── 多机身份四件套（单机脚本没有这一段）──────────────
NNODES=${NNODES:-2}
NODE_RANK=${NODE_RANK:-${OMPI_COMM_WORLD_RANK:-0}}     # mpirun 给的节点序号
MASTER_ADDR=${MASTER_ADDR:-node0}
MASTER_PORT=${MASTER_PORT:-29500}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

torchrun --nproc_per_node ${GPUS_PER_NODE} \
    --nnodes ${NNODES} --node_rank ${NODE_RANK} \
    --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} \
    pretrain_gpt.py \
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 2 \
    ...（模型/数据/训练参数与单机完全相同）
```

**记住这个不等式**：`TP × DP × PP = NNODES × GPUS_PER_NODE`，两边对不上 Megatron 启动即崩。

### 4.3 L3 本框架方式（声明式，当前推荐的主路径）

**双机 16 卡 Llama-3.1-8B 全流程**（0/1 是一次性的，2/3 每次实验重跑）：

```bash
# ── 0. 前置（两台节点都要）────────────────────────────
cd /root/private_data/thousand_model
git submodule update --init --recursive   # 补齐上游 Megatron-LM（MEGATRON 后端硬依赖）
# SSH 免密：主节点 ssh-keygen -t rsa，公钥追加到两台 ~/.ssh/authorized_keys
# 验证远端非交互环境：
ssh node1 "which python3 && python3 -c 'import torch; print(torch.cuda.device_count())'"

# ── 1. 下载模型（主节点；NFS 共享则一份）───────────────
pip install -U modelscope
modelscope download --model LLM-Research/Meta-Llama-3.1-8B \
    --local-dir model/Llama-3.1-8B

# ── 2. 规划 + 生成多机脚本（主节点，不占卡，秒级）──────
export PYTHONPATH=$(pwd)
python3 examples/autoparallel_multinode_train.py \
    --num-nodes 2 --gpus-per-node 8 \
    --model-path model/Llama-3.1-8B \
    --backend MEGATRON --run-mode LOWER \
    --sequence-length 4096 --train-iters 20 \
    --output-dir generated/dual16_llama8b
# 预期：搜索结果 world_size=16（如 TP=4/DP=4），显存 <10GB/卡

# ── 3. 填 hostfile（第一行=主节点，决定 MASTER_ADDR）──
vim generated/dual16_llama8b/hostfile
#   node0 slots=8
#   node1 slots=8        ← 真实主机名/IP；SSH 非 22 端口写 node1:36000

# ── 4.（可选）node0 单节点冒烟启动链路 ────────────────
NNODES=1 NODE_RANK=0 GPUS_PER_NODE=8 bash generated/dual16_llama8b/launch_megatron_smoke.sh

# ── 5. 双机拉起（在主节点执行）────────────────────────
python3 tools/launch_cluster.py \
    --hostfile generated/dual16_llama8b/hostfile \
    --script  generated/dual16_llama8b/launch_megatron_plan.sh \
    --gpus-per-node 8 --log-dir logs/dual16_llama8b
```

框架在背后替你做的事：搜索并行方案 → 渲染 Megatron 参数 → 写出带多机默认值的
launch 脚本 + hostfile 模板 → 逐节点 SSH 注入身份四件套并发拉起 → 聚合日志。

**成功判据**：两节点日志均出现 `iteration X/20 ... loss=...`，无 RCCL timeout；
`hy-smi` 两台各 8 卡满载。

> 注意：此链路是**随机初始化 + mock 数据**的多机验证（loss 数值无意义）；
> 真实权重的 105B 微调走 `scripts/glm45_air` 的 bridge 路线（见第十节）。

### 4.4 hostfile 两种格式对照（别混用）

| 来源 | 格式 | 消费者 |
|---|---|---|
| 本框架生成的 | `host[:ssh_port] slots=N`，`#` 注释，第一行=主节点 | `tools/launch_cluster.py` |
| das 官方脚本 | 纯主机名每行一个（脚本自动补 `slots=8`） | `run_multinode.sh` / mpirun |

---

## 五、环境变量与参数总表（背下来）

### 5.1 torchrun 分布式参数

| 参数 | 含义 | 单机 | 双机 16 卡 |
|---|---|---|---|
| `--nproc_per_node` | 每节点进程数 | 8 | 8 |
| `--nnodes` | 节点数（可 `min:max` 弹性） | 省略=1 | **2** |
| `--node_rank` | 本节点编号（决定 rank 偏移） | 省略=0 | **node0=0, node1=1** |
| `--master_addr` | rendezvous 主节点地址 | 省略=127.0.0.1 | **主节点对外 IP** |
| `--master_port` | rendezvous TCP 端口 | 省略=29500 | 29500（防火墙放行） |
| `--rdzv-backend/cfr` | 弹性模式用 | 一般不用 | 固定拓扑不需要 |

### 5.2 进程内环境变量（torchrun 注入，训练代码读取）

| 变量 | 含义 | 双机 16 卡下 node1 的值 |
|---|---|---|
| `RANK` | 全局 rank | 8~15 |
| `LOCAL_RANK` | 节点内 rank | 0~7 |
| `WORLD_SIZE` | 总进程数 | 16 |
| `MASTER_ADDR/PORT` | 会合地址 | 主节点 IP:29500 |
| `NODE_RANK` / `NNODES` / `GPUS_PER_NODE` | 本框架/脚本的拓扑约定 | 1 / 2 / 8 |

### 5.3 本框架生成脚本的变量约定（launch_megatron_plan.sh 头部）

```bash
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"   # ← 由 launch_cluster.py 注入主节点 IP
export MASTER_PORT="${MASTER_PORT:-29500}"
export NNODES="${NNODES:-2}"                     # ← 节点数默认值来自规划
export NODE_RANK="${NODE_RANK:-0}"               # ← 每节点不同！
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
export WORLD_SIZE="$((NNODES * GPUS_PER_NODE))"
export DIST_TIMEOUT_MIN="${DIST_TIMEOUT_MIN:-60}"  # 建链超时(分钟)
```

### 5.4 网络/RCCL 变量（多机专属）

见 3.2 节表格。排障第一招：`export NCCL_DEBUG=INFO` 后重跑，日志会打印
实际使用的网卡/IB 卡/拓扑，一眼看出有没有选错网卡。

---

## 六、上集群前的体检清单

```bash
# 1. 节点体检（das 自带工具，两台都要能过）
cd Megatron-LM-das/multi/check_nodes
vim clushnode                 # 填节点清单
bash clush.sh                 # SSH 连通性筛查
bash run_check.sh 2           # 驱动/卡数/显存/版本一致性

# 2. 多机 RCCL 带宽体检（验证网卡选择是否正确）
cd Megatron-LM-das/multi/rccltest
vim hostfile                  # 两台节点
bash test_rccl.sh hostfile 2  # 看 all_reduce 带宽是否打满计算网
# 打满（如 IB 200Gbps 级）= 网卡对了；只有几 GB/s = NCCL_IB_HCA/SOCKET_IFNAME 选错

# 3. 手工最小验证（4.1 的 dist_hello.py）——rccltest 编译不了时的替代

# 4. 基础项
ssh node1 "hostname -I && which python3 && python3 -c 'import torch;print(torch.__version__)'"
# 两台 PATH/python/torch 版本一致；chrony 时间同步；MASTER_PORT 与 NCCL 端口段防火墙放行
```

---

## 七、故障排查手册（症状→原因→修法）

| # | 症状 | 最可能原因 | 修法 |
|---|---|---|---|
| 1 | 卡住不动，日志无报错，最后 `torch.distributed... timeout` | rendezvous 没凑齐：NNODES 与实际节点数不符 / MASTER_ADDR 是 127.0.0.1 / 防火墙挡了 MASTER_PORT / 某节点根本没被拉起 | 检查每个节点进程是否都在（`ps -ef | grep torchrun`）；`NCCL_DEBUG=INFO`；telnet 主节点端口 |
| 2 | `world size mismatch` / `data_parallel_size=0` 类 Megatron 报错 | **TP×DP×PP ≠ NNODES×GPUS_PER_NODE**（本框架 P0-2：16 卡拓扑生成了 32 卡方案） | 核对脚本头 `规划推荐 WORLD_SIZE` 与 `NNODES×GPUS_PER_NODE` 是否相等；不等就重新规划（换小模型或改卡数） |
| 3 | 能跑但极慢，GPU 利用率锯齿状 | 跨机通信走了管理网/没走 IB | `NCCL_DEBUG=INFO` 看实际通道；修正 `NCCL_IB_HCA`/`NCCL_SOCKET_IFNAME`；跑 rccltest 验证 |
| 4 | `NCCL ... timeout` 中途报错 | 某节点掉线/慢节点超时 | 看 `--distributed-timeout-minutes`；查掉线节点的卡/网络；先跑体检 |
| 5 | 远端报 `python3: command not found` 或 `import torch` 失败 | 非交互 SSH 不加载 conda/环境模块 | 用系统 python 或把环境写进 `/etc/environment`；或生成脚本里 source 环境后再调 python |
| 6 | 节点失联时编排器整个挂死 | 已知缺陷：`launch_cluster.py` 的 `--ssh-timeout` 未生效（无 ConnectTimeout） | 先 `ssh 节点 true` 探活再拉起；等修复 |
| 7 | `bash launch_cluster.sh` 秒退 `unbound variable` | 已知 bug：模板占位变量未替换（`lowering.py:119`） | 不用它，直接 `python3 tools/launch_cluster.py ...` |
| 8 | 训练正常、保存 checkpoint 时段错误 | 海光栈已知问题 | smoke 默认关保存；正式跑前单独小规模验证存档 |
| 9 | `megatron` import 失败 / 提示 submodule | `Megatron-LM-das/3rdparty` 子模块未初始化 | `git submodule update --init --recursive` |
| 10 | 两边 loss 曲线不一致/数据疑似重复 | DP 分片失效（各 rank 喂了同样数据） | 检查 sampler 是否按全局 world_size 切；Megatron 检查 data path 一致性 |

---

## 八、单机→多机迁移检查表

把一个单机训练脚本变多机，逐项打勾：

- [ ] 拆出"身份四件套"：`NNODES/NODE_RANK/MASTER_ADDR/MASTER_PORT`（带默认值 export）
- [ ] torchrun 加 `--nnodes --node_rank --master_addr --master_port`
- [ ] 初始化用 `init_method="env://"`（写死 tcp://ip:port 的单机旧代码要改）
- [ ] 校验 `TP×DP×PP == NNODES×GPUS_PER_NODE`（写个 assert 放脚本开头）
- [ ] 数据：全局分片 + 所有节点可读（NFS/同步）
- [ ] 代码/权重/数据在所有节点**相同绝对路径**
- [ ] `--distributed-timeout-minutes` 调大（60+）
- [ ] 网络：rccltest 打满；`NCCL_IB_HCA/SOCKET_IFNAME` 与计算网一致
- [ ] SSH 免密 + 非交互 PATH 可用 + 端口放行
- [ ] 编排方式三选一（手工/mpirun/launch_cluster.py），hostfile 第一行=主节点
- [ ] checkpoint 只让 rank0（或 Megatron 的规定 rank）保存，路径共享

---

## 九、一次双机 16 卡训练的完整时序

```
T0  主节点: python3 tools/launch_cluster.py --hostfile hf --script launch_megatron_plan.sh
    │ 解析 hostfile → 2 节点，主节点=node0
    │ 并发 fork 两个 ssh 进程：
    │   ssh node0 "cd repo && NODE_RANK=0 NNODES=2 MASTER_ADDR=node0 PORT=29500 bash launch.sh"
    │   ssh node1 "cd repo && NODE_RANK=1 NNODES=2 MASTER_ADDR=node0 PORT=29500 bash launch.sh"
    ▼
T1  两台各自: source activate_megatron_hygon.sh（DTK/RCCL 环境）
    │ 各自起 torchrun --nproc_per_node=8 --nnodes=2 --node_rank=k --master_addr=node0
    ▼
T2  rendezvous：16 个进程向 node0:29500 报到（此阶段最易挂：地址错/防火墙/节点没起来）
    ▼
T3  torchrun 注入 RANK(0-15)/LOCAL_RANK(0-7)/WORLD_SIZE=16 → 运行 run_megatron_hygon.py
    │ 剥掉脚本里的静态 --rank/--world-size/--dist-url，按环境变量重注入（框架已处理）
    ▼
T4  Megatron 初始化：按 TP/PP/EP/DP 切分 16 个 rank，建进程组
    │ 跨机组走 RCCL+IB（首次集合通信会触发网络库建链，可能要几十秒）
    ▼
T5  训练循环：iteration 1..N，每步梯度 all-reduce/all-to-all 跨机往返
    ▼
T6  结束：两边进程退出 → launch_cluster 聚合退出码 → 全 0 = 成功
```

---

## 十、本仓库当前状态与已知雷区（2026-10 实测）

**已验证可用**：
- 单机规划/生成链路（128 卡 GLM-4.5-Air → TP8/DP8/PP2/EP8，24.6GB/64GB，产物自洽）
- 32 卡自动搜索（TP8/PP4/DP1，56.3GB/64GB）
- `tools/launch_cluster.py` 直接调用（dry-run 与 SSH 命令构造正确）

**已知 bug（上集群前处理）**：
1. 生成的 `launch_cluster.sh` 占位变量未替换 → 必炸，绕过=直接调 `tools/launch_cluster.py`
2. **拓扑错配**：16 卡拓扑对 105B 会生成 32 卡方案的脚本（16 进程 vs 32 rank）→ 16 卡只跑小模型
3. smoke 脚本 NNODES 默认=节点数 → 本机跑冒烟必须显式 `NNODES=1`
4. `scripts/glm45_air/train_glm45_air.sh` PP=4 与 46 层不整除 → 128 卡前改 PP=2
5. `launch_cluster.py --ssh-timeout` 未生效

**硬依赖**：
- `Megatron-LM-das/3rdparty` 三个子模块当前为空 → MEGATRON 后端必须先 `git submodule update --init --recursive`
- GLM-4.5-Air 真实微调必须走 `--use-bridge`（原生 GPT 参数表达不了 partial rotary/longrope/MTP/qkv-bias）

---

## 十一、推荐学习路径（六级闯关）

| 关卡 | 内容 | 通过标准 |
|---|---|---|
| 1 | 单机 1 进程跑通 `dist_hello.py`（`torchrun --nproc_per_node=1`） | 打印 OK |
| 2 | 单机 8 进程 `dist_hello.py` | world_size=8 |
| 3 | **双机 16 进程 `dist_hello.py`（4.1 节手搓）** | world_size=16，all_reduce=136 |
| 4 | rccltest 双机带宽 | 打满计算网 |
| 5 | 本框架双机 16 卡 Llama-3.1-8B（4.3 节） | 两节点 iteration/loss 输出 |
| 6 | 32 卡 GLM-4.5-Air（bridge 路线）→ 128 卡 | 全量微调跑通 |

每关只用上一关的"通过标准"作准入——**层级排障，不跳关**，多机问题 90% 出在第 3、4 关就能暴露的层面。
