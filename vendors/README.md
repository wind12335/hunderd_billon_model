# vendors/ —— 三家国产加速卡适配包

百亿参数稠密线共四家厂商：海光（基线，脚本在 `scripts/`＋`tools/` 仓库根）＋
本目录三家；千亿参数 EP/MoE 线当前仅海光 BW1000（范围 2026-11-05 定）。
每家子目录自包含：脚本内根目录按 `BASH_SOURCE` 相对推导，整体搬运不改路径。

差异不在各家的 README 重复，统一收敛在 `megatron_mvp/vendor_profile.py` 的
`VendorProfile`（激活脚本、运行包装器、能力位、设备前缀、后端 import、
TE 替身目录、替身卡规格）。本文件是总览与上手入口。

## 能力矩阵

| | 海光 hygon（基线） | 沐曦 maca | 摩尔 musa | 天数 corex |
| --- | --- | --- | --- | --- |
| 目标卡（替身规格） | BW1000 64G | C500 64G | S5000 80G | 天垓150 32G |
| Megatron 形态 | dcu_megatron 0.13 补丁 wheel | 厂商官方 0.12 源码直挂 | 上游 0.14.0rc2＋运行时补丁 | 上游 0.14.0rc2＋6 处幂等补丁 |
| TransformerEngine | dcu 补丁 wheel 内置 | 厂商版 TE 2.3.0 | 无 → `te_stub/` | 无 → `te_stub/` |
| sequence parallel | 允许 | 允许（有厂商 TE） | 强制关（local 实现的 torch LayerNorm 不支持） | 强制关 |
| transformer_impl | 不渲染（旧参数体系无此旗标） | 固定 `local` | 固定 `local` | 固定 `local` |
| EP/MoE（千亿线） | ✅ 唯一档位 | ❌ 搜索直接拒并留痕 | ❌ | ❌ |
| torch 设备前缀 | cuda | cuda（cu-bridge 兼容层） | musa（需显式 `import torch_musa`） | cuda（CUDA 兼容层） |
| 探测依据 | `torch.version.hip` | torch 版本串 `+metax` | `torch.musa.is_available()` | 设备名含 Iluvatar／`+corex` |

## 无厂商硬件做跨厂商纯规划

框架的搜索与渲染完全不感知厂商，只按"厂商档"取差异项。开发机上：

```bash
AUTOPLAN_VENDOR=musa AUTOPLAN_STANDIN_GPUS=8 \
python3 tools/regress_vendor.py vendors   # 例：按 8×S5000-80G 替身规划
```

- `AUTOPLAN_VENDOR=<maca|musa|corex|hygon>`：切厂商档（替身卡规格随档切换）；
  非法值报错并列出可选档。缺省按真机探测的 accelerator_type 匹配，
  未识别类型（含通用 cuda）回落海光档＝多厂商合并前的既有行为。
- `AUTOPLAN_STANDIN_GPUS=<卡数>`：替身卡数（优先于真卡）。
- 千亿 EP/MoE 规划（如 `generated/glm64_g2` 的 TP8/DP8/EP8）只在 hygon 档可复现；
  其余档位对 MoE spec 的 EP 候选直接拒绝并在 `search_result.json` 的
  `rejected_reasons` 留痕（"千亿 MoE 线当前仅海光档"）。

## 各厂商子目录

| 子目录 | 内容 | 额外项 |
| --- | --- | --- |
| `maca/` | scripts/（activate＋启动脚本）、tools/（bootstrap＋运行包装器＋环境检查） | 无 |
| `musa/` | 同上四件套＋调试脚本（debug_*、diagnose_layers、verify_backward_13b 等） | `musa_patch/`（随 import 生效的运行时补丁包）、`te_stub/`、`docker/` |
| `corex/` | 同上四件套 | `te_stub/` |

Megatron 源码获取（activate/bootstrap 按 `${本目录}/Megatron-LM` 挂载，不入库）：

- musa / corex：**网络可达时自动拉取**——目录不在位则 bootstrap 自动
  `git clone` 上游 `core_v0.14.0rc2` 干净源码（musa 施加 7 处幂等源码补丁＋
  `import musa_patch` 运行期转译；corex 施加 6 处幂等补丁），随后每次启动
  幂等跳过。内网作业机拉取失败不报错，回落人工放置；`MEGATRON_AUTO_FETCH=0`
  可关闭。多 rank 并发启动由文件锁串行化；
- maca：沐曦官方 maca-megatron-lm 0.12 源码包（bootstrap 原样挂载、不改
  源码）。**厂商私有分发、公网没有，只能人工放置**到 `vendors/maca/Megatron-LM`；
- hygon：内部 Megatron-LM-das 仓库（作业机在位），activate 三级解析。

## 已知坑（详见各家 README）

- **musa**：宿主 CPU 无 AVX-512，TE/apex 融合内核一加载即 SIGILL（硬件级崩溃，
  无法异常兜底）→ te_stub 整体屏蔽；`torch.compile` 须在 Python 启动前禁用；
  权重加载固定 `map_location="cpu"`。
- **corex**：32G 卡放不下 13B 全量微调的优化器状态（四种纯 GPU 拓扑实测全
  OOM）→ TP4/PP2＋优化器 CPU 卸载；厂商 flash-attention 训练路径数值异常，
  默认关、留开关。
- **maca**：编译 C++ 扩展须钉死系统 gcc；注意力后端环境变量不可预置，交由
  Megatron 按旗标自行设置。
- 三家新版 Megatron 参数体系不识别旧版显式 `--rank/--world-size/--dist-url`，
  由各家运行包装器统一剔除、改用 torchrun 环境变量注入。

## 回归

```bash
python3 tools/regress_vendor.py   # 海光字节级等价＋三厂商档＋musa 前端冒烟＋纯函数
python3 tools/regress_ep.py       # 千亿 EP 基线（海光档）
```
