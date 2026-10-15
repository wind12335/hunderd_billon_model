# GLM-4.5-Air 105B 规格 stub（spec-only，无权重）

这个目录只包含 GLM-4.5-Air 的 `config.json`（结构与 zai-org/GLM-4.5-Air 对齐）
和一个占位权重文件，**没有任何真实权重**。

用途：
1. 供 autoparallel 的 `detect_hf_model` / `load_model_spec` 做结构探测与并行规划
   （规划只需要 config，不需要权重）。
2. 双机 16 卡 / 128 卡的并行方案搜索、显存可行性估算、启动脚本生成。

要做真实全量微调时，请把 HF 权重完整下载到独立目录，例如：

```bash
huggingface-cli download zai-org/GLM-4.5-Air --local-dir /data/models/glm-4.5-air
```

然后把 `Config(model_path=...)` 指向真实权重目录。

关键结构参数（glm4_moe / Glm4MoeForCausalLM）：
- 46 层，其中前 1 层稠密（first_k_dense_replace=1），后 45 层 MoE
- 128 个路由专家 top-8 + 1 个共享专家，moe_intermediate_size=1408
- hidden 4096，attention 96 头 / 8 KV 头（GQA），head_dim=128（q_proj 上采样）
- partial rotary 0.5（旋转位置编码作用于每头前 64 维）
- vocab 151552，untied lm_head
- 总参数 ≈ 105~112B，每 token 激活 ≈ 12B
