#!/usr/bin/env python3
"""单卡编译验证：真实 13B checkpoint 在 MUSA 上的前向 + 反向。

任务 5 的第一道闸门。证明训练路径在 MUSA 上打通：
  1. 模型构建 + strict 加载 ckpt_13b（按头交错 qkv 布局）；
  2. forward 得到有限 loss；
  3. backward 梯度全有限、grad-norm 落在合理范围（bf16 反向在 MUSA 上
     是此前从未验证过的一环，MUSA_AUTODETECT_NAN 即为此准备）。

注意：本脚本不建优化器（13B 的 bf16 优化器状态 fp32 单卡要 ~157GB，
单卡放不下），所以它不是完整训练，只验证 模型构建/加载/前向/反向。
完整训练（含优化器）由 8 卡 DP=8 + distributed optimizer 承担。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

_SELF_DIR = Path(__file__).resolve().parent
if str(_SELF_DIR) not in sys.path:
    sys.path.insert(0, str(_SELF_DIR))

from megatron_musa_bootstrap import bootstrap  # noqa: E402

bootstrap(verbose=False)

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from megatron.core import parallel_state  # noqa: E402
from megatron.core.models.gpt import GPTModel  # noqa: E402
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec  # noqa: E402
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed  # noqa: E402
from megatron.core.transformer import TransformerConfig  # noqa: E402

CKPT = "/ws/musa_framework/generated/ckpt_13b/iter_0000001/mp_rank_00/model_optim_rng.pt"
SEQ = 64
VOCAB = 69104
HIDDEN = 5120
FFN = 13824
NLAYERS = 40
NHEADS = 40


def main() -> int:
    dist.init_process_group("gloo", rank=0, world_size=1, init_method="tcp://127.0.0.1:29995")
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(0)

    config = TransformerConfig(
        num_layers=NLAYERS,
        hidden_size=HIDDEN,
        ffn_hidden_size=FFN,
        num_attention_heads=NHEADS,
        num_query_groups=NHEADS,
        normalization="RMSNorm",
        layernorm_epsilon=1e-6,
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        add_bias_linear=False,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        params_dtype=torch.bfloat16,
    )
    if not hasattr(config, "offload_moe_fc1_input"):
        config.offload_moe_fc1_input = False
    model = GPTModel(
        config,
        transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=VOCAB,
        max_sequence_length=4096,
        position_embedding_type="rope",
        rotary_base=10000,
    )
    model.cuda().to(torch.bfloat16).train()

    ret = model.load_state_dict(
        torch.load(CKPT, map_location="cpu", weights_only=True)["model"], strict=True
    )
    assert not ret.missing_keys and not ret.unexpected_keys, ret
    print("[gate] strict 加载 ckpt_13b 通过", flush=True)

    torch.manual_seed(1234)
    input_ids = torch.randint(0, VOCAB, (1, SEQ), dtype=torch.long).to("musa:0")
    pos = torch.arange(SEQ, dtype=torch.long, device="musa:0").unsqueeze(0)
    mask = torch.tril(torch.ones(SEQ, SEQ, dtype=torch.bool, device="musa:0")).unsqueeze(0).unsqueeze(0)

    # 前向
    logits = model(input_ids=input_ids, position_ids=pos, attention_mask=mask)
    vocab = logits.shape[-1]
    flat = logits.reshape(-1, vocab).float()
    labels = input_ids.reshape(-1)
    loss = F.cross_entropy(flat, labels, ignore_index=0, reduction="mean")
    print(f"[gate] forward loss = {loss.item():.6f}  finite={bool(torch.isfinite(loss))}", flush=True)

    # 反向
    model.zero_grad(set_to_none=True)
    loss.backward()

    gnorm = 0.0
    all_finite = True
    grad_count = 0
    for p in model.parameters():
        if p.grad is None:
            continue
        grad_count += 1
        g = p.grad.float()
        if not torch.isfinite(g).all():
            all_finite = False
        gnorm += (g * g).sum().item()
    gnorm = math.sqrt(gnorm)
    print(
        f"[gate] grad 参数数={grad_count}  grad_norm={gnorm:.6f}  all_finite={all_finite}",
        flush=True,
    )

    ok = bool(torch.isfinite(loss)) and all_finite and gnorm > 0.0 and gnorm < 1e5
    print("RESULT:", "PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
