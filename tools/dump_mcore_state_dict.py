#!/usr/bin/env python3
"""枚举 mcore local transformer 的 GPTModel state_dict 结构。

用于编写 HF→mcore 权重转换器：先确认真实训练时 mcore 模型的确切 key 与 shape，
再据此构造映射。默认用缩小版模型（快），shape 反映相对结构即可。
"""

from __future__ import annotations

import sys
from pathlib import Path

_SELF_DIR = Path(__file__).resolve().parent
if str(_SELF_DIR) not in sys.path:
    sys.path.insert(0, str(_SELF_DIR))

from megatron_corex_bootstrap import bootstrap  # noqa: E402

bootstrap(verbose=False)

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from megatron.core import parallel_state  # noqa: E402
from megatron.core.models.gpt import GPTModel  # noqa: E402
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec  # noqa: E402
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed  # noqa: E402
from megatron.core.transformer import TransformerConfig  # noqa: E402


def main() -> int:
    dist.init_process_group("gloo", rank=0, world_size=1, init_method="tcp://127.0.0.1:29993")
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)

    # 手动建模型前必须注册 RNG tracker（正常训练由 megatron.initialize 负责）。
    model_parallel_cuda_manual_seed(0)

    config = TransformerConfig(
        num_layers=2,
        hidden_size=512,
        ffn_hidden_size=1382,
        num_attention_heads=8,
        num_query_groups=8,
        normalization="RMSNorm",
        layernorm_epsilon=1e-6,
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        add_bias_linear=False,
        params_dtype=torch.bfloat16,
    )
    # 本 Megatron 快照 transformer_block.py 引用但 dataclass 缺失的字段（MoE offload，非 frozen 可直接赋值）。
    if not hasattr(config, "offload_moe_fc1_input"):
        config.offload_moe_fc1_input = False
    model = GPTModel(
        config,
        transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=69104,
        max_sequence_length=4096,
        position_embedding_type="rope",
        rotary_base=10000,
    )
    model.cuda().to(torch.bfloat16)
    model.eval()

    # mcore 部分 Linear 权重惰性初始化，先跑一次 forward 触发创建再枚举。
    device = torch.device("cuda:0")
    seq_len = 16
    input_ids = torch.randint(0, 69104, (1, seq_len), dtype=torch.long, device=device)
    position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
    # 显式因果 mask（True=可 attend），绕开 fused_softmax 里硬编码 device="cuda" 的默认路径。
    attn_mask = torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
    ).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        model(input_ids=input_ids, position_ids=position_ids, attention_mask=attn_mask)

    for k, v in model.state_dict().items():
        if v is None:
            print(f"{k} None")
        else:
            print(f"{k} {tuple(v.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
