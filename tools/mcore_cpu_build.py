#!/usr/bin/env python3
"""开发机(CPU/无 GPU)构建 mcore GLM 模型的公共脚手架。

被 tools/convert_hf_glm_to_mcore.py 与 tools/dump_mcore_keys.py 共用。
所有猴子补丁只影响本进程: mcore 建层时硬编码 CUDA 设备/RNG/apex 扩展，
与键名/数值无关，关掉后 CPU 单进程即可构建任意尺寸模型（gloo, world=1）。

构建链与集群 pretrain_gpt.py 完全一致:
  parse_args(全 GLM 旗标) -> core_transformer_config_from_args ->
  get_gpt_decoder_block_spec(use_transformer_engine=False) -> GPTModel
正式跑显式 --transformer-impl local，键名在此即完全确定。
"""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
UPSTREAM = REPO / "Megatron-LM-das" / "3rdparty" / "Megatron-LM"
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(1, str(UPSTREAM))

import sitecustomize  # noqa: F401  typing.override shim (py3.11)
import warnings

warnings.filterwarnings("ignore")

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29575")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")

import torch

torch.cuda.current_device = lambda: torch.device("cpu")
torch.cuda.set_device = lambda *a, **k: None
torch.cuda.synchronize = lambda *a, **k: None


class _DummyRngTracker:
    @contextlib.contextmanager
    def fork(self, *a, **k):
        yield

    def add(self, *a, **k):
        pass


def build_glm_model(
    *,
    num_layers: int = 46,
    hidden: int = 4096,
    heads: int = 96,
    groups: int = 8,
    kv_channels: int = 128,
    ffn_dense: int = 10944,
    moe_ffn: int = 1408,
    shared_ffn: int = 1408,
    experts: int = 128,
    vocab: int = 151552,
    moe_layer_freq: str = "[0]*1+[1]*45",
    bf16: bool = True,
    port: int = 29575,
) -> tuple:
    """返回 (model, args)。尺寸参数来自 HF 张量形状推导，见转换器。"""
    os.environ["MASTER_PORT"] = str(port)
    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group("gloo", rank=0, world_size=1)

    from megatron.core import parallel_state
    from megatron.core.tensor_parallel import random as tp_random

    tp_random.CudaRNGStatesTracker.fork = _DummyRngTracker.fork
    tp_random.CudaRNGStatesTracker.add = _DummyRngTracker.add

    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(1, 1)

    from megatron.training.arguments import parse_args

    sys.argv = [
        "build",
        "--num-layers", str(num_layers),
        "--hidden-size", str(hidden),
        "--ffn-hidden-size", str(ffn_dense),
        "--num-attention-heads", str(heads),
        "--num-query-groups", str(groups),
        "--kv-channels", str(kv_channels),
        "--seq-length", "2048",
        "--max-position-embeddings", "131072",
        "--position-embedding-type", "rope",
        "--normalization", "RMSNorm",
        "--swiglu",
        "--untie-embeddings-and-output-weights",
        "--disable-bias-linear",
        "--add-qkv-bias",
        "--rotary-base", "1000000",
        "--rotary-percent", "0.5",
        "--num-experts", str(experts),
        "--moe-router-topk", "8",
        "--moe-ffn-hidden-size", str(moe_ffn),
        "--moe-shared-expert-intermediate-size", str(shared_ffn),
        "--moe-layer-freq", moe_layer_freq,
        "--moe-aux-loss-coeff", "0.0001",
        "--moe-router-load-balancing-type", "aux_loss",
        "--moe-router-score-function", "sigmoid",
        "--moe-router-enable-expert-bias",
        "--transformer-impl", "local",
        "--tokenizer-type", "NullTokenizer",
        "--vocab-size", str(vocab),
        "--train-iters", "1",  # 仅为过校验
    ]
    args = parse_args()
    args.params_dtype = torch.bfloat16 if bf16 else torch.float32
    args.apply_rope_fusion = False          # 开发机无 TE；键名不受影响
    args.no_persist_layer_norm = True       # 开发机无 apex 走 TorchNorm；键名不受影响
    args.gradient_accumulation_fusion = False

    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
    from megatron.training.arguments import core_transformer_config_from_args

    config = core_transformer_config_from_args(args)
    assert config.add_qkv_bias and config.moe_router_enable_expert_bias
    assert config.moe_router_score_function == "sigmoid" and not config.add_bias_linear

    spec = get_gpt_decoder_block_spec(
        config,
        use_transformer_engine=False,  # --transformer-impl local
        normalization=args.normalization,
        qk_l2_norm=args.qk_l2_norm,
    )
    model = GPTModel(
        config=config,
        transformer_layer_spec=spec,
        vocab_size=vocab,
        max_sequence_length=args.max_position_embeddings,
        pre_process=True,
        post_process=True,
        fp16_lm_cross_entropy=False,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
        rope_scaling=args.use_rope_scaling,
        mtp_block_spec=None,
    )
    return model, args
