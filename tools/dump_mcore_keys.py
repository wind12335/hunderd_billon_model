#!/usr/bin/env python3
"""在开发机上按集群同一条代码路径构建迷你 GLM，导出权威键名清单。

用途: HF->mcore 权重转换器(tools/convert_hf_glm_to_mcore.py)的输出键名必须与
训练时 GPTModel.sharded_state_dict() 完全一致。与其从源码推断，不如走集群
pretrain_gpt.py 的同一条构建链（gpt_builders.gpt_builder + core_transformer_config_from_args
+ get_gpt_decoder_block_spec），把 GLM-4.5-Air 的结构旗标（MoE/GQA/sigmoid 路由
+expert_bias/--add-qkv-bias/--disable-bias-linear/--transformer-impl local）缩小尺寸
真建一次模型，打印键名与全局形状。CPU/gloo 单进程即可，不需要 GPU。

正式跑将显式传 --transformer-impl local（键名确定、不依赖 TE 的 MoE 分组 GEMM）。

注意: 开发机 Python 3.11 缺 typing.override，先导入 tools/sitecustomize.py 补丁。
"""

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
os.environ.setdefault("MASTER_PORT", "29572")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")

import torch
from megatron.core import parallel_state
from megatron.training.arguments import parse_args

# 无 GPU 构建的猴子补丁：mcore 在建层时硬编码 CUDA 设备/RNG；键名与这两者无关。
torch.cuda.current_device = lambda: torch.device("cpu")
torch.cuda.set_device = lambda *a, **k: None
torch.cuda.synchronize = lambda *a, **k: None


class _DummyRngTracker:
    import contextlib

    @contextlib.contextmanager
    def fork(self, *a, **k):
        yield

    def add(self, *a, **k):
        pass


from megatron.core.tensor_parallel import random as _tp_random

# 各模块按名字 import 了 tracker，直接补类方法让所有引用同时生效
_tp_random.CudaRNGStatesTracker.fork = _DummyRngTracker.fork
_tp_random.CudaRNGStatesTracker.add = _DummyRngTracker.add

# 与正式 render 对齐的 GLM 结构旗标，尺寸缩小；vocab 用真值暴露 padding 问题。
sys.argv = [
    "dump",
    "--num-layers", "2",                       # 正式 46（层0稠密+层1MoE 已覆盖两种形态）
    "--hidden-size", "64",
    "--ffn-hidden-size", "128",
    "--num-attention-heads", "4",
    "--num-query-groups", "2",
    "--kv-channels", "16",
    "--seq-length", "4",
    "--max-position-embeddings", "2048",
    "--position-embedding-type", "rope",
    "--normalization", "RMSNorm",
    "--swiglu",
    "--untie-embeddings-and-output-weights",
    "--disable-bias-linear",
    "--add-qkv-bias",
    "--rotary-base", "1000000",
    "--rotary-percent", "0.5",
    "--num-experts", "4",
    "--moe-router-topk", "2",
    "--moe-ffn-hidden-size", "32",
    "--moe-shared-expert-intermediate-size", "32",
    "--moe-layer-freq", "[0]*1+[1]*1",
    "--moe-aux-loss-coeff", "0.0001",
    "--moe-router-load-balancing-type", "aux_loss",
    "--moe-router-score-function", "sigmoid",
    "--moe-router-enable-expert-bias",
    "--transformer-impl", "local",
    "--tokenizer-type", "NullTokenizer",
    "--vocab-size", "151552",
]


def main() -> None:
    torch.distributed.init_process_group("gloo", rank=0, world_size=1)
    parallel_state.initialize_model_parallel(1, 1)

    args = parse_args()
    args.params_dtype = torch.float32  # 正式流程里由 initialize_megatron 填的默认值
    args.apply_rope_fusion = False     # 开发机无 TE；键名不受影响
    args.no_persist_layer_norm = True   # 开发机无 apex 走 TorchNorm；键名不受影响
    args.gradient_accumulation_fusion = False  # 同上，apex 扩展缺失

    # 与集群 gpt_builder() 内部完全一致的构建链（MoE 走 decoder block spec）
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
    from megatron.training.arguments import core_transformer_config_from_args

    config = core_transformer_config_from_args(args)
    print(f"config 校验: add_qkv_bias={config.add_qkv_bias} "
          f"score_fn={config.moe_router_score_function} "
          f"expert_bias={config.moe_router_enable_expert_bias} "
          f"add_bias_linear={config.add_bias_linear}")

    spec = get_gpt_decoder_block_spec(
        config,
        use_transformer_engine=False,  # --transformer-impl local
        normalization=args.normalization,
        qk_l2_norm=args.qk_l2_norm,
    )
    model = GPTModel(
        config=config,
        transformer_layer_spec=spec,
        vocab_size=151552,
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

    print("=== sharded_state_dict 键（转换器输出必须精确匹配这套名）===")
    for key, st in sorted(model.sharded_state_dict().items()):
        gs = getattr(st, "global_shape", None)
        if gs is None:
            gs = getattr(getattr(st, "data", None), "global_shape", None)
        shape = tuple(gs) if gs is not None else "<non-tensor>"
        dtype = getattr(getattr(st, "data", st), "dtype", None)
        print(f"{key}\t{shape}\t{dtype}")

    print("=== state_dict 非张量项 ===")
    for key, t in model.state_dict().items():
        if not torch.is_tensor(t):
            print(f"{key}\t<extra_state {type(t).__name__}>")


if __name__ == "__main__":
    main()
