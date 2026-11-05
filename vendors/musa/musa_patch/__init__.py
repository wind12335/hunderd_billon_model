"""autoparallel 定制版 musa_patch（无 TransformerEngine 依赖）。

由摩尔线程官方 megatron-lm-musa-patch 的 musa_patch 改编：
1. 保留完整的 patch_after_import_torch() —— torch.cuda.* → torch.musa.* 的
   CUDA 兼容垫片，这是整套框架能在 MUSA 上跑的关键。
2. patch_before_import_megatron() 砍掉所有 TransformerEngine 依赖：
   MT-TransformerEngine 的 .so 用 AVX-512 指令集编译，而本机 CPU
   (海光 Hygon C86-4G) 没有 AVX-512，import 即 SIGILL(硬崩溃)，
   try/except 无法捕获。因此本地训练必须用 --transformer-impl local，
   不需要 TE，也不引入 optimizer/router/token_dispatcher 等 TE 相关补丁。
3. 对 CPU 版 torch（torch.cuda 未编译）可能缺失的属性做 hasattr 防护。
"""

import os
import sys
import torch
import torch.utils
import torch.utils.data
import torch_musa
from contextlib import nullcontext


def patch_before_import_megatron():
    # 屏蔽 Apex 的 amp_C 扩展：它用 -march=native 在支持 AVX-512 的机器上编译
    # （含 EVEX 前缀指令如 vpgatherqq，寄存器用 ymm/xmm 不用 zmm，zmm 检测不到），
    # 本机海光 C86 无 AVX-512，import 能成功但调用 pybind 函数即 SIGILL(硬崩溃，
    # 无法 try/except)。把 sys.modules['amp_C'] 置 None 令 `import amp_C` 抛
    # ImportError，megatron 的 try/except 便自动落入纯 torch 的
    # local_multi_tensor_* 实现(megatron/core/utils.py)，功能等价。
    import sys
    sys.modules['amp_C'] = None
    # fused_weight_gradient_mlp_cuda 同理(AVX-512 编译)。正常由
    # gradient_accumulation_fusion=False 关闭；这里再兜底，万一误开该选项，
    # 至少得到可诊断的 ImportError 而不是硬崩溃。
    sys.modules['fused_weight_gradient_mlp_cuda'] = None
    # apex 的 FusedAdam/FusedSGD 强依赖 amp_C（构造时调 amp_C.multi_tensor_adam，
    # step 时同样调用，都会 SIGILL）。阻断 apex.optimizers 下 fused 子模块，
    # 让 megatron 的 `from apex.optimizers import FusedAdam` 抛 ImportError，
    # 自动落回 torch.optim.AdamW（megatron/core/optimizer/__init__.py 注释即
    # 明言 Apex FusedAdam 是 torch AdamW 的 drop-in 替代）。
    sys.modules['apex.optimizers.fused_adam'] = None
    sys.modules['apex.optimizers.fused_sgd'] = None

    # 用 musa 的设备架构替代 cuda 版本探测
    def patched_get_device_arch_version():
        return torch.musa.get_device_properties(torch.musa.current_device()).major
    import megatron.training.utils
    megatron.training.utils.get_device_arch_version = patched_get_device_arch_version
    import importlib
    importlib.reload(megatron.training.arguments)

    # 与 TE 无关的核心补丁子模块（已确认不含 transformer_engine import）。
    # fused_layer_norm 子模块会就地补丁原始 FusedLayerNorm 类(纯 torch 的
    # torch.layer_norm 路径)，无论 megatron 以什么别名持有它都有效。
    from . import fused_layer_norm
    from . import dot_product_attention
    from . import checkpointing
    from . import training
    from . import rotary_pos_embedding
    from . import p2p_communication
    from . import fused_bias_swiglu
    from . import arguments

    # 注意：原版在这里会 import transformer_engine.pytorch.utils /
    # attention / optimizer / router / token_dispatcher / recomupte_variance
    # / fault_tolerance_epx / core_pipeline_parallel_schedules /
    # yarn_rotary_pos_embedding / ckpt_torch_dist。
    # 它们全部依赖 MT-TransformerEngine，在本机(无AVX512)会 SIGILL，故跳过。

    # 禁用不支持的 feature
    def set_jit_fusion_options():
        pass
    import megatron.training.initialize
    megatron.training.training.set_jit_fusion_options = set_jit_fusion_options
    megatron.training.initialize.set_jit_fusion_options = set_jit_fusion_options
    # 禁用 fused_kernels
    import megatron.legacy.fused_kernels
    megatron.legacy.fused_kernels.load = lambda args: None
    # _compile_dependencies 只保留数据集 C++ helpers 编译（纯 CPU，helpers_cpp），
    # 跳过 fused kernels 等需要 CUDA/apex 扩展的部分（本机无 CUDA，编译必然失败）。
    # 即使 --mock-data 也会走 GPTDataset._build_document_sample_shuffle_indices
    # import helpers_cpp，所以必须编译。
    def _compile_dependencies():
        import torch.distributed as dist

        if not dist.is_initialized() or dist.get_rank() == 0:
            try:
                from megatron.core.datasets.utils import compile_helpers

                compile_helpers()
            except Exception as exc:
                print(f"[musa_patch] warning: compile_helpers failed: {exc}", flush=True)
        if dist.is_initialized():
            dist.barrier()
    megatron.training.initialize._compile_dependencies = _compile_dependencies


def patch_after_import_torch():
    # 1. Patch for torch.xxx —— 与摩尔线程官方 musa_patch 完全一致
    torch.cuda.is_available = torch.musa.is_available
    torch.cuda.current_device = lambda: f'musa:{torch.musa.current_device()}'
    torch.cuda.device_count = torch.musa.device_count
    torch.cuda.set_device = torch.musa.set_device
    torch.cuda.DoubleTensor = torch.musa.DoubleTensor
    torch.cuda.FloatTensor = torch.musa.FloatTensor
    torch.cuda.LongTensor = torch.musa.LongTensor
    torch.cuda.HalfTensor = torch.musa.HalfTensor
    torch.cuda.BFloat16Tensor = torch.musa.BFloat16Tensor
    torch.cuda.IntTensor = torch.musa.IntTensor
    torch.cuda.synchronize = torch.musa.synchronize
    torch.cuda.get_rng_state = torch.musa.get_rng_state
    torch.cuda.set_rng_state = torch.musa.set_rng_state
    torch.cuda.random.get_rng_state = torch.musa.get_rng_state
    torch.cuda.synchronize = torch.musa.synchronize
    torch.cuda.empty_cache = torch.musa.empty_cache
    torch.Tensor.cuda = torch.Tensor.musa
    # Tensor.to() 设备钩子：框架代码常写 .to("cuda") / .to(device=torch.device('cuda'))
    # 而 CPU 版 torch 不认识 cuda device 类型，这里把 cuda->musa 再转交给 MUSA 的 to。
    _orig_tensor_to = torch.Tensor.to
    def _musa_tensor_to(self, *args, **kwargs):
        new_args = []
        for arg in args:
            if isinstance(arg, str) and arg.startswith("cuda"):
                arg = arg.replace("cuda", "musa")
            elif isinstance(arg, torch.device) and arg.type == "cuda":
                arg = torch.device("musa", arg.index)
            new_args.append(arg)
        if "device" in kwargs:
            v = kwargs["device"]
            if isinstance(v, str) and v.startswith("cuda"):
                kwargs["device"] = v.replace("cuda", "musa")
            elif isinstance(v, torch.device) and v.type == "cuda":
                kwargs["device"] = torch.device("musa", v.index)
        return _orig_tensor_to(self, *new_args, **kwargs)
    torch.Tensor.to = _musa_tensor_to
    torch.cuda.manual_seed = torch.musa.manual_seed
    torch.cuda.Event = torch.musa.Event
    torch.cuda.Stream = torch.musa.Stream
    torch.cuda.stream = torch.musa.stream
    torch.cuda.get_device_properties = torch.musa.get_device_properties
    # add torch.musa.current_devce() to activate torch.musa.default_generators
    d = torch.musa.current_device()
    # torch.musa.default_generators 是 int 索引的 tuple，而 Megatron 的
    # tensor_parallel/random.py 会用 torch.cuda.current_device()(返回字符串
    # 'musa:N') 或 torch.device 作索引。包一层代理兼容 int/str/device 三种键，
    # 否则 `default_generators['musa:0']` 会 TypeError。
    class _MusaDefaultGenerators:
        def __init__(self, gens):
            self._gens = gens

        def __getitem__(self, key):
            if isinstance(key, str):
                key = int(key.rsplit(":", 1)[-1])
            elif isinstance(key, torch.device):
                key = key.index if key.index is not None else torch.musa.current_device()
            return self._gens[key]

        def __iter__(self):
            return iter(self._gens)

        def __len__(self):
            return len(self._gens)

    torch.cuda.default_generators = _MusaDefaultGenerators(torch.musa.default_generators)
    # torch.cuda.amp = torch.musa.amp
    # Memory
    torch.cuda.memory_allocated = torch.musa.memory_allocated
    torch.cuda.max_memory_allocated = torch.musa.max_memory_allocated
    torch.cuda.memory_reserved = torch.musa.memory_reserved
    torch.cuda.max_memory_reserved = torch.musa.max_memory_reserved
    if hasattr(torch.cuda.memory, "_record_memory_history"):
        torch.cuda.memory._record_memory_history = torch.musa.memory._record_memory_history
    if hasattr(torch.cuda.memory, "_snapshot"):
        torch.cuda.memory._snapshot = torch.musa.memory._snapshot

    # 替换 lazy_call 避免 CPU 内存泄漏
    torch.cuda._lazy_call = torch.musa.core._lazy_init._lazy_call
    torch.cuda._lazy_init = torch.musa.core._lazy_init._lazy_init

    # 2. Patch for torch args related to cuda/musa
    def hook_cuda_device(device):
        if isinstance(device, str) and device.startswith("cuda"):
            return device.replace("cuda", "musa")
        if isinstance(device, torch.device) and device.type == "cuda":
            return torch.device("musa", device.index)
        return device

    def maybe_hook_cuda_args(args, kwargs):
        new_args = []
        for arg in args:
            new_args.append(hook_cuda_device(arg))
        if "device" in kwargs:
            v = kwargs["device"]
            kwargs['device'] = hook_cuda_device(v)
        return tuple(new_args), kwargs

    original_full = torch.full
    def patched_full(*args, **kwargs):
        args, kwargs = maybe_hook_cuda_args(args, kwargs)
        result = original_full(*args, **kwargs)
        return result
    torch.full = patched_full

    original_tensor = torch.tensor
    def patched_tensor(*args, **kwargs):
        # 同时兼容 'cuda'/'cuda:N' 字符串和 torch.device('cuda', N)
        args, kwargs = maybe_hook_cuda_args(args, kwargs)
        result = original_tensor(*args, **kwargs)
        return result
    torch.tensor = patched_tensor

    orig_type = torch.Tensor.type
    def musa_type(*args, **kwargs):
        result = orig_type(*args, **kwargs)
        if isinstance(result, str):
            result = result.replace("musa", "cuda")
            # 私有后端(musa)的无参数 type() 返回 'torch.BFloat16Tensor'
            # （不带设备标记），而 Megatron Float16Optimizer 用
            # 'torch.cuda.BFloat16Tensor' 精确匹配。这里给 musa 张量补上
            # 'torch.cuda.' 前缀。
            if args and getattr(args[0], "is_musa", False):
                if result.startswith("torch.") and not result.startswith("torch.cuda."):
                    result = result.replace("torch.", "torch.cuda.", 1)
        return result
    torch.Tensor.type = musa_type

    original_zeros = torch.zeros
    def patched_zeros(*args, **kwargs):
        args, kwargs = maybe_hook_cuda_args(args, kwargs)
        result = original_zeros(*args, **kwargs)
        return result
    torch.zeros = patched_zeros

    original_empty = torch.empty
    def patched_empty(*args, **kwargs):
        args, kwargs = maybe_hook_cuda_args(args, kwargs)
        result = original_empty(*args, **kwargs)
        return result
    torch.empty = patched_empty

    original_is_cuda = torch.Tensor.is_cuda
    def always_cuda(self):
        return self.is_musa
    torch.Tensor.is_cuda = property(always_cuda)

    # 3. Patch for nccl/mccl
    origin_init_process_group = torch.distributed.init_process_group
    def patched_init_process_group(*args, **kwargs):
        if 'backend' in kwargs and kwargs['backend'] == 'nccl':
            kwargs['backend'] = 'mccl'
        result = origin_init_process_group(*args, **kwargs)
        return result
    torch.distributed.init_process_group = patched_init_process_group

    # 4. disable nvtx
    if hasattr(torch.cuda, "nvtx"):
        def _pass_pvtx(*args, **kwargs):
            return
        torch.cuda.nvtx.range_push = _pass_pvtx
        torch.cuda.nvtx.range_pop = _pass_pvtx

    # 5. disable torch.compile：MUSA 下 dynamo 追踪会在
    #    torch._dynamo.device_interface 崩溃(musa 设备字符串作列表索引)，
    #    且本机 CPU 无 AVX-512。@torch.compile 装饰函数退化为普通执行。
    os.environ["NVTE_TORCH_COMPILE"] = "0"
    os.environ["TORCHDYNAMO_DISABLE"] = "1"

    def get_device_capability_musa():
        # TODO(@ai-dist-infra): check device capability for MUSA backend
        if torch.version.__version__ > "2.5.0":
            return (8, 3)
        return [8, 3]
    torch.cuda.get_device_capability = get_device_capability_musa


def py_patch():
    if sys.version_info >= (3.9, 0):
        return
    import math
    def lcm(a, b):
        return abs(a * b) // math.gcd(a, b)
    math.lcm = lcm
    return


# Apply patch
py_patch()
patch_after_import_torch()
patch_before_import_megatron()

# 临时调试开关: MUSA_AUTODETECT_NAN=1 时启用 torch.autograd 的 NaN 检测，
# backward 产生 NaN 时会抛出带算子名的异常，用于定位 MUSA bf16 下哪些
# 算子 backward 输出 NaN。
if os.environ.get("MUSA_AUTODETECT_NAN", "0") == "1":
    torch.autograd.set_detect_anomaly(True)
    print("[musa_patch] torch.autograd.detect_anomaly enabled", flush=True)
