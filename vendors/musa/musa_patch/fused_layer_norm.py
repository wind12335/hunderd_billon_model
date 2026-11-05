# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# autoparallel MUSA 改编: 纯 torch 的 LayerNorm，替换/就地补丁 megatron 的 apex
# FusedLayerNorm，因为本机 apex 只有纯 python(无 fused_layer_norm_cuda 扩展)。
#
# 注意: __init__/forward 不能依赖 super() 的 __class__ 闭包 —— 它们会被
# 就地绑定到 megatron 原始 FusedLayerNorm 类上，在原始类实例上调用，
# super(本类, 原始类实例) 会失败。所以统一用 torch.nn.Module.__init__(self)
# 显式初始化，并导出独立函数 _init/_forward 供两种用法共用。

import numbers

import torch
import torch_musa
from torch.nn import init
from torch.nn.parameter import Parameter

from megatron.core.transformer import TransformerConfig


def _init(
    self,
    config: TransformerConfig,
    hidden_size,
    eps: float = 1e-5,
    persist_layer_norm: bool = True,
    zero_centered_gamma: bool = False,
    normalization: str = "LayerNorm",  # included to match TE interface
):
    torch.nn.Module.__init__(self)  # 不用 super()，便于就地补丁原始类
    self.config = config
    self.zero_centered_gamma = self.config.layernorm_zero_centered_gamma

    if self.config.normalization == "LayerNorm":
        self.norm_impl = torch.layer_norm
    elif self.config.normalization == "RMSNorm":
        self.norm_impl = torch.rms_norm
    else:
        raise ValueError(f'({self.config.normalization}) is not supported in FusedLayerNorm')

    if isinstance(hidden_size, numbers.Integral):
        hidden_size = (hidden_size,)
    self.hidden_size = hidden_size
    self.eps = eps
    self.weight = Parameter(torch.Tensor(*hidden_size))
    self.bias = Parameter(torch.Tensor(*hidden_size)) if self.config.normalization == "LayerNorm" else None
    self.reset_parameters()
    self.sequence_parallel = self.config.sequence_parallel

    # set sequence parallelism flag on weight and bias parameters
    setattr(self.weight, 'sequence_parallel', self.sequence_parallel)
    if self.config.normalization == "LayerNorm":
        setattr(self.bias, 'sequence_parallel', self.sequence_parallel)

    # 兼容原类可能被外部访问的字段
    self.persist_layer_norm = persist_layer_norm


def _reset_parameters(self):
    if self.zero_centered_gamma:
        init.zeros_(self.weight)
        if self.config.normalization == "LayerNorm":
            init.zeros_(self.bias)
    else:
        init.ones_(self.weight)
        if self.config.normalization == "LayerNorm":
            init.zeros_(self.bias)


def _forward(self, input):
    weight = self.weight + 1 if self.zero_centered_gamma else self.weight
    if self.config.normalization == "LayerNorm":
        output = self.norm_impl(input, self.hidden_size, weight, self.bias, self.eps)
    else:
        output = self.norm_impl(input, self.hidden_size, weight, self.eps)
    return output


class FusedLayerNorm(torch.nn.Module):

    """Layer Norm，纯 torch 实现（替代 apex CUDA fused 版本）。"""

    def __init__(self, *args, **kwargs):
        _init(self, *args, **kwargs)

    def reset_parameters(self):
        _reset_parameters(self)

    def forward(self, input):
        return _forward(self, input)


import megatron.core.fusions.fused_layer_norm as _mfln
_orig_fused_ln = _mfln.FusedLayerNorm
_mfln.FusedLayerNorm = FusedLayerNorm
_mfln.HAVE_FUSED_LAYER_NORM = False
# 就地补丁原始类: 本机 apex 包可 import(纯 python) 但 CUDA 扩展未编译,
# megatron 导入树可能在子模块执行前就把原始类绑进 transformer_block/
# gpt_layer_specs 等模块的别名(FusedLayerNorm/LNImpl/LayerNormImpl)。
# 替换原类方法后，无论哪个别名持有它都走纯 torch 的 torch.layer_norm 路径。
if getattr(_orig_fused_ln, "__module__", "") != "musa_patch.fused_layer_norm":
    _orig_fused_ln.__init__ = _init
    _orig_fused_ln.reset_parameters = _reset_parameters
    _orig_fused_ln.forward = _forward
