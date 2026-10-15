"""PyTorch backend 的 HF LLaMA 结构 DP+TP 运行时。

当前实现目标：
1. 读取 Hugging Face LLaMA 配置（结构超参）。
2. 按标准 Megatron-style TP 规则构造并行模块。
3. 默认随机初始化权重（与 Megatron 对齐）；可选加载真实 HF checkpoint。
4. 用 PyTorch distributed + DDP + 自定义 TP 模块执行最小训练闭环。
5. 在单卡环境下提供 planning 配置的静态 inspect-only 校验入口。

当前约束：
1. 只支持标准 DP+TP 公共策略域。
2. smoke_test 在卡数不足时可能截断层数；卡数覆盖规划时用完整层数。
3. 规划脚本保留完整层数与完整并行方案，用于后续多卡正式执行。
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel


def _load_runtime_config(path: str) -> dict[str, Any]:
    """读取 lowering 生成的运行配置。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _resolve_megatron_data_prefix(data_path: str | Path) -> Path:
    """把用户传入的 data_path 解析成 Megatron IndexedDataset 前缀。"""
    path = Path(data_path).expanduser().resolve()
    if path.suffix in {".bin", ".idx"}:
        return path.with_suffix("")
    return path


def _load_megatron_token_stream(data_path: str | Path) -> np.ndarray:
    """读取与 Megatron 相同的 wikitext2 indexed dataset，拼成一维 token 流。"""
    from megatron.core.datasets.indexed_dataset import IndexedDataset

    prefix = str(_resolve_megatron_data_prefix(data_path))
    dataset = IndexedDataset(prefix, multimodal=False, mmap=True)
    document_count = len(dataset)
    if document_count <= 0:
        raise RuntimeError(f"Megatron indexed dataset 为空: {prefix}")

    parts: list[np.ndarray] = []
    total_tokens = 0
    for document_index in range(document_count):
        document_tokens = np.asarray(dataset[document_index], dtype=np.int64)
        if document_tokens.size == 0:
            continue
        parts.append(document_tokens)
        total_tokens += int(document_tokens.size)

    if total_tokens <= 0 or not parts:
        raise RuntimeError(f"Megatron indexed dataset 未读到有效 token: {prefix}")

    token_stream = np.concatenate(parts, axis=0)
    return token_stream


def _build_input_ids_batch(
    *,
    token_stream: np.ndarray | None,
    vocab_size: int,
    micro_batch_size: int,
    seq_length: int,
    iteration: int,
    dp_rank: int,
    dp_size: int,
    device: torch.device,
) -> torch.Tensor:
    """构造当前 step 的 input_ids。

    - 有真实 token 流时：按 (dp_rank, iteration) 顺序切片，与 Megatron 同源数据。
    - 无真实数据时：回退到随机 token，保持旧的 mock 行为。
    """
    if token_stream is None or token_stream.size < 2:
        return torch.randint(
            low=0,
            high=vocab_size,
            size=(micro_batch_size, seq_length),
            device=device,
        )

    # 每个 micro-batch 需要连续 seq_length 个 token。
    # 不同 DP rank 错开起点，避免完全读同一段文本。
    tokens_per_step = micro_batch_size * seq_length
    stream_length = int(token_stream.size)
    base_offset = (
        (iteration * dp_size + dp_rank) * tokens_per_step
    ) % max(stream_length - 1, 1)

    batch = np.empty((micro_batch_size, seq_length), dtype=np.int64)
    for batch_index in range(micro_batch_size):
        start = (base_offset + batch_index * seq_length) % stream_length
        end = start + seq_length
        if end <= stream_length:
            window = token_stream[start:end]
        else:
            # 越过末尾时首尾拼接，保证始终凑满 seq_length。
            window = np.concatenate(
                (token_stream[start:], token_stream[: end - stream_length]),
                axis=0,
            )
        batch[batch_index] = window

    return torch.as_tensor(batch, device=device, dtype=torch.long)


def _pick_backend() -> str:
    """根据当前设备环境选择分布式 backend。"""
    if torch.cuda.is_available():
        return "nccl"
    return "gloo"


def _resolve_dtype(name: str) -> torch.dtype:
    """把字符串 dtype 转成 torch dtype。"""
    normalized = str(name).lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    return torch.float32


def _all_reduce_tensor(
    tensor: torch.Tensor,
    group: dist.ProcessGroup | None,
    op: dist.ReduceOp.RedOpType = dist.ReduceOp.SUM,
) -> torch.Tensor:
    """对张量执行普通 all-reduce。"""
    if group is None:
        return tensor
    reduced = tensor.clone()
    dist.all_reduce(reduced, op=op, group=group)
    return reduced


class _CopyToTensorParallelRegion(torch.autograd.Function):
    """Megatron 语义的 copy-to-tp：前向直通，反向 all-reduce。"""

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, group: dist.ProcessGroup | None) -> torch.Tensor:
        ctx.group = group
        return tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if ctx.group is None:
            return grad_output, None
        reduced = grad_output.contiguous()
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM, group=ctx.group)
        return reduced, None


class _ReduceFromTensorParallelRegion(torch.autograd.Function):
    """Megatron 语义的 reduce-from-tp：前向 all-reduce，反向直通。"""

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, group: dist.ProcessGroup | None) -> torch.Tensor:
        if group is None:
            return tensor
        reduced = tensor.contiguous().clone()
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM, group=group)
        return reduced

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None


class _VocabParallelCrossEntropy(torch.autograd.Function):
    """支持 `vocab_shard` 的自定义交叉熵，显式定义前后向通信语义。"""

    @staticmethod
    def forward(
        ctx,
        local_logits: torch.Tensor,
        target: torch.Tensor,
        vocab_start: int,
        vocab_end: int,
        tp_group: dist.ProcessGroup | None,
    ) -> torch.Tensor:
        flat_logits = local_logits.float().reshape(-1, local_logits.shape[-1])
        flat_target = target.reshape(-1)

        if tp_group is None:
            loss = F.cross_entropy(flat_logits, flat_target, reduction="mean")
            ctx.save_for_backward(
                flat_logits.softmax(dim=-1),
                flat_target,
            )
            ctx.vocab_start = 0
            ctx.vocab_end = flat_logits.shape[-1]
            ctx.original_shape = tuple(local_logits.shape)
            ctx.output_dtype = local_logits.dtype
            return loss

        logits_max = flat_logits.max(dim=-1).values
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=tp_group)

        shifted_logits = flat_logits - logits_max.unsqueeze(-1)
        exp_logits = shifted_logits.exp()
        sum_exp = exp_logits.sum(dim=-1)
        dist.all_reduce(sum_exp, op=dist.ReduceOp.SUM, group=tp_group)

        local_target = flat_target - vocab_start
        target_mask = (flat_target < vocab_start) | (flat_target >= vocab_end)
        safe_local_target = local_target.clone()
        safe_local_target[target_mask] = 0

        predicted_logits = shifted_logits.gather(
            dim=-1,
            index=safe_local_target.unsqueeze(-1),
        ).squeeze(-1)
        predicted_logits[target_mask] = 0.0
        dist.all_reduce(predicted_logits, op=dist.ReduceOp.SUM, group=tp_group)

        softmax = exp_logits / sum_exp.unsqueeze(-1)
        loss = (torch.log(sum_exp) - predicted_logits).mean()

        ctx.save_for_backward(softmax, target_mask, safe_local_target)
        ctx.vocab_start = vocab_start
        ctx.vocab_end = vocab_end
        ctx.original_shape = tuple(local_logits.shape)
        ctx.output_dtype = local_logits.dtype
        return loss

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        saved_tensors = ctx.saved_tensors
        if len(saved_tensors) == 2:
            softmax, flat_target = saved_tensors
            grad_input = softmax
            grad_input.scatter_add_(
                dim=-1,
                index=flat_target.unsqueeze(-1),
                src=-torch.ones_like(flat_target, dtype=grad_input.dtype).unsqueeze(-1),
            )
        else:
            softmax, target_mask, safe_local_target = saved_tensors
            grad_input = softmax
            valid_mask = (~target_mask).to(dtype=grad_input.dtype)
            grad_input.scatter_add_(
                dim=-1,
                index=safe_local_target.unsqueeze(-1),
                src=-valid_mask.unsqueeze(-1),
            )

        grad_input = grad_input / float(grad_input.shape[0])
        grad_input = grad_input * grad_output.to(dtype=grad_input.dtype)
        grad_input = grad_input.reshape(ctx.original_shape).to(dtype=ctx.output_dtype)
        return grad_input, None, None, None, None


def _copy_to_tensor_parallel_region(
    tensor: torch.Tensor,
    group: dist.ProcessGroup | None,
) -> torch.Tensor:
    """应用 Megatron 语义的 copy-to-tp。"""
    if group is None:
        return tensor
    return _CopyToTensorParallelRegion.apply(tensor, group)


def _reduce_from_tensor_parallel_region(
    tensor: torch.Tensor,
    group: dist.ProcessGroup | None,
) -> torch.Tensor:
    """应用 Megatron 语义的 reduce-from-tp。"""
    if group is None:
        return tensor
    return _ReduceFromTensorParallelRegion.apply(tensor, group)


def _build_required_tensor_names(effective_num_layers: int) -> list[str]:
    """构造当前运行配置必须存在的 checkpoint tensor 名。"""
    tensor_names = [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    ]
    for layer_index in range(effective_num_layers):
        prefix = f"model.layers.{layer_index}"
        tensor_names.extend(
            [
                f"{prefix}.input_layernorm.weight",
                f"{prefix}.post_attention_layernorm.weight",
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.v_proj.weight",
                f"{prefix}.self_attn.o_proj.weight",
                f"{prefix}.mlp.gate_proj.weight",
                f"{prefix}.mlp.up_proj.weight",
                f"{prefix}.mlp.down_proj.weight",
            ]
        )
    return tensor_names


@dataclass(frozen=True)
class ParallelContext:
    """描述当前进程看到的 DP/TP 并行上下文。"""

    world_size: int
    rank: int
    local_rank: int
    device: torch.device
    tp_size: int
    dp_size: int
    tp_rank: int
    dp_rank: int
    tp_group: dist.ProcessGroup | None
    dp_group: dist.ProcessGroup | None


@dataclass(frozen=True)
class HfLlamaConfig:
    """描述 Hugging Face LLaMA 模型配置。"""

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool

    @classmethod
    def from_model_dir(cls, model_dir: Path) -> "HfLlamaConfig":
        """从本地 Hugging Face 模型目录读取配置。"""
        payload = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        num_attention_heads = int(payload["num_attention_heads"])
        num_key_value_heads = int(payload.get("num_key_value_heads") or num_attention_heads)
        return cls(
            hidden_size=int(payload["hidden_size"]),
            intermediate_size=int(payload["intermediate_size"]),
            num_hidden_layers=int(payload["num_hidden_layers"]),
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            vocab_size=int(payload["vocab_size"]),
            rms_norm_eps=float(payload.get("rms_norm_eps", 1.0e-6)),
            rope_theta=float(payload.get("rope_theta") or 10000.0),
            tie_word_embeddings=bool(payload.get("tie_word_embeddings", False)),
        )


class HfCheckpointLoader:
    """按需读取 Hugging Face checkpoint shard。"""

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self._shard_cache: dict[str, dict[str, torch.Tensor]] = {}
        index_path = model_dir / "pytorch_model.bin.index.json"
        if index_path.exists():
            index_payload = json.loads(index_path.read_text(encoding="utf-8"))
            self.weight_map: dict[str, str] = dict(index_payload["weight_map"])
            return

        single_file_path = model_dir / "pytorch_model.bin"
        if not single_file_path.exists():
            raise FileNotFoundError(
                "当前只支持两种 Hugging Face checkpoint 形式："
                " `pytorch_model.bin.index.json` 分片格式，"
                "或单文件 `pytorch_model.bin` 格式。"
            )

        shard = self._load_shard(single_file_path.name)
        self.weight_map = {
            tensor_name: single_file_path.name for tensor_name in shard.keys()
        }

    def _load_shard(self, shard_name: str) -> dict[str, torch.Tensor]:
        """读取单个 checkpoint shard，并放入缓存。"""
        if shard_name not in self._shard_cache:
            self._shard_cache[shard_name] = torch.load(
                self.model_dir / shard_name,
                map_location="cpu",
                weights_only=True,
            )
        return self._shard_cache[shard_name]

    def get_tensor(self, tensor_name: str) -> torch.Tensor:
        """按 tensor 名获取权重。"""
        shard_name = self.weight_map[tensor_name]
        shard = self._load_shard(shard_name)
        return shard[tensor_name]


def _validate_runtime_parallel_plan(
    hf_config: "HfLlamaConfig",
    runtime_parallel_plan: dict[str, Any],
) -> list[str]:
    """校验当前 runtime 配置是否落在已实现的标准 DP+TP 域内。"""
    errors: list[str] = []
    tp_size = int(runtime_parallel_plan["tp"])
    dp_size = int(runtime_parallel_plan["dp"])
    attention_policy = str(runtime_parallel_plan["attention_policy"])
    mlp_policy = str(runtime_parallel_plan["mlp_policy"])
    embedding_policy = str(runtime_parallel_plan["embedding_policy"])
    lm_head_policy = str(runtime_parallel_plan["lm_head_policy"])

    if tp_size < 1 or dp_size < 1:
        errors.append("tp 和 dp 都必须大于等于 1。")
    if embedding_policy not in {"replicate", "vocab_shard"}:
        errors.append(f"当前 runtime 不支持 embedding_policy={embedding_policy}。")
    if lm_head_policy not in {"replicate", "vocab_shard"}:
        errors.append(f"当前 runtime 不支持 lm_head_policy={lm_head_policy}。")
    if tp_size > 1 and attention_policy != "head_shard":
        errors.append("当前 runtime 在 tp>1 时只支持 attention_policy=head_shard。")
    if tp_size > 1 and mlp_policy != "colrow_shard":
        errors.append("当前 runtime 在 tp>1 时只支持 mlp_policy=colrow_shard。")
    if tp_size > 1 and lm_head_policy != "vocab_shard":
        errors.append("当前 runtime 在 tp>1 时只支持 lm_head_policy=vocab_shard。")
    if hf_config.num_attention_heads % tp_size != 0:
        errors.append("num_attention_heads 不能被 tp_size 整除。")
    if hf_config.num_key_value_heads % tp_size != 0:
        errors.append("num_key_value_heads 不能被 tp_size 整除。")
    if hf_config.tie_word_embeddings and embedding_policy != lm_head_policy:
        errors.append("tie_word_embeddings=True 时 embedding_policy 必须与 lm_head_policy 一致。")
    return errors


def _inspect_checkpoint_index(
    model_dir: Path,
    effective_num_layers: int,
) -> dict[str, Any]:
    """只读取 checkpoint index，检查 planning/runtime 所需 tensor 是否齐全。"""
    loader = HfCheckpointLoader(model_dir)
    required_tensors = _build_required_tensor_names(effective_num_layers)
    missing_tensors = [
        tensor_name for tensor_name in required_tensors if tensor_name not in loader.weight_map
    ]
    return {
        "required_tensor_count": len(required_tensors),
        "missing_tensor_count": len(missing_tensors),
        "has_all_required_tensors": not missing_tensors,
        "missing_tensors": missing_tensors[:32],
    }


def _build_rank_layout(
    hf_config: "HfLlamaConfig",
    runtime_parallel_plan: dict[str, Any],
    global_rank: int,
) -> dict[str, Any]:
    """构造单个 rank 的静态切分布局摘要。"""
    tp_size = int(runtime_parallel_plan["tp"])
    dp_size = int(runtime_parallel_plan["dp"])
    tp_rank = global_rank % tp_size
    dp_rank = global_rank // tp_size
    head_dim = hf_config.hidden_size // hf_config.num_attention_heads
    local_attention_heads = hf_config.num_attention_heads // tp_size
    local_kv_heads = hf_config.num_key_value_heads // tp_size

    embedding_policy = str(runtime_parallel_plan["embedding_policy"])
    if embedding_policy == "vocab_shard" and tp_size > 1:
        embedding_start, embedding_end = _split_evenly(hf_config.vocab_size, tp_size, tp_rank)
    else:
        embedding_start, embedding_end = 0, hf_config.vocab_size

    lm_head_policy = str(runtime_parallel_plan["lm_head_policy"])
    if lm_head_policy == "vocab_shard" and tp_size > 1:
        lm_head_start, lm_head_end = _split_evenly(hf_config.vocab_size, tp_size, tp_rank)
    else:
        lm_head_start, lm_head_end = 0, hf_config.vocab_size

    q_proj_rows = _split_evenly(hf_config.hidden_size, tp_size, tp_rank)
    kv_proj_rows = _split_evenly(hf_config.num_key_value_heads * head_dim, tp_size, tp_rank)
    mlp_rows = _split_evenly(hf_config.intermediate_size, tp_size, tp_rank)

    return {
        "global_rank": global_rank,
        "tp_rank": tp_rank,
        "dp_rank": dp_rank,
        "tp_group_ranks": [dp_rank * tp_size + offset for offset in range(tp_size)],
        "dp_group_ranks": [offset * tp_size + tp_rank for offset in range(dp_size)],
        "embedding": {
            "policy": embedding_policy,
            "vocab_range": [embedding_start, embedding_end],
            "weight_shape": [embedding_end - embedding_start, hf_config.hidden_size],
        },
        "attention": {
            "policy": str(runtime_parallel_plan["attention_policy"]),
            "local_attention_heads": local_attention_heads,
            "local_kv_heads": local_kv_heads,
            "head_dim": head_dim,
            "q_proj_weight_shape": [q_proj_rows[1] - q_proj_rows[0], hf_config.hidden_size],
            "k_proj_weight_shape": [kv_proj_rows[1] - kv_proj_rows[0], hf_config.hidden_size],
            "v_proj_weight_shape": [kv_proj_rows[1] - kv_proj_rows[0], hf_config.hidden_size],
            "o_proj_weight_shape": [hf_config.hidden_size, q_proj_rows[1] - q_proj_rows[0]],
        },
        "mlp": {
            "policy": str(runtime_parallel_plan["mlp_policy"]),
            "gate_proj_weight_shape": [mlp_rows[1] - mlp_rows[0], hf_config.hidden_size],
            "up_proj_weight_shape": [mlp_rows[1] - mlp_rows[0], hf_config.hidden_size],
            "down_proj_weight_shape": [hf_config.hidden_size, mlp_rows[1] - mlp_rows[0]],
        },
        "lm_head": {
            "policy": lm_head_policy,
            "vocab_range": [lm_head_start, lm_head_end],
            "weight_shape": [lm_head_end - lm_head_start, hf_config.hidden_size],
        },
    }


def _inspect_runtime_config(config: dict[str, Any]) -> dict[str, Any]:
    """在不真正启动 distributed 的情况下，静态检查 planning/runtime 配置。"""
    model_dir = Path(config["model_path"])
    hf_config = HfLlamaConfig.from_model_dir(model_dir)
    runtime_parallel_plan = config["runtime_parallel_plan"]
    runtime_model = config["runtime_model"]
    effective_num_layers = int(runtime_model["effective_num_layers"])

    validation_errors = _validate_runtime_parallel_plan(hf_config, runtime_parallel_plan)
    checkpoint_validation = _inspect_checkpoint_index(
        model_dir=model_dir,
        effective_num_layers=effective_num_layers,
    )
    world_size = int(runtime_parallel_plan["tp"]) * int(runtime_parallel_plan["dp"])
    rank_layouts = [
        _build_rank_layout(
            hf_config=hf_config,
            runtime_parallel_plan=runtime_parallel_plan,
            global_rank=global_rank,
        )
        for global_rank in range(world_size)
    ]

    return {
        "backend": config["backend"],
        "script_role": config["script_role"],
        "runtime_note": config["runtime_note"],
        "model_path": str(model_dir),
        "model_summary": {
            "hidden_size": hf_config.hidden_size,
            "intermediate_size": hf_config.intermediate_size,
            "num_hidden_layers": hf_config.num_hidden_layers,
            "num_attention_heads": hf_config.num_attention_heads,
            "num_key_value_heads": hf_config.num_key_value_heads,
            "vocab_size": hf_config.vocab_size,
            "tie_word_embeddings": hf_config.tie_word_embeddings,
        },
        "planned_parallel_plan": config["planned_parallel_plan"],
        "runtime_parallel_plan": runtime_parallel_plan,
        "runtime_model": runtime_model,
        "is_runtime_lowerable": not validation_errors,
        "validation_errors": validation_errors,
        "checkpoint_validation": checkpoint_validation,
        "rank_layouts": rank_layouts,
        "layout_semantics": {
            "style": "megatron_standard",
            "supports_dp_tp_only": True,
            "tp_attention": "head_shard",
            "tp_mlp": "colrow_shard",
            "lm_head": runtime_parallel_plan["lm_head_policy"],
        },
    }


def _maybe_init_process_group() -> tuple[int, int, int, torch.device]:
    """初始化分布式环境，并返回 rank/device 信息。"""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    if not dist.is_initialized():
        dist.init_process_group(backend=_pick_backend(), init_method="env://")
    return dist.get_world_size(), dist.get_rank(), local_rank, device


def _build_parallel_context(runtime_parallel_plan: dict[str, Any]) -> ParallelContext:
    """构造当前 rank 对应的 DP/TP 通信组。"""
    world_size, rank, local_rank, device = _maybe_init_process_group()
    tp_size = int(runtime_parallel_plan["tp"])
    dp_size = int(runtime_parallel_plan["dp"])
    expected_world_size = tp_size * dp_size
    if world_size != expected_world_size:
        raise RuntimeError(
            f"PyTorch backend 期望 WORLD_SIZE={expected_world_size}，实际收到 {world_size}。"
        )

    tp_rank = rank % tp_size
    dp_rank = rank // tp_size
    tp_group: dist.ProcessGroup | None = None
    dp_group: dist.ProcessGroup | None = None

    if tp_size > 1:
        for current_dp_rank in range(dp_size):
            ranks = [current_dp_rank * tp_size + offset for offset in range(tp_size)]
            group = dist.new_group(ranks=ranks)
            if rank in ranks:
                tp_group = group

    if dp_size > 1:
        for current_tp_rank in range(tp_size):
            ranks = [current_dp_rank * tp_size + current_tp_rank for current_dp_rank in range(dp_size)]
            group = dist.new_group(ranks=ranks)
            if rank in ranks:
                dp_group = group

    return ParallelContext(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        device=device,
        tp_size=tp_size,
        dp_size=dp_size,
        tp_rank=tp_rank,
        dp_rank=dp_rank,
        tp_group=tp_group,
        dp_group=dp_group,
    )


def _split_evenly(total_size: int, part_count: int, part_rank: int) -> tuple[int, int]:
    """把总长度按尽量均匀的方式切到某个分片。"""
    base = total_size // part_count
    remain = total_size % part_count
    start = part_rank * base + min(part_rank, remain)
    stop = start + base + (1 if part_rank < remain else 0)
    return start, stop


def _copy_tensor_to_parameter(param: nn.Parameter, tensor: torch.Tensor) -> None:
    """把 CPU tensor 拷到目标参数。"""
    with torch.no_grad():
        param.copy_(tensor.to(device=param.device, dtype=param.dtype))


def _slice_tensor_parallel_dim0(
    tensor: torch.Tensor,
    context: ParallelContext,
) -> torch.Tensor:
    """按 dim=0 做 TP 切片。"""
    if context.tp_size == 1:
        return tensor
    start, stop = _split_evenly(tensor.shape[0], context.tp_size, context.tp_rank)
    return tensor[start:stop].contiguous()


def _slice_tensor_parallel_dim1(
    tensor: torch.Tensor,
    context: ParallelContext,
) -> torch.Tensor:
    """按 dim=1 做 TP 切片。"""
    if context.tp_size == 1:
        return tensor
    start, stop = _split_evenly(tensor.shape[1], context.tp_size, context.tp_rank)
    return tensor[:, start:stop].contiguous()


def _rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    """实现 rotary 里的 half rotation。"""
    first_half = hidden_states[..., : hidden_states.shape[-1] // 2]
    second_half = hidden_states[..., hidden_states.shape[-1] // 2 :]
    return torch.cat((-second_half, first_half), dim=-1)


def _build_rotary_cos_sin(
    head_dim: int,
    seq_length: int,
    base: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """构造 rotary embedding 的 cos/sin。"""
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
            / float(head_dim)
        )
    )
    positions = torch.arange(seq_length, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype=dtype).unsqueeze(0).unsqueeze(0)
    sin = emb.sin().to(dtype=dtype).unsqueeze(0).unsqueeze(0)
    return cos, sin


def _apply_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """对 Q/K 应用 rotary embedding。"""
    return (
        (query * cos) + (_rotate_half(query) * sin),
        (key * cos) + (_rotate_half(key) * sin),
    )


def _repeat_kv(hidden_states: torch.Tensor, repeat_count: int) -> torch.Tensor:
    """把 KV 按组扩展到 attention head 数。"""
    if repeat_count == 1:
        return hidden_states
    batch_size, kv_heads, seq_length, head_dim = hidden_states.shape
    expanded = hidden_states[:, :, None, :, :].expand(
        batch_size,
        kv_heads,
        repeat_count,
        seq_length,
        head_dim,
    )
    return expanded.reshape(batch_size, kv_heads * repeat_count, seq_length, head_dim)


def _vocab_parallel_cross_entropy(
    local_logits: torch.Tensor,
    target: torch.Tensor,
    vocab_start: int,
    vocab_end: int,
    tp_group: dist.ProcessGroup | None,
) -> torch.Tensor:
    """对 `vocab_shard` 的本地 logits 计算真实 cross entropy。"""
    return _VocabParallelCrossEntropy.apply(
        local_logits,
        target,
        vocab_start,
        vocab_end,
        tp_group,
    )


class HfRmsNorm(nn.Module):
    """LLaMA 使用的 RMSNorm。"""

    def __init__(self, hidden_size: int, eps: float, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))

    def load_weights(self, loader: HfCheckpointLoader, tensor_name: str) -> None:
        """加载 norm 权重。"""
        _copy_tensor_to_parameter(self.weight, loader.get_tensor(tensor_name))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """执行 RMSNorm。"""
        variance = hidden_states.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden_states.float() * torch.rsqrt(variance + self.eps)
        normalized = normalized.to(dtype=hidden_states.dtype)
        return normalized * self.weight


class HfEmbedding(nn.Module):
    """支持 replicate/vocab_shard 的 embedding。"""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        embedding_policy: str,
        context: ParallelContext,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.context = context
        self.embedding_policy = embedding_policy
        if embedding_policy == "vocab_shard" and context.tp_size > 1:
            self.vocab_start, self.vocab_end = _split_evenly(vocab_size, context.tp_size, context.tp_rank)
        else:
            self.vocab_start, self.vocab_end = 0, vocab_size
        local_vocab_size = self.vocab_end - self.vocab_start
        self.weight = nn.Parameter(
            torch.empty(local_vocab_size, hidden_size, device=context.device, dtype=dtype)
        )

    def load_weights(self, loader: HfCheckpointLoader, tensor_name: str) -> None:
        """加载 embedding 权重。"""
        weight = loader.get_tensor(tensor_name)
        if self.embedding_policy == "vocab_shard" and self.context.tp_size > 1:
            weight = _slice_tensor_parallel_dim0(weight, self.context)
        _copy_tensor_to_parameter(self.weight, weight)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """执行 embedding lookup。"""
        if self.embedding_policy == "vocab_shard" and self.context.tp_size > 1:
            input_mask = (input_ids < self.vocab_start) | (input_ids >= self.vocab_end)
            local_input = input_ids.clone() - self.vocab_start
            local_input[input_mask] = 0
            output = F.embedding(local_input, self.weight)
            output[input_mask] = 0.0
            return _reduce_from_tensor_parallel_region(output, self.context.tp_group)
        return F.embedding(input_ids, self.weight)


class ColumnParallelLinear(nn.Module):
    """Megatron-style 的列切分 Linear。"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        context: ParallelContext,
        dtype: torch.dtype,
        bias: bool = False,
    ) -> None:
        super().__init__()
        output_start, output_end = _split_evenly(output_size, context.tp_size, context.tp_rank)
        self.context = context
        self.local_output_size = output_end - output_start
        self.output_start = output_start
        self.output_end = output_end
        self.weight = nn.Parameter(
            torch.empty(self.local_output_size, input_size, device=context.device, dtype=dtype)
        )
        self.bias = None
        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.local_output_size, device=context.device, dtype=dtype)
            )

    def load_weights(self, loader: HfCheckpointLoader, tensor_name: str) -> None:
        """加载列切分权重。"""
        weight = loader.get_tensor(tensor_name)
        _copy_tensor_to_parameter(self.weight, _slice_tensor_parallel_dim0(weight, self.context))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """返回本 rank 的本地输出分片。"""
        if self.context.tp_size > 1:
            hidden_states = _copy_to_tensor_parallel_region(
                hidden_states,
                self.context.tp_group,
            )
        return F.linear(hidden_states, self.weight, self.bias)


class RowParallelLinear(nn.Module):
    """Megatron-style 的行切分 Linear。"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        context: ParallelContext,
        dtype: torch.dtype,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.context = context
        input_start, input_end = _split_evenly(input_size, context.tp_size, context.tp_rank)
        self.local_input_size = input_end - input_start
        self.weight = nn.Parameter(
            torch.empty(output_size, self.local_input_size, device=context.device, dtype=dtype)
        )
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size, device=context.device, dtype=dtype))

    def load_weights(self, loader: HfCheckpointLoader, tensor_name: str) -> None:
        """加载行切分权重。"""
        weight = loader.get_tensor(tensor_name)
        _copy_tensor_to_parameter(self.weight, _slice_tensor_parallel_dim1(weight, self.context))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """在 TP group 上聚合输出。"""
        output = F.linear(hidden_states, self.weight, None)
        if self.context.tp_size > 1:
            output = _reduce_from_tensor_parallel_region(output, self.context.tp_group)
        if self.bias is not None:
            output = output + self.bias
        return output


class HfLlamaAttention(nn.Module):
    """支持标准 `head_shard` 的 LLaMA attention。"""

    def __init__(
        self,
        config: HfLlamaConfig,
        context: ParallelContext,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if config.num_attention_heads % context.tp_size != 0:
            raise RuntimeError("num_attention_heads 必须能被 tp_size 整除。")
        if config.num_key_value_heads % context.tp_size != 0:
            raise RuntimeError("num_key_value_heads 必须能被 tp_size 整除。")

        self.context = context
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.local_num_attention_heads = config.num_attention_heads // context.tp_size
        self.local_num_key_value_heads = config.num_key_value_heads // context.tp_size
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.rope_theta = config.rope_theta

        q_output_size = self.local_num_attention_heads * self.head_dim
        kv_output_size = self.local_num_key_value_heads * self.head_dim
        self.q_proj = ColumnParallelLinear(config.hidden_size, config.hidden_size, context, dtype)
        self.k_proj = ColumnParallelLinear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            context,
            dtype,
        )
        self.v_proj = ColumnParallelLinear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            context,
            dtype,
        )
        self.o_proj = RowParallelLinear(config.hidden_size, config.hidden_size, context, dtype)
        self.q_output_size = q_output_size
        self.kv_output_size = kv_output_size

    def load_weights(self, loader: HfCheckpointLoader, layer_prefix: str) -> None:
        """加载 attention 权重。"""
        self.q_proj.load_weights(loader, f"{layer_prefix}.self_attn.q_proj.weight")
        self.k_proj.load_weights(loader, f"{layer_prefix}.self_attn.k_proj.weight")
        self.v_proj.load_weights(loader, f"{layer_prefix}.self_attn.v_proj.weight")
        self.o_proj.load_weights(loader, f"{layer_prefix}.self_attn.o_proj.weight")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """执行 attention 前向。"""
        batch_size, seq_length, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(
            batch_size,
            seq_length,
            self.local_num_attention_heads,
            self.head_dim,
        ).transpose(1, 2)
        key = self.k_proj(hidden_states).view(
            batch_size,
            seq_length,
            self.local_num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)
        value = self.v_proj(hidden_states).view(
            batch_size,
            seq_length,
            self.local_num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)

        cos, sin = _build_rotary_cos_sin(
            head_dim=self.head_dim,
            seq_length=seq_length,
            base=self.rope_theta,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        query, key = _apply_rotary_pos_emb(query, key, cos, sin)

        key = _repeat_kv(key, self.num_key_value_groups)
        value = _repeat_kv(value, self.num_key_value_groups)

        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(
            batch_size,
            seq_length,
            self.local_num_attention_heads * self.head_dim,
        )
        return self.o_proj(attn_output)


class HfLlamaMlp(nn.Module):
    """支持标准 `colrow_shard` 的 LLaMA MLP。"""

    def __init__(
        self,
        config: HfLlamaConfig,
        context: ParallelContext,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.gate_proj = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            context,
            dtype,
        )
        self.up_proj = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            context,
            dtype,
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            context,
            dtype,
        )

    def load_weights(self, loader: HfCheckpointLoader, layer_prefix: str) -> None:
        """加载 MLP 权重。"""
        self.gate_proj.load_weights(loader, f"{layer_prefix}.mlp.gate_proj.weight")
        self.up_proj.load_weights(loader, f"{layer_prefix}.mlp.up_proj.weight")
        self.down_proj.load_weights(loader, f"{layer_prefix}.mlp.down_proj.weight")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """执行 MLP 前向。"""
        gate = F.silu(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        return self.down_proj(gate * up)


class HfLlamaDecoderLayer(nn.Module):
    """LLaMA decoder layer。"""

    def __init__(
        self,
        layer_index: int,
        config: HfLlamaConfig,
        context: ParallelContext,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.layer_index = layer_index
        self.input_layernorm = HfRmsNorm(
            config.hidden_size,
            config.rms_norm_eps,
            context.device,
            dtype,
        )
        self.post_attention_layernorm = HfRmsNorm(
            config.hidden_size,
            config.rms_norm_eps,
            context.device,
            dtype,
        )
        self.self_attn = HfLlamaAttention(config, context, dtype)
        self.mlp = HfLlamaMlp(config, context, dtype)

    def load_weights(self, loader: HfCheckpointLoader) -> None:
        """加载当前层权重。"""
        prefix = f"model.layers.{self.layer_index}"
        self.input_layernorm.load_weights(loader, f"{prefix}.input_layernorm.weight")
        self.post_attention_layernorm.load_weights(
            loader,
            f"{prefix}.post_attention_layernorm.weight",
        )
        self.self_attn.load_weights(loader, prefix)
        self.mlp.load_weights(loader, prefix)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """执行 decoder layer 前向。"""
        hidden_states = hidden_states + self.self_attn(self.input_layernorm(hidden_states))
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states


class HfParallelLmHead(nn.Module):
    """支持 `vocab_shard` 的 LM Head。"""

    def __init__(
        self,
        config: HfLlamaConfig,
        context: ParallelContext,
        dtype: torch.dtype,
        lm_head_policy: str,
    ) -> None:
        super().__init__()
        self.context = context
        self.lm_head_policy = lm_head_policy
        self.vocab_size = config.vocab_size
        if lm_head_policy == "vocab_shard" and context.tp_size > 1:
            self.proj = ColumnParallelLinear(config.hidden_size, config.vocab_size, context, dtype)
            self.vocab_start = self.proj.output_start
            self.vocab_end = self.proj.output_end
        else:
            self.proj = ColumnParallelLinear(config.hidden_size, config.vocab_size, context, dtype)
            self.vocab_start = 0
            self.vocab_end = config.vocab_size

    def load_weights(self, loader: HfCheckpointLoader, tensor_name: str) -> None:
        """加载 lm_head 权重。"""
        self.proj.load_weights(loader, tensor_name)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """返回 logits。"""
        return self.proj(hidden_states)


class HfLlamaForCausalLm(nn.Module):
    """最小可训练的 HF LLaMA 模型。"""

    def __init__(
        self,
        hf_config: HfLlamaConfig,
        runtime_config: dict[str, Any],
        runtime_parallel_plan: dict[str, Any],
        context: ParallelContext,
    ) -> None:
        super().__init__()
        self.hf_config = hf_config
        self.runtime_config = runtime_config
        self.runtime_parallel_plan = runtime_parallel_plan
        self.context = context
        self.dtype = _resolve_dtype(runtime_config["parameter_dtype"])
        self.embedding = HfEmbedding(
            vocab_size=hf_config.vocab_size,
            hidden_size=hf_config.hidden_size,
            embedding_policy=runtime_parallel_plan["embedding_policy"],
            context=context,
            dtype=self.dtype,
        )
        effective_num_layers = int(runtime_config["effective_num_layers"])
        self.layers = nn.ModuleList(
            [
                HfLlamaDecoderLayer(
                    layer_index=index,
                    config=hf_config,
                    context=context,
                    dtype=self.dtype,
                )
                for index in range(effective_num_layers)
            ]
        )
        self.norm = HfRmsNorm(
            hf_config.hidden_size,
            hf_config.rms_norm_eps,
            context.device,
            self.dtype,
        )
        self.lm_head = HfParallelLmHead(
            config=hf_config,
            context=context,
            dtype=self.dtype,
            lm_head_policy=runtime_parallel_plan["lm_head_policy"],
        )

    def load_weights(self, loader: HfCheckpointLoader) -> None:
        """从 HF checkpoint 加载权重。"""
        self.embedding.load_weights(loader, "model.embed_tokens.weight")
        for layer in self.layers:
            layer.load_weights(loader)
        self.norm.load_weights(loader, "model.norm.weight")
        self.lm_head.load_weights(loader, "lm_head.weight")

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """执行前向，返回 logits。"""
        hidden_states = self.embedding(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)


@torch.no_grad()
def _apply_random_init(module: nn.Module, std: float = 0.02) -> None:
    """按 Megatron 风格做随机初始化，避免一边预训练、一边随机导致 loss 不可比。

    - Linear / Embedding 权重：N(0, std)
    - RMSNorm 权重：全 1
    - 其余 1D 参数：0
    """
    for name, param in module.named_parameters():
        is_norm_weight = (
            param.dim() == 1
            and name.endswith("weight")
            and ("norm" in name or "layernorm" in name)
        )
        if is_norm_weight:
            nn.init.ones_(param)
        elif param.dim() >= 2 or name.endswith("weight"):
            nn.init.normal_(param, mean=0.0, std=float(std))
        else:
            nn.init.zeros_(param)


def _build_model(
    model_path: str,
    runtime_parallel_plan: dict[str, Any],
    runtime_model: dict[str, Any],
    context: ParallelContext,
    *,
    use_real_hf_weights: bool = False,
    init_method_std: float = 0.02,
) -> nn.Module:
    """构造模型，并按 DP group 包装 DDP。

    默认随机初始化，与 Megatron backend 的 `init_method_std` 语义对齐。
    仅当 `use_real_hf_weights=True` 时才加载本地 HF checkpoint。
    """
    model_dir = Path(model_path)
    hf_config = HfLlamaConfig.from_model_dir(model_dir)
    model = HfLlamaForCausalLm(
        hf_config=hf_config,
        runtime_config=runtime_model,
        runtime_parallel_plan=runtime_parallel_plan,
        context=context,
    )
    if use_real_hf_weights:
        model.load_weights(HfCheckpointLoader(model_dir))
        weight_mode = "hf_pretrained"
    else:
        _apply_random_init(model, std=init_method_std)
        weight_mode = f"random_init(std={init_method_std})"
        # 同一 DP 组内广播，保证 data-parallel replica 起点一致。
        # 组内 global rank 布局为: tp_rank, tp_size+tp_rank, ...
        if context.dp_size > 1 and context.dp_group is not None:
            src_global_rank = int(context.tp_rank)
            for parameter in model.parameters():
                dist.broadcast(
                    parameter.data,
                    src=src_global_rank,
                    group=context.dp_group,
                )

    if context.rank == 0:
        print(f"[PYTORCH] model weights: {weight_mode}", flush=True)

    if context.dp_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[context.local_rank] if context.device.type == "cuda" else None,
            process_group=context.dp_group,
            broadcast_buffers=False,
        )
    return model


def _compute_language_model_loss(
    model: nn.Module,
    input_ids: torch.Tensor,
    runtime_parallel_plan: dict[str, Any],
    tp_group: dist.ProcessGroup | None,
) -> torch.Tensor:
    """计算最小 next-token loss。"""
    logits = model(input_ids)
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]

    lm_head_policy = runtime_parallel_plan["lm_head_policy"]
    if int(runtime_parallel_plan["tp"]) > 1 and lm_head_policy == "vocab_shard":
        if isinstance(model, DistributedDataParallel):
            lm_head = model.module.lm_head
        else:
            lm_head = model.lm_head
        return _vocab_parallel_cross_entropy(
            shift_logits,
            shift_labels,
            vocab_start=int(lm_head.vocab_start),
            vocab_end=int(lm_head.vocab_end),
            tp_group=tp_group,
        )

    return F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]).float(),
        shift_labels.reshape(-1),
        reduction="mean",
    )


def _run_train_loop(config: dict[str, Any]) -> dict[str, Any]:
    """执行一次最小训练循环。"""
    runtime_parallel_plan = config["runtime_parallel_plan"]
    runtime_model = config["runtime_model"]
    train_config = config["train"]
    context = _build_parallel_context(runtime_parallel_plan)
    # 默认与 Megatron 一致：随机初始化。仅显式打开 use_real_hf_weights 时才加载预训练。
    use_real_hf_weights = bool(train_config.get("use_real_hf_weights", False))
    init_method_std = float(train_config.get("init_method_std", 0.02))
    model = _build_model(
        model_path=config["model_path"],
        runtime_parallel_plan=runtime_parallel_plan,
        runtime_model=runtime_model,
        context=context,
        use_real_hf_weights=use_real_hf_weights,
        init_method_std=init_method_std,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config["lr"]),
        weight_decay=float(train_config["weight_decay"]),
    )

    hf_config = HfLlamaConfig.from_model_dir(Path(config["model_path"]))
    train_iters = int(train_config["train_iters"])
    micro_batch_size = int(train_config["micro_batch_size"])
    seq_length = int(train_config["seq_length"])
    log_interval = max(int(train_config.get("log_interval", 1)), 1)
    use_mock_data = bool(train_config.get("use_mock_data", True))
    data_path = train_config.get("data_path")
    final_loss = 0.0

    token_stream: np.ndarray | None = None
    data_source = "mock_random"
    if not use_mock_data and data_path:
        # 只在每个 DP 组的 tp_rank=0 加载一次即可；同组其余 TP rank 复用相同切片逻辑。
        # 这里每个 rank 都 mmap 同一份 indexed dataset，开销可接受，实现更简单。
        token_stream = _load_megatron_token_stream(str(data_path))
        data_source = f"megatron_indexed:{data_path}"
        if context.rank == 0:
            print(
                "[PYTORCH] using real megatron dataset tokens=%d source=%s"
                % (int(token_stream.size), data_source),
                flush=True,
            )
    elif context.rank == 0:
        print("[PYTORCH] using mock random tokens", flush=True)

    for iteration in range(train_iters):
        input_ids = _build_input_ids_batch(
            token_stream=token_stream,
            vocab_size=hf_config.vocab_size,
            micro_batch_size=micro_batch_size,
            seq_length=seq_length,
            iteration=iteration,
            dp_rank=context.dp_rank,
            dp_size=context.dp_size,
            device=context.device,
        )
        optimizer.zero_grad(set_to_none=True)
        loss = _compute_language_model_loss(
            model=model,
            input_ids=input_ids,
            runtime_parallel_plan=runtime_parallel_plan,
            tp_group=context.tp_group,
        )
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())

        # 只在 rank 0 打印逐 step loss；格式与 e2e 汇报脚本中的 Megatron 统一输出保持一致。
        if context.rank == 0 and (iteration + 1) % log_interval == 0:
            print(
                "[PYTORCH][step %d/%d] loss=%.6f"
                % (iteration + 1, train_iters, final_loss),
                flush=True,
            )

    if dist.is_initialized():
        if context.device.type == "cuda":
            dist.barrier(device_ids=[context.local_rank])
        else:
            dist.barrier()

    return {
        "backend": config["backend"],
        "script_role": config["script_role"],
        "rank": context.rank,
        "world_size": context.world_size,
        "tp_size": context.tp_size,
        "dp_size": context.dp_size,
        "tp_rank": context.tp_rank,
        "dp_rank": context.dp_rank,
        "effective_num_layers": int(runtime_model["effective_num_layers"]),
        "effective_seq_length": int(runtime_model["effective_seq_length"]),
        "final_loss": final_loss,
        "data_source": data_source,
        "runtime_note": config["runtime_note"],
    }


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="运行 PyTorch backend 的真实 HF LLaMA DP+TP runtime")
    parser.add_argument("--config", required=True, help="lowering 生成的运行配置路径")
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="只做 planning/runtime 静态检查，不真正启动 distributed 训练",
    )
    args = parser.parse_args()

    loaded_config = _load_runtime_config(args.config)
    if args.inspect_only:
        print(json.dumps(_inspect_runtime_config(loaded_config), ensure_ascii=False, indent=2))
        return

    result = _run_train_loop(loaded_config)
    if result["rank"] == 0:
        print(json.dumps(result, ensure_ascii=False, indent=2))

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
