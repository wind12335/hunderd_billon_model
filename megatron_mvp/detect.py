"""模型自动检测与结构规格提取。"""

from __future__ import annotations

import json
from pathlib import Path

from megatron_mvp.types import ModelArtifacts
from megatron_mvp.types import ModelSpec


_WEIGHT_GLOBS = (
    "pytorch_model.bin",
    "pytorch_model-*.bin",
    "model.safetensors",
    "model-*.safetensors",
    "*.safetensors",
)

_TOKENIZER_NAMES = (
    "tokenizer.model",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)


def _iter_candidate_dirs(root: Path) -> list[Path]:
    """枚举可能的模型目录。

    这里优先看根目录本身，再做一层和两层子目录扫描，避免误扫整个大仓库。
    """
    candidates = [root]
    candidates.extend(path for path in root.iterdir() if path.is_dir())
    for child in root.iterdir():
        if child.is_dir():
            candidates.extend(path for path in child.iterdir() if path.is_dir())
    return candidates


def _collect_weight_files(model_dir: Path) -> tuple[list[Path], str]:
    """收集模型权重文件并推断权重格式。"""
    matched_files: list[Path] = []
    for pattern in _WEIGHT_GLOBS:
        matched_files.extend(sorted(model_dir.glob(pattern)))

    unique_files = sorted({path.resolve() for path in matched_files})
    if not unique_files:
        return [], "unknown"

    suffixes = {path.suffix for path in unique_files}
    if ".safetensors" in suffixes:
        return unique_files, "safetensors"
    return unique_files, "bin"


def detect_hf_model(search_root: Path) -> ModelArtifacts:
    """自动检测 Hugging Face 模型目录。"""
    best_match: ModelArtifacts | None = None

    for candidate_dir in _iter_candidate_dirs(search_root):
        config_path = candidate_dir / "config.json"
        if not config_path.is_file():
            continue

        weight_files, weight_format = _collect_weight_files(candidate_dir)
        if not weight_files:
            continue

        tokenizer_files = [
            path.resolve()
            for name in _TOKENIZER_NAMES
            for path in [candidate_dir / name]
            if path.is_file()
        ]

        artifacts = ModelArtifacts(
            model_dir=candidate_dir.resolve(),
            config_path=config_path.resolve(),
            weight_files=weight_files,
            tokenizer_files=tokenizer_files,
            shard_count=len(weight_files),
            weight_format=weight_format,
        )

        # 优先选择更浅的目录，避免把 tokenizer 等子目录误判成主模型目录。
        if best_match is None or len(artifacts.model_dir.parts) < len(best_match.model_dir.parts):
            best_match = artifacts

    if best_match is None:
        raise FileNotFoundError(
            f"在 {search_root} 下未检测到 Hugging Face 模型目录，需要同时存在 config.json 和权重文件。"
        )

    return best_match


def _estimate_transformer_block_params(
    hidden_size: int,
    intermediate_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int | None = None,
) -> int:
    """按 LLaMA 风格结构做 block-level 参数量粗估。

    第一版只需要粗搜索，因此这里不追求逐张量精确值，只要求数量级稳定。
    head_dim 显式给出时按 q/kv 头维计算（GLM-4.5 系列的 q_proj 会从
    hidden_size 上采样到 num_heads*head_dim，不能按 hidden 方阵估算）。
    """
    if head_dim is None or head_dim <= 0:
        head_dim = hidden_size // num_heads if hidden_size % num_heads == 0 else hidden_size
    kv_hidden_size = num_kv_heads * head_dim
    q_hidden_size = num_heads * head_dim
    attention_params = (
        hidden_size * q_hidden_size
        + hidden_size * kv_hidden_size
        + hidden_size * kv_hidden_size
        + q_hidden_size * hidden_size
    )
    mlp_params = hidden_size * intermediate_size * 3
    norm_params = hidden_size * 2
    return attention_params + mlp_params + norm_params


def _first_config_value(config: dict, *keys: str, default=None):
    """按优先级返回第一个存在的配置键值。"""
    for key in keys:
        if key in config and config[key] is not None:
            return config[key]
    return default


def _extract_moe_fields(config: dict) -> dict:
    """从 Hugging Face config 中提取 MoE 字段（兼容多套命名习惯）。

    覆盖的命名族：
    1. GLM-4.5 / glm4_moe: n_routed_experts / num_experts_per_tok /
       moe_intermediate_size / shared_expert_intermediate_size /
       first_k_dense_replace / n_shared_experts。
    2. Qwen3-MoE / LLaMA-MoE: num_experts / num_experts_per_tok /
       moe_intermediate_size / decoder_sparse_step。
    3. Mixtral: num_local_experts / num_experts_per_tok。
    """
    num_experts = int(
        _first_config_value(
            config,
            "n_routed_experts",
            "num_experts",
            "num_local_experts",
            default=0,
        )
    )
    num_experts_per_tok = int(
        _first_config_value(
            config,
            "num_experts_per_tok",
            "n_experts_per_tok",
            "num_experts_topk",
            default=0,
        )
    )
    moe_intermediate_size = _first_config_value(config, "moe_intermediate_size")
    shared_expert_intermediate_size = _first_config_value(
        config,
        "shared_expert_intermediate_size",
        "moe_shared_expert_intermediate_size",
    )
    # GLM-4.5 用 n_shared_experts=1 + shared_expert_intermediate_size；
    # 只给 n_shared_experts 时退回到 intermediate/moe intermediate 当作共享专家宽度。
    n_shared_experts = int(_first_config_value(config, "n_shared_experts", default=0))
    if shared_expert_intermediate_size is None and n_shared_experts > 0:
        shared_expert_intermediate_size = moe_intermediate_size or intermediate_fallback(
            config
        )
    first_k_dense_replace = int(
        _first_config_value(
            config,
            "first_k_dense_replace",
            default=0,
        )
    )
    head_dim = _first_config_value(config, "head_dim")
    partial_rotary_factor = _first_config_value(config, "partial_rotary_factor")
    num_nextn_predict_layers = int(
        _first_config_value(config, "num_nextn_predict_layers", default=0)
    )
    return {
        "num_experts": num_experts,
        "num_experts_per_tok": num_experts_per_tok,
        "moe_intermediate_size": (
            int(moe_intermediate_size) if moe_intermediate_size is not None else None
        ),
        "shared_expert_intermediate_size": (
            int(shared_expert_intermediate_size)
            if shared_expert_intermediate_size is not None
            else None
        ),
        "first_k_dense_replace": first_k_dense_replace,
        "head_dim": int(head_dim) if head_dim is not None else None,
        "partial_rotary_factor": (
            float(partial_rotary_factor) if partial_rotary_factor is not None else None
        ),
        "num_nextn_predict_layers": num_nextn_predict_layers,
    }


def intermediate_fallback(config: dict):
    """返回用于共享专家宽度回退的 intermediate_size。"""
    raw = config.get("intermediate_size")
    return int(raw) if raw is not None else None


def _estimate_moe_model_params(
    hidden_size: int,
    intermediate_size: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int | None,
    num_experts: int,
    moe_intermediate_size: int | None,
    shared_expert_intermediate_size: int | None,
    first_k_dense_replace: int,
) -> tuple[int, float]:
    """按 MoE 结构估算总参数量与平均每层参数量。

    返回 (estimated_total_params, avg_params_per_layer)。
   MoE 层 = 路由专家全体 + 共享专家；dense 层 = 普通 SwiGLU MLP。
    """
    if head_dim is None or head_dim <= 0:
        head_dim = hidden_size // num_heads if hidden_size % num_heads == 0 else hidden_size
    kv_hidden_size = num_kv_heads * head_dim
    q_hidden_size = num_heads * head_dim
    attention_params = (
        hidden_size * q_hidden_size
        + hidden_size * kv_hidden_size * 2
        + q_hidden_size * hidden_size
    )
    norm_params = hidden_size * 2

    routed_inter = moe_intermediate_size or intermediate_size
    shared_inter = shared_expert_intermediate_size
    routed_expert_params = hidden_size * routed_inter * 3
    moe_layer_params = num_experts * routed_expert_params
    if shared_inter:
        moe_layer_params += hidden_size * int(shared_inter) * 3

    moe_layers = max(num_layers - first_k_dense_replace, 0)
    dense_layers = num_layers - moe_layers
    dense_mlp_params = hidden_size * intermediate_size * 3

    transformer_params = moe_layers * (attention_params + moe_layer_params + norm_params)
    transformer_params += dense_layers * (attention_params + dense_mlp_params + norm_params)
    avg_per_layer = transformer_params / max(num_layers, 1)
    return int(transformer_params), avg_per_layer


def load_model_spec(artifacts: ModelArtifacts) -> ModelSpec:
    """从 Hugging Face config 中提取模型规格。"""
    config = json.loads(artifacts.config_path.read_text(encoding="utf-8"))

    model_type = str(config.get("model_type", "unknown"))
    architecture = str(config.get("architectures", ["unknown"])[0])
    hidden_size = int(config["hidden_size"])
    intermediate_size = int(config["intermediate_size"])
    num_layers = int(config["num_hidden_layers"])
    num_heads = int(config["num_attention_heads"])
    num_kv_heads = int(config.get("num_key_value_heads", num_heads))
    vocab_size = int(config["vocab_size"])
    max_seq_len = int(
        config.get("max_position_embeddings")
        or config.get("max_sequence_length")
        or 2048
    )
    rope_theta = float(config.get("rope_theta", 10000.0))
    rms_norm_eps = float(config.get("rms_norm_eps", 1.0e-6))
    tie_word_embeddings = bool(config.get("tie_word_embeddings", False))
    source_torch_dtype = str(config.get("torch_dtype", "unknown"))

    moe_fields = _extract_moe_fields(config)
    head_dim = moe_fields["head_dim"]

    if moe_fields["num_experts"] > 0:
        transformer_params, avg_params_per_layer = _estimate_moe_model_params(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            num_experts=moe_fields["num_experts"],
            moe_intermediate_size=moe_fields["moe_intermediate_size"],
            shared_expert_intermediate_size=moe_fields["shared_expert_intermediate_size"],
            first_k_dense_replace=moe_fields["first_k_dense_replace"],
        )
    else:
        params_per_layer = _estimate_transformer_block_params(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )
        transformer_params = params_per_layer * num_layers
        avg_params_per_layer = float(params_per_layer)

    embedding_params = vocab_size * hidden_size
    lm_head_params = 0 if tie_word_embeddings else vocab_size * hidden_size
    final_norm_params = hidden_size
    estimated_total_params = transformer_params + embedding_params + lm_head_params + final_norm_params

    return ModelSpec(
        model_dir=artifacts.model_dir,
        model_type=model_type,
        architecture=architecture,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        rope_theta=rope_theta,
        rms_norm_eps=rms_norm_eps,
        tie_word_embeddings=tie_word_embeddings,
        source_torch_dtype=source_torch_dtype,
        estimated_total_params=estimated_total_params,
        estimated_params_per_layer=int(avg_params_per_layer),
        head_dim=head_dim,
        partial_rotary_factor=moe_fields["partial_rotary_factor"],
        num_experts=moe_fields["num_experts"],
        num_experts_per_tok=moe_fields["num_experts_per_tok"],
        moe_intermediate_size=moe_fields["moe_intermediate_size"],
        shared_expert_intermediate_size=moe_fields["shared_expert_intermediate_size"],
        first_k_dense_replace=moe_fields["first_k_dense_replace"],
        num_nextn_predict_layers=moe_fields["num_nextn_predict_layers"],
    )

