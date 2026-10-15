"""声明式 autoparallel 示例。"""

from __future__ import annotations

from autoparallel import Config
from autoparallel import ParallelPlan
from autoparallel import parallelize


config = Config(
    model_path='/root/private_data/thousand_model/model/Llama-3.1-8B',
    search_target_devices=16,
    runtime_test_devices=1,
    memory_target_gb=64.0,
    prefer='MEMORY',
    search_strategy='BLOCK_LEVEL_FAST',
    search_policy='BALANCED',
    data_path=None,
    tokenizer_path=None,
    run_mode="LAUNCH",
    train_iters=20,
)
config.parallel_mode = "MODEL_PARALLEL"

# 如果你想显式固定搜索后的方案，也可以直接使用 ParallelPlan(...)：
# explicit_plan = ParallelPlan(
#     model_path='/root/private_data/thousand_model/model/Llama-3.1-8B',
#     tp=1,
#     dp=1,
#     attention_policy="replicate",
#     mlp_policy="replicate",
#     embedding_policy="replicate",
#     lm_head_policy="replicate",
# )
# @parallelize(explicit_plan)
# def train() -> None:
#     pass


@parallelize(config)
def train() -> None:
    """这里的函数体只保留训练意图，不手写 PP/TP shard 和通信细节。"""
    pass


if __name__ == "__main__":
    train()
