"""手动显式并行实现示例。

这个文件故意保留大量并行细节，作为和声明式 autoparallel 的对照基线。
"""

from __future__ import annotations

import torch
import torch.nn as nn


TP_SIZE = 8
PP_SIZE = max(2, 1)
NUM_LAYERS = 46
HIDDEN_SIZE = 4096
FFN_HIDDEN_SIZE = 10944
NUM_HEADS = 96
SEQ_LENGTH = 4096
VOCAB_SIZE = 151552


def get_pp_rank() -> int:
    """返回当前 PP rank。"""
    return 0


def get_tp_rank() -> int:
    """返回当前 TP rank。"""
    return 0


def pp_send(tensor: torch.Tensor) -> None:
    """显式发送 PP 激活。"""
    _ = tensor


def pp_recv(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    """显式接收 PP 激活。"""
    return torch.zeros(shape, device=device)


def AllReduce(tensor: torch.Tensor, group: object) -> torch.Tensor:
    """显式 TP AllReduce。"""
    _ = group
    return tensor


def build_pp_info(num_layers: int, pp_size: int) -> dict[str, list[int]]:
    """手工构造 PP 切分信息。"""
    base = num_layers // max(pp_size, 1)
    remain = num_layers % max(pp_size, 1)
    layer_splits = [base] * max(pp_size, 1)
    for idx in range(remain):
        layer_splits[idx] += 1
    return {"num_layers": layer_splits}


def build_tp_info(tp_size: int) -> dict[str, int]:
    """手工构造 TP 切分信息。"""
    return {"tp_size": tp_size, "rank": get_tp_rank()}


class LlamaEmbedding(nn.Module):
    """手工实现的 embedding 层。"""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(VOCAB_SIZE, HIDDEN_SIZE))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """执行 embedding lookup。"""
        return self.weight[input_ids]


class LlamaAttentionBlock(nn.Module):
    """手工显式 TP Attention block。"""

    def __init__(self, tp_info: dict[str, int], device_tp: torch.device) -> None:
        super().__init__()
        tp_size = tp_info["tp_size"]

        # attention 的 q/k/v/o 都需要手工按 TP 维度切分。
        self.q_proj = nn.Parameter(torch.empty(HIDDEN_SIZE, HIDDEN_SIZE // tp_size, device=device_tp))
        self.k_proj = nn.Parameter(torch.empty(HIDDEN_SIZE, HIDDEN_SIZE // tp_size, device=device_tp))
        self.v_proj = nn.Parameter(torch.empty(HIDDEN_SIZE, HIDDEN_SIZE // tp_size, device=device_tp))
        self.o_proj = nn.Parameter(torch.empty(HIDDEN_SIZE // tp_size, HIDDEN_SIZE, device=device_tp))

        # MLP 也需要手工按 TP 维度切分。
        self.gate_proj = nn.Parameter(torch.empty(HIDDEN_SIZE, FFN_HIDDEN_SIZE // tp_size, device=device_tp))
        self.up_proj = nn.Parameter(torch.empty(HIDDEN_SIZE, FFN_HIDDEN_SIZE // tp_size, device=device_tp))
        self.down_proj = nn.Parameter(torch.empty(FFN_HIDDEN_SIZE // tp_size, HIDDEN_SIZE, device=device_tp))

        for param in [
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.o_proj,
            self.gate_proj,
            self.up_proj,
            self.down_proj,
        ]:
            nn.init.normal_(param, mean=0.0, std=0.02)

    def forward(self, hidden_states: torch.Tensor, tp_group: object) -> torch.Tensor:
        """显式实现 TP 计算和通信。"""
        q_local = hidden_states @ self.q_proj
        k_local = hidden_states @ self.k_proj
        v_local = hidden_states @ self.v_proj
        attn_local = q_local + k_local + v_local
        attn_local = attn_local @ self.o_proj

        # attention 输出需要手工 AllReduce。
        attn_output = AllReduce(attn_local, tp_group)

        gate_local = hidden_states @ self.gate_proj
        up_local = hidden_states @ self.up_proj
        mlp_local = (gate_local.sigmoid() * gate_local) * up_local
        mlp_local = mlp_local @ self.down_proj

        # MLP 输出也需要手工 AllReduce。
        mlp_output = AllReduce(mlp_local, tp_group)
        return attn_output + mlp_output


class LlamaModel(nn.Module):
    """手动显式 PP/TP 并行模型。"""

    def __init__(self, config: dict[str, int], pp_info: dict[str, list[int]], tp_info: dict[str, int]) -> None:
        super().__init__()
        self.config = config
        self.pp_info = pp_info
        self.tp_info = tp_info
        self.device_tp = torch.device("cuda:0")
        self.tp_group = object()
        self.pp_first_stage = get_pp_rank() == 0
        self.pp_last_stage = get_pp_rank() == len(pp_info["num_layers"]) - 1

        self.emb = LlamaEmbedding()
        self.layers = self.build_layers(pp_info, tp_info)
        self.final_norm = nn.LayerNorm(HIDDEN_SIZE)
        self.lm_head = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE, bias=False)

    def build_layers(self, pp_info: dict[str, list[int]], tp_info: dict[str, int]) -> nn.ModuleList:
        """按照 PP/TP 信息手工构造本 stage 的层。"""
        rank_id = get_pp_rank()
        pp_num_layers = pp_info["num_layers"][rank_id]
        layers = nn.ModuleList()

        # 每一层都需要开发者显式手工做 attention/MLP shard。
        for _ in range(pp_num_layers):
            layers.append(LlamaAttentionBlock(tp_info=tp_info, device_tp=self.device_tp))
        return layers

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """显式实现 PP/TP 前向逻辑。"""
        if self.pp_first_stage:
            hidden_states = self.emb(input_ids)
        else:
            hidden_states = pp_recv((input_ids.shape[0], SEQ_LENGTH, HIDDEN_SIZE), self.device_tp)

        # 每一层都显式调用 TP 计算和通信。
        for layer in self.layers:
            hidden_states = layer(hidden_states, self.tp_group)

        output_layer = self.final_norm(hidden_states)
        if not self.pp_last_stage:
            pp_send(output_layer)
            return output_layer
        return self.lm_head(output_layer)


def build_optimizer(model: LlamaModel) -> torch.optim.Optimizer:
    """手工为切分参数构造优化器。"""
    return torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.1)


def train_step() -> None:
    """手工训练入口。"""
    config = {
        "hidden_size": HIDDEN_SIZE,
        "ffn_hidden_size": FFN_HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "num_heads": NUM_HEADS,
    }
    pp_info = build_pp_info(NUM_LAYERS, PP_SIZE)
    tp_info = build_tp_info(TP_SIZE)

    model = LlamaModel(config=config, pp_info=pp_info, tp_info=tp_info)
    optimizer = build_optimizer(model)

    input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LENGTH), device=torch.device("cuda:0"))
    logits = model(input_ids)
    loss = logits.float().mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


if __name__ == "__main__":
    train_step()
