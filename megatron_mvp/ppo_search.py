"""基于 clipped PPO 的分解式并行策略搜索控制器。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
from torch.distributions import Categorical

from megatron_mvp.types import ModelSpec
from megatron_mvp.types import RuntimeEnvironment
from megatron_mvp.types import SearchCandidate


@dataclass(frozen=True)
class DecisionNode:
    """描述一个 PPO 搜索决策节点。"""

    node_name: str
    choices: tuple[str, ...]


@dataclass(frozen=True)
class PpoControllerConfig:
    """描述 clipped PPO 控制器超参数。"""

    hidden_dim: int = 96
    rollout_episodes: int = 12
    update_epochs: int = 4
    clip_ratio: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    learning_rate: float = 3.0e-3


@dataclass(frozen=True)
class ControllerEvaluation:
    """描述控制器一次完整策略赋值后的评估结果。"""

    reward: float
    assignment: dict[str, str]
    feasible_current_candidates: list[SearchCandidate]
    scalable_future_candidates: list[SearchCandidate]
    rejected_reasons: list[str]


@dataclass(frozen=True)
class EpisodeRollout:
    """描述一次 PPO rollout 采样结果。"""

    states: list[torch.Tensor]
    actions: list[int]
    valid_action_counts: list[int]
    log_probs: list[float]
    values: list[float]
    rewards: list[float]
    dones: list[float]


EvaluationFn = Callable[[dict[str, str]], ControllerEvaluation]


class PlacementController(nn.Module):
    """分解式策略搜索控制器。"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """返回动作 logits 和状态价值。"""
        hidden = self.backbone(states)
        return self.policy_head(hidden), self.value_head(hidden).squeeze(-1)


def _masked_logits(logits: torch.Tensor, valid_action_count: int) -> torch.Tensor:
    """对超过有效动作数的 logits 做硬掩码。"""
    if valid_action_count >= logits.shape[-1]:
        return logits
    masked = logits.clone()
    masked[..., valid_action_count:] = -1.0e9
    return masked


def _build_state_tensor(
    node_index: int,
    decision_nodes: list[DecisionNode],
    action_matrix: torch.Tensor,
    spec: ModelSpec,
    env: RuntimeEnvironment,
) -> torch.Tensor:
    """构造控制器输入状态。"""
    max_action_dim = action_matrix.shape[-1]
    current_node = decision_nodes[node_index]
    base_features = [
        node_index / max(len(decision_nodes) - 1, 1),
        len(current_node.choices) / max_action_dim,
        spec.hidden_size / 8192.0,
        spec.num_heads / 128.0,
        spec.num_kv_heads / 128.0,
        spec.num_layers / 128.0,
        spec.estimated_total_params / 20_000_000_000.0,
        env.device_count / 16.0,
    ]
    feature_vector = base_features + action_matrix.flatten().tolist()
    return torch.tensor(feature_vector, dtype=torch.float32)


def _collect_episode(
    controller: PlacementController,
    decision_nodes: list[DecisionNode],
    spec: ModelSpec,
    env: RuntimeEnvironment,
    evaluate_assignment_fn: EvaluationFn,
) -> tuple[EpisodeRollout, ControllerEvaluation]:
    """采集一条 rollout 轨迹。"""
    max_action_dim = max(len(node.choices) for node in decision_nodes)
    action_matrix = torch.zeros(
        (len(decision_nodes), max_action_dim),
        dtype=torch.float32,
    )
    states: list[torch.Tensor] = []
    actions: list[int] = []
    valid_action_counts: list[int] = []
    log_probs: list[float] = []
    values: list[float] = []
    rewards: list[float] = []
    dones: list[float] = []
    assignment: dict[str, str] = {}

    for node_index, decision_node in enumerate(decision_nodes):
        state_tensor = _build_state_tensor(
            node_index=node_index,
            decision_nodes=decision_nodes,
            action_matrix=action_matrix,
            spec=spec,
            env=env,
        )
        logits, value = controller(state_tensor.unsqueeze(0))
        masked = _masked_logits(logits.squeeze(0), len(decision_node.choices))
        distribution = Categorical(logits=masked)
        action = int(distribution.sample().item())

        assignment[decision_node.node_name] = decision_node.choices[action]
        action_matrix[node_index, action] = 1.0

        states.append(state_tensor)
        actions.append(action)
        valid_action_counts.append(len(decision_node.choices))
        log_probs.append(float(distribution.log_prob(torch.tensor(action)).item()))
        values.append(float(value.squeeze(0).item()))
        rewards.append(0.0)
        dones.append(0.0)

    evaluation = evaluate_assignment_fn(assignment)
    rewards[-1] = evaluation.reward
    dones[-1] = 1.0

    return (
        EpisodeRollout(
            states=states,
            actions=actions,
            valid_action_counts=valid_action_counts,
            log_probs=log_probs,
            values=values,
            rewards=rewards,
            dones=dones,
        ),
        evaluation,
    )


def _compute_advantages(
    rollout: EpisodeRollout,
    controller_config: PpoControllerConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """基于 GAE 计算 returns 和 advantages。"""
    returns = torch.zeros(len(rollout.rewards), dtype=torch.float32)
    advantages = torch.zeros(len(rollout.rewards), dtype=torch.float32)
    next_return = 0.0
    next_value = 0.0
    next_advantage = 0.0

    for index in reversed(range(len(rollout.rewards))):
        mask = 1.0 - rollout.dones[index]
        delta = (
            rollout.rewards[index]
            + controller_config.gamma * next_value * mask
            - rollout.values[index]
        )
        advantage = (
            delta
            + controller_config.gamma * controller_config.gae_lambda * mask * next_advantage
        )
        current_return = rollout.rewards[index] + controller_config.gamma * next_return * mask

        advantages[index] = advantage
        returns[index] = current_return
        next_advantage = float(advantage)
        next_return = float(current_return)
        next_value = rollout.values[index]

    return returns, advantages


def _flatten_rollouts(
    rollouts: list[EpisodeRollout],
    controller_config: PpoControllerConfig,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """把多条 rollout 拼成 PPO 更新批次。"""
    all_states: list[torch.Tensor] = []
    all_actions: list[int] = []
    all_valid_action_counts: list[int] = []
    all_log_probs: list[float] = []
    all_returns: list[torch.Tensor] = []
    all_advantages: list[torch.Tensor] = []

    for rollout in rollouts:
        returns, advantages = _compute_advantages(rollout, controller_config)
        all_states.extend(rollout.states)
        all_actions.extend(rollout.actions)
        all_valid_action_counts.extend(rollout.valid_action_counts)
        all_log_probs.extend(rollout.log_probs)
        all_returns.append(returns)
        all_advantages.append(advantages)

    states_tensor = torch.stack(all_states)
    actions_tensor = torch.tensor(all_actions, dtype=torch.int64)
    valid_action_counts_tensor = torch.tensor(all_valid_action_counts, dtype=torch.int64)
    old_log_probs_tensor = torch.tensor(all_log_probs, dtype=torch.float32)
    returns_tensor = torch.cat(all_returns)
    advantages_tensor = torch.cat(all_advantages)
    advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (
        advantages_tensor.std(unbiased=False) + 1.0e-6
    )
    return (
        states_tensor,
        actions_tensor,
        valid_action_counts_tensor,
        old_log_probs_tensor,
        returns_tensor,
        advantages_tensor,
    )


def _run_ppo_update(
    controller: PlacementController,
    optimizer: torch.optim.Optimizer,
    states_tensor: torch.Tensor,
    actions_tensor: torch.Tensor,
    valid_action_counts_tensor: torch.Tensor,
    old_log_probs_tensor: torch.Tensor,
    returns_tensor: torch.Tensor,
    advantages_tensor: torch.Tensor,
    controller_config: PpoControllerConfig,
) -> None:
    """执行 clipped PPO 控制器更新。"""
    for _ in range(controller_config.update_epochs):
        logits, values = controller(states_tensor)
        masked_logits = logits.clone()
        action_dim = masked_logits.shape[-1]
        for action_index in range(action_dim):
            invalid_mask = valid_action_counts_tensor <= action_index
            masked_logits[invalid_mask, action_index] = -1.0e9

        distribution = Categorical(logits=masked_logits)
        new_log_probs = distribution.log_prob(actions_tensor)
        entropy = distribution.entropy().mean()

        ratio = torch.exp(new_log_probs - old_log_probs_tensor)
        unclipped_objective = ratio * advantages_tensor
        clipped_objective = torch.clamp(
            ratio,
            1.0 - controller_config.clip_ratio,
            1.0 + controller_config.clip_ratio,
        ) * advantages_tensor
        policy_loss = -torch.min(unclipped_objective, clipped_objective).mean()
        value_loss = torch.mean((returns_tensor - values) ** 2)
        loss = (
            policy_loss
            + controller_config.value_coef * value_loss
            - controller_config.entropy_coef * entropy
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def _run_greedy_controller(
    controller: PlacementController,
    decision_nodes: list[DecisionNode],
    spec: ModelSpec,
    env: RuntimeEnvironment,
    evaluate_assignment_fn: EvaluationFn,
) -> ControllerEvaluation:
    """用更新后的控制器执行一次贪心搜索。"""
    max_action_dim = max(len(node.choices) for node in decision_nodes)
    action_matrix = torch.zeros(
        (len(decision_nodes), max_action_dim),
        dtype=torch.float32,
    )
    assignment: dict[str, str] = {}

    for node_index, decision_node in enumerate(decision_nodes):
        state_tensor = _build_state_tensor(
            node_index=node_index,
            decision_nodes=decision_nodes,
            action_matrix=action_matrix,
            spec=spec,
            env=env,
        )
        logits, _ = controller(state_tensor.unsqueeze(0))
        masked = _masked_logits(logits.squeeze(0), len(decision_node.choices))
        action = int(torch.argmax(masked).item())
        assignment[decision_node.node_name] = decision_node.choices[action]
        action_matrix[node_index, action] = 1.0

    return evaluate_assignment_fn(assignment)


def run_rl_guided_ppo_search(
    spec: ModelSpec,
    env: RuntimeEnvironment,
    decision_nodes: list[DecisionNode],
    evaluate_assignment_fn: EvaluationFn,
    controller_config: PpoControllerConfig | None = None,
    seed_assignments: list[dict[str, str]] | None = None,
) -> tuple[list[SearchCandidate], list[SearchCandidate], list[str]]:
    """执行基于 clipped PPO 的在线搜索。"""
    if not decision_nodes:
        return [], [], ["RL_GUIDED_PPO 没有任何决策节点，无法构造控制器。"]

    controller_config = controller_config or PpoControllerConfig()
    torch.manual_seed(
        spec.hidden_size
        + spec.num_layers * 17
        + env.device_count * 31
        + len(decision_nodes) * 7
    )

    max_action_dim = max(len(node.choices) for node in decision_nodes)
    sample_state = _build_state_tensor(
        node_index=0,
        decision_nodes=decision_nodes,
        action_matrix=torch.zeros((len(decision_nodes), max_action_dim), dtype=torch.float32),
        spec=spec,
        env=env,
    )
    controller = PlacementController(
        state_dim=sample_state.numel(),
        action_dim=max_action_dim,
        hidden_dim=controller_config.hidden_dim,
    )
    optimizer = torch.optim.Adam(controller.parameters(), lr=controller_config.learning_rate)

    rollouts: list[EpisodeRollout] = []
    rollout_notes: list[str] = []
    best_valid_evaluation: ControllerEvaluation | None = None
    if seed_assignments:
        for assignment in seed_assignments:
            evaluation = evaluate_assignment_fn(assignment)
            rollout_notes.extend(evaluation.rejected_reasons[:1])
            if (
                evaluation.feasible_current_candidates
                or evaluation.scalable_future_candidates
            ) and (
                best_valid_evaluation is None
                or evaluation.reward > best_valid_evaluation.reward
            ):
                best_valid_evaluation = evaluation

    for _ in range(controller_config.rollout_episodes):
        rollout, evaluation = _collect_episode(
            controller=controller,
            decision_nodes=decision_nodes,
            spec=spec,
            env=env,
            evaluate_assignment_fn=evaluate_assignment_fn,
        )
        rollouts.append(rollout)
        rollout_notes.extend(evaluation.rejected_reasons[:2])
        if (
            evaluation.feasible_current_candidates
            or evaluation.scalable_future_candidates
        ) and (
            best_valid_evaluation is None
            or evaluation.reward > best_valid_evaluation.reward
        ):
            best_valid_evaluation = evaluation

    (
        states_tensor,
        actions_tensor,
        valid_action_counts_tensor,
        old_log_probs_tensor,
        returns_tensor,
        advantages_tensor,
    ) = _flatten_rollouts(rollouts, controller_config)
    _run_ppo_update(
        controller=controller,
        optimizer=optimizer,
        states_tensor=states_tensor,
        actions_tensor=actions_tensor,
        valid_action_counts_tensor=valid_action_counts_tensor,
        old_log_probs_tensor=old_log_probs_tensor,
        returns_tensor=returns_tensor,
        advantages_tensor=advantages_tensor,
        controller_config=controller_config,
    )

    final_evaluation = _run_greedy_controller(
        controller=controller,
        decision_nodes=decision_nodes,
        spec=spec,
        env=env,
        evaluate_assignment_fn=evaluate_assignment_fn,
    )
    final_is_valid = (
        final_evaluation.feasible_current_candidates
        or final_evaluation.scalable_future_candidates
    )
    if best_valid_evaluation is not None:
        if not final_is_valid:
            rollout_notes.append(
                "RL_GUIDED_PPO 贪心 assignment 没有可执行候选，回退到 rollout 中奖励最高的可执行 assignment。"
            )
            final_evaluation = best_valid_evaluation
        elif best_valid_evaluation.reward > final_evaluation.reward:
            rollout_notes.append(
                "RL_GUIDED_PPO 贪心 assignment 不是本轮最高奖励有效结果，回退到 rollout 中奖励更高的可执行 assignment。"
            )
            final_evaluation = best_valid_evaluation

    rejected_reasons = [
        "RL_GUIDED_PPO 使用在线 rollout + clipped PPO 更新控制器参数。",
        "RL_GUIDED_PPO 决策节点数=%d, 搜索空间维度=%s。"
        % (len(decision_nodes), [len(node.choices) for node in decision_nodes]),
        "RL_GUIDED_PPO rollout 数量=%d, update_epochs=%d。"
        % (controller_config.rollout_episodes, controller_config.update_epochs),
        "RL_GUIDED_PPO 最终 assignment=%s" % final_evaluation.assignment,
    ]
    rejected_reasons.extend(rollout_notes[:8])
    rejected_reasons.extend(final_evaluation.rejected_reasons)
    return (
        final_evaluation.feasible_current_candidates,
        final_evaluation.scalable_future_candidates,
        rejected_reasons,
    )
