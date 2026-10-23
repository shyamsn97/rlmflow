"""Recursive Agent Optimization over one shared world per rollout.

A rollout is one world: forkable envs give each agent an independent episode
view while sharing mutable world state. Each agent is scored on its own task,
gains a bonus for its immediate children, and is compared against a
leave-one-out baseline over the other rollouts of the same root task.
"""

from __future__ import annotations

from rlmflow.rao.credit import (
    DELEGATION_BONUS,
    IncompleteTreeError,
    NodeScore,
    RewardFn,
    agent_id,
    assign_advantages,
    assign_depth_weights,
    score_tree,
)
from rlmflow.rao.env import (
    Env,
    EnvSession,
    EnvStep,
    EpisodeOver,
    ForkableEnv,
    env_tool,
    environment_rewards,
    open_env,
    plain,
)
from rlmflow.rao.export import Trajectory, read_jsonl, stats, trajectories, write_jsonl
from rlmflow.rao.rollout import (
    Budget,
    Collector,
    RolloutFlow,
    RolloutTree,
    TaskSpec,
    TurnSample,
)
from rlmflow.rao.trainer import (
    JsonlTrainer,
    Optimizer,
    RunConfig,
    TinkerConfig,
    TinkerSampler,
    TinkerTrainer,
    Trainer,
    tinker_loss_fn,
    train,
)

__all__ = [
    "DELEGATION_BONUS",
    "Budget",
    "Collector",
    "Env",
    "EnvSession",
    "EnvStep",
    "EpisodeOver",
    "ForkableEnv",
    "IncompleteTreeError",
    "JsonlTrainer",
    "NodeScore",
    "Optimizer",
    "RolloutFlow",
    "RolloutTree",
    "RewardFn",
    "RunConfig",
    "TaskSpec",
    "TinkerConfig",
    "TinkerSampler",
    "TinkerTrainer",
    "Trainer",
    "Trajectory",
    "TurnSample",
    "agent_id",
    "assign_advantages",
    "assign_depth_weights",
    "env_tool",
    "environment_rewards",
    "open_env",
    "plain",
    "read_jsonl",
    "score_tree",
    "stats",
    "tinker_loss_fn",
    "train",
    "trajectories",
    "write_jsonl",
]
