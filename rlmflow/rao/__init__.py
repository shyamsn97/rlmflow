"""Recursive Agent Optimization over one shared environment per rollout.

A rollout is one episode: the root and every subagent act on the same env. Each
agent is scored on the env reward its own actions earned, gains a bonus for its
immediate children, and is compared against a leave-one-out baseline over the
other rollouts of the same task. See ``docs/research/rao_implementation_plan.md``.
"""

from __future__ import annotations

from rlmflow.rao.credit import (
    DELEGATION_BONUS,
    IncompleteTreeError,
    NodeScore,
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
    env_tool,
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

__all__ = [
    "DELEGATION_BONUS",
    "Budget",
    "Collector",
    "Env",
    "EnvSession",
    "EnvStep",
    "EpisodeOver",
    "IncompleteTreeError",
    "NodeScore",
    "RolloutFlow",
    "RolloutTree",
    "TaskSpec",
    "Trajectory",
    "TurnSample",
    "agent_id",
    "assign_advantages",
    "assign_depth_weights",
    "env_tool",
    "open_env",
    "plain",
    "read_jsonl",
    "score_tree",
    "stats",
    "trajectories",
    "write_jsonl",
]
