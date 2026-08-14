"""Recursive Agent Optimization over one shared environment per rollout.

A rollout is one episode: the root and every subagent act on the same env. Each
agent is scored on the env reward its own actions earned, gains a bonus for its
immediate children, and is compared against a leave-one-out baseline over the
other rollouts of the same task. See ``docs/internal/rao_design.md``.
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
    OpenEnvAdapter,
    env_tool,
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
    "OpenEnvAdapter",
    "RolloutFlow",
    "RolloutTree",
    "TaskSpec",
    "Trajectory",
    "TurnSample",
    "agent_id",
    "assign_advantages",
    "assign_depth_weights",
    "env_tool",
    "plain",
    "read_jsonl",
    "score_tree",
    "stats",
    "trajectories",
    "write_jsonl",
]
