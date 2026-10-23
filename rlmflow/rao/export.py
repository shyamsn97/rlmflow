"""One training example per agent trajectory, and its JSONL form."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rlmflow.graph import reads
from rlmflow.rao.credit import NodeScore, agent_id
from rlmflow.rao.rollout import RolloutFlow, RolloutTree, TurnSample


@dataclass(frozen=True)
class Trajectory:
    """One agent's training example: what it sampled, and what credit it earned."""

    task_id: str
    rollout_id: str
    agent_id: str
    depth: int
    goal: str
    turns: list[TurnSample]
    score: NodeScore
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def credit(self) -> float:
        return self.score.credit

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["credit"] = self.credit
        return data


def trajectories(trees: Sequence[RolloutTree], flow: RolloutFlow) -> list[Trajectory]:
    """Every scored, recorded agent trajectory in these rollouts.

    A tree that never reached the barrier is dropped whole, and an agent whose
    client did not record tokens is dropped rather than exported empty — an
    example with no sampled tokens has nothing to reinforce.
    """
    out: list[Trajectory] = []
    for tree in trees:
        if not tree.usable:
            continue
        g = tree.graph
        for agent in tree.agents:
            key = agent_id(g, agent)
            turns = [turn for turn in flow.turns(g, agent) if turn.trainable]
            if not turns:
                continue
            out.append(
                Trajectory(
                    task_id=tree.task.id,
                    rollout_id=tree.rollout_id,
                    agent_id=key,
                    depth=tree.scores[key].depth,
                    goal=agent.content,
                    turns=turns,
                    score=tree.scores[key],
                    metadata={
                        "env_steps": sum(step.agent_id == key for step in tree.session.steps),
                        "episode_reward": tree.session.reward_for(key),
                        "children": len(reads.children(g, agent)),
                        "result": reads.result(g, agent),
                    },
                )
            )
    return out


def write_jsonl(items: Sequence[Trajectory], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item.to_dict(), default=str) + "\n")
    return target


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stats(items: Sequence[Trajectory], trees: Sequence[RolloutTree]) -> dict[str, Any]:
    """Batch metrics worth watching: they catch a collapsed batch early."""
    depths: dict[int, int] = {}
    for item in items:
        depths[item.depth] = depths.get(item.depth, 0) + 1
    usable = [tree for tree in trees if tree.usable]
    dropped: dict[str, int] = {}
    for tree in trees:
        if tree.error is not None:
            dropped[tree.error] = dropped.get(tree.error, 0) + 1
    rewards = [tree.scores[tree.root_id].reward for tree in usable]
    locals_by_depth: dict[int, list[float]] = {}
    for tree in usable:
        for score in tree.scores.values():
            locals_by_depth.setdefault(score.depth, []).append(score.local)
    agent_counts = [len(tree.agents) for tree in usable]
    return {
        "trajectories": len(items),
        "rollouts": len(trees),
        "usable_rollouts": len(usable),
        "dropped": dropped,
        "by_depth": depths,
        "success_by_depth": {
            str(depth): sum(values) / len(values)
            for depth, values in sorted(locals_by_depth.items())
        },
        "mean_root_reward": sum(rewards) / len(rewards) if rewards else 0.0,
        "mean_agents": sum(agent_counts) / len(agent_counts) if agent_counts else 0.0,
        "max_agent_depth": max(locals_by_depth, default=0),
        "delegations": sum(max(count - 1, 0) for count in agent_counts),
        "mean_env_steps": (
            sum(len(tree.session.steps) for tree in usable) / len(usable) if usable else 0.0
        ),
        "sampled_tokens": sum(len(turn.sampled_tokens) for item in items for turn in item.turns),
    }


__all__ = ["Trajectory", "read_jsonl", "stats", "trajectories", "write_jsonl"]
