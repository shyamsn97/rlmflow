"""Local reward, delegation credit, advantages, and depth weights.

Pure functions over finished ``AgentStart`` trees: no env, no model, no I/O.
This is the part the RAO paper is actually about, so it stays testable on
hand-built trees.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rlmflow.graph.nodes import AgentStart

#: Weight on the mean local score of an agent's immediate children.
DELEGATION_BONUS = 0.4


class IncompleteTreeError(RuntimeError):
    """Raised when scoring a tree that still holds a non-terminal agent.

    A terminal root is not a complete tree. A delegation bonus computed over
    children that are still running is silently wrong, so scoring refuses
    rather than guessing.
    """


@dataclass
class NodeScore:
    """One agent's credit.

    Mutable on purpose: reward, advantage, and weight are filled in by three
    separate stages, the last two of which need the whole batch to be known.
    """

    agent_id: str
    depth: int
    local: float
    delegation: float = 0.0
    reward: float = 0.0
    advantage: float = 0.0
    weight: float = 1.0

    @property
    def credit(self) -> float:
        """What the trainer scales this trajectory's gradient by."""
        return self.advantage * self.weight


def agent_id(agent: AgentStart) -> str:
    """The readable, tree-unique id used to key scores: ``root.child0.child1``."""
    return agent.config.path


def agents(root: AgentStart) -> list[AgentStart]:
    return [node for node in root.walk() if isinstance(node, AgentStart)]


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def score_tree(
    root: AgentStart,
    local: Mapping[str, float],
    *,
    delegation_bonus: float = DELEGATION_BONUS,
) -> dict[str, NodeScore]:
    """Score every agent in a finished tree, keyed by :func:`agent_id`.

    ``local`` must cover every agent. A missing entry is a bug in the caller's
    scoring, not a zero, so it raises.
    """
    found = agents(root)
    unfinished = [agent_id(agent) for agent in found if not agent.terminal]
    if unfinished:
        raise IncompleteTreeError(f"cannot score a tree with agents still running: {unfinished}")

    missing = [agent_id(agent) for agent in found if agent_id(agent) not in local]
    if missing:
        raise KeyError(f"no local score for {missing}")

    scores: dict[str, NodeScore] = {}
    for agent in found:
        # Each immediate child's *local* score, not its augmented reward, so a
        # successful grandchild credits its own parent and stops there.
        delegation = mean([local[agent_id(child)] for child in agent.sub_agents])
        own = local[agent_id(agent)]
        scores[agent_id(agent)] = NodeScore(
            agent_id=agent_id(agent),
            depth=agent.config.depth,
            local=own,
            delegation=delegation,
            reward=own + delegation_bonus * delegation,
        )
    return scores


def assign_advantages(group: Sequence[tuple[str, dict[str, NodeScore]]]) -> None:
    """Set every trajectory's advantage from a leave-one-out root baseline.

    ``group`` is one task's rollouts as ``(root_id, scores)``. The baseline is
    the mean root reward of the *other* rollouts, so it is shared within a tree
    while the reward is not: a weak child can carry a negative advantage under a
    succeeding root. One rollout has no baseline, so its advantages stay zero —
    fine for plumbing, but no policy gradient.
    """
    rewards = [scores[root_id].reward for root_id, scores in group]
    if len(group) < 2:
        for _root_id, scores in group:
            for score in scores.values():
                score.advantage = 0.0
        return

    total = sum(rewards)
    for (_root_id, scores), own in zip(group, rewards, strict=True):
        baseline = (total - own) / (len(group) - 1)
        for score in scores.values():
            score.advantage = score.reward - baseline


def depth_counts(batch: Sequence[dict[str, NodeScore]]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for scores in batch:
        for score in scores.values():
            counts[score.depth] = counts.get(score.depth, 0) + 1
    return counts


def assign_depth_weights(batch: Sequence[dict[str, NodeScore]]) -> dict[int, float]:
    """Reweight each depth by inverse frequency across the whole batch.

    Recursive trees hold far more deep trajectories than roots. This keeps each
    represented depth's aggregate influence comparable without changing which
    trajectories have positive or negative advantage.
    """
    counts = depth_counts(batch)
    if not counts:
        return {}
    average = sum(counts.values()) / len(counts)
    weights = {depth: average / count for depth, count in counts.items()}
    for scores in batch:
        for score in scores.values():
            score.weight = weights[score.depth]
    return weights


__all__ = [
    "DELEGATION_BONUS",
    "IncompleteTreeError",
    "NodeScore",
    "agent_id",
    "agents",
    "assign_advantages",
    "assign_depth_weights",
    "depth_counts",
    "mean",
    "score_tree",
]
