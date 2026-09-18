"""Frozen ARC-AGI-3 public interactive-game benchmark."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from benchmarks.eval import dataset
from benchmarks.eval.environments.arc_agi_3 import (
    ARC_AGI_VERSION,
    ARCENGINE_VERSION,
    DEFAULT_ACTION_BUDGET,
    FROZEN_PUBLIC_GAME_IDS,
    MAX_BATCH_ACTIONS,
    PILOT_GAME_IDS,
    WRAPPER_REVISION,
    ArcAgi3Adapter,
    corpus_metadata_hashes,
    preflight_arc3_corpus,
)
from benchmarks.eval.environments.base import EnvironmentEpisode
from benchmarks.eval.tasks.environment import grade_environment
from benchmarks.eval.types import Dataset, Example, Prediction, Score

TASK_PROMPT = """\
Play the ARC-AGI-3 game through the shared host environment and solve as many
levels as possible, ideally reaching WIN. First inspect the frame and available
actions. Use only the host-proxy operations observe, status, and act.

Actions are ACTION1..ACTION7 or RESET. ACTION6 is a click and requires integer
x and y coordinates in 0..63; every other action requires empty data. Submit at
most 20 actions per act call. Root actions and delegated probe actions consume
one shared 500-action budget. Reassess whenever a level changes or a reset
occurs. Finish with a concise JSON summary; the official local scorecard's RHAE,
not the text answer, is the primary score.
"""

SEARCH_TASK_PROMPT = """\
Play this ARC-AGI-3 microgame by searching several genuinely different action
trajectories. First inspect the frame and legal actions. Use the host-proxy
operations observe, status, and act; ACTION6 is a click with integer x and y
coordinates in 0..63, while every other action uses empty data. An act call may
contain at most 20 actions, and root plus probe actions share one short budget.

Explore diverse hypotheses or routes instead of making near-duplicate probes.
After evaluating a useful branch, call submit_candidate with a descriptive
label. Use list_candidates to compare the evidence from all submitted
trajectories. Finally, the root agent must explicitly call
select_candidate(candidate_id) for the trajectory to receive native scoring;
the evaluator never silently chooses the best child. Finish with a concise JSON
summary. Native ARC scorecard RHAE and level progress, not the text answer, are
the score.
"""

# This manifest is deliberately derived only from frozen public corpus
# structure, never from model rollouts. Games qualify when every published
# per-level baseline action count is at most 160. The two lowest maximum
# horizons (ties broken by full game ID) form development; all other qualifying
# games are held out.
SEARCH_MANIFEST_REVISION = "arc3-public-baseline-max-160-v1"
SEARCH_BASELINE_MAX_ACTIONS = {
    "cd82-fb555c5d": 55,
    "ft09-0d8bbf25": 65,
    "lp85-305b61c3": 159,
    "r11l-495a7899": 52,
    "sb26-7fbdac44": 58,
    "sc25-635fd71a": 143,
    "sp80-589a99af": 152,
    "su15-1944f8ab": 115,
    "tn36-ef4dde99": 72,
    "tr87-cd924810": 146,
    "tu93-0768757b": 123,
    "vc33-5430563c": 152,
}
SEARCH_ELIGIBLE_GAME_IDS = tuple(
    sorted(
        SEARCH_BASELINE_MAX_ACTIONS,
        key=lambda game_id: (SEARCH_BASELINE_MAX_ACTIONS[game_id], game_id),
    )
)
SEARCH_PILOT_GAME_IDS = SEARCH_ELIGIBLE_GAME_IDS[:2]
SEARCH_FULL_GAME_IDS = SEARCH_ELIGIBLE_GAME_IDS[2:]
SEARCH_HELDOUT_GAME_IDS = SEARCH_FULL_GAME_IDS
SEARCH_RESTRAINT_GAME_IDS = frozenset({SEARCH_PILOT_GAME_IDS[-1], SEARCH_FULL_GAME_IDS[0]})
SEARCH_ACTION_BUDGET = 160
SEARCH_CANDIDATE_BUDGET = 4

SEARCH_SELECTION_PROVENANCE = {
    "population": "frozen_arc_agi_3_public_25",
    "source_field": "public_corpus_metadata.baseline_actions",
    "eligibility_rule": "max(baseline_actions) <= 160",
    "split_rule": (
        "development is the two eligible games with the lowest maximum per-level "
        "baseline horizon; heldout is every other eligible game; ties use full game ID"
    ),
    "model_outputs_used": False,
}


@dataset(
    "arc-agi-3",
    tags=["interactive", "environment", "visual"],
)
class ArcAgi3Dataset(Dataset):
    """The frozen public 25, defaulting to the representative three-game pilot."""

    def __init__(
        self,
        games: str | Sequence[str] | None = None,
        *,
        corpus: str | Path | None = None,
        action_budget: int = DEFAULT_ACTION_BUDGET,
        candidate_budget: int = 0,
        manifest_split: str | None = None,
    ) -> None:
        self.game_ids = _select_games(games) if games is not None else None
        self.corpus = corpus
        self.action_budget = action_budget
        self.candidate_budget = candidate_budget
        self.manifest_split = manifest_split

    def examples(self, *, split: str, limit: int | None, seed: int) -> list[Example]:
        if split not in {"test", "public", "pilot", "development", "heldout"}:
            raise ValueError("ARC-AGI-3 exposes only the frozen public test split")
        game_ids = (
            self.game_ids
            if self.game_ids is not None
            else FROZEN_PUBLIC_GAME_IDS
            if split == "full"
            else PILOT_GAME_IDS
        )
        selected = game_ids if limit is None else game_ids[:limit]
        # This is an intentional preflight, never a materializer: no network
        # fallback exists anywhere in this task or environment adapter.
        corpus = preflight_arc3_corpus(selected, corpus=self.corpus)
        return [self._example(game_id, seed=seed, corpus=corpus) for game_id in selected]

    def score(self, example: Example, prediction: Prediction) -> Score:
        grade = grade_environment(
            prediction,
            native_keys=("native_score", "rhae"),
            bounded=False,
        )
        primary = grade.view.primary
        terminal = str(primary.get("terminal", primary.get("state", "")))
        levels_completed = primary.get("levels_completed")
        return Score(
            value=grade.value,
            correct=terminal == "WIN" and grade.selected,
            details=grade.details(
                error=prediction.error,
                native_scorer="arc_agi_scorecard_rhae",
                rhae=grade.value,
                terminal=terminal or None,
                levels_completed=levels_completed,
                actions=grade.view.accepted_actions,
                replay_actions=primary.get("replay_actions", 0),
            ),
        )

    def make_episode(
        self,
        example: Example,
        *,
        role: str = "primary",
        arcade_factory: Any = None,
    ) -> EnvironmentEpisode:
        """Open the runner-neutral episode represented by ``example``."""

        del role  # The base episode assigns primary/probe roles from the graph.
        environment = example.metadata.get("environment", {})
        if not isinstance(environment, dict) or environment.get("kind") != "arc_agi_3":
            raise ValueError("example is not an ARC-AGI-3 environment task")
        adapter = ArcAgi3Adapter(
            str(environment["game_id"]),
            seed=int(environment["seed"]),
            corpus=str(environment["corpus"]),
            max_actions=int(environment["max_actions"]),
            metadata_hashes=environment["corpus_metadata_hashes"],
            arcade_factory=arcade_factory,
        )
        return EnvironmentEpisode(
            adapter,
            max_actions=int(environment["max_actions"]),
            episode_id=example.id,
            candidate_budget=int(environment.get("candidate_budget", 0)),
        )

    # Runner-neutral episode APIs use both spellings in downstream harnesses.
    create_episode = make_episode
    episode = make_episode

    def _example(self, game_id: str, *, seed: int, corpus: Path) -> Example:
        return Example(
            id=f"arc_agi_3_{game_id}",
            prompt=SEARCH_TASK_PROMPT if self.candidate_budget else TASK_PROMPT,
            expected={"terminal": "WIN"},
            context={
                "game_id": game_id,
                "seed": str(seed),
                "protocol": "observe/status/act",
            },
            metadata={
                "source": "Fchaubard/prime-agent-arc-agi-playground",
                "source_id": game_id,
                "revision": WRAPPER_REVISION,
                "license": "MIT",
                "domain": "arc_agi_3",
                "split": "public",
                "source_split": "public",
                "route_hypothesis": ("branch-and-select" if self.candidate_budget else "dependent"),
                "native_scorer": "arc_agi_scorecard_rhae",
                "arc_agi_version": ARC_AGI_VERSION,
                "arcengine_version": ARCENGINE_VERSION,
                **(
                    {
                        "benchmark": "visual_branch_and_select",
                        "benchmark_variant": "arc_agi_3_microgame",
                        "benchmark_split": self.manifest_split,
                        "manifest_revision": SEARCH_MANIFEST_REVISION,
                        "selection_provenance": dict(SEARCH_SELECTION_PROVENANCE),
                        "baseline_max_actions": SEARCH_BASELINE_MAX_ACTIONS[game_id],
                        "candidate_budget": self.candidate_budget,
                        "action_budget": self.action_budget,
                        "restraint_control": game_id in SEARCH_RESTRAINT_GAME_IDS,
                    }
                    if self.candidate_budget
                    else {}
                ),
                "environment": {
                    "kind": "arc_agi_3",
                    "adapter": ("benchmarks.eval.environments.arc_agi_3:ArcAgi3EnvironmentAdapter"),
                    "game_id": game_id,
                    "seed": seed,
                    "corpus": str(corpus),
                    "corpus_metadata_hashes": corpus_metadata_hashes(
                        corpus,
                        (game_id,),
                    ),
                    "operation_mode": "OFFLINE",
                    "operations": ["observe", "status", "act"],
                    "action_budget": self.action_budget,
                    "max_actions": self.action_budget,
                    "max_batch": MAX_BATCH_ACTIONS,
                    **(
                        {
                            "candidate_budget": self.candidate_budget,
                            "candidate_operations": [
                                "submit_candidate",
                                "list_candidates",
                                "select_candidate",
                            ],
                            "selection_policy": "explicit_root_only",
                        }
                        if self.candidate_budget
                        else {}
                    ),
                    "shared_budget": "root_and_probes",
                    "fork": "fresh_same_seed_replay_immediate_parent_prefix",
                    "records": ["action", "frame", "replay"],
                },
            },
        )


def _select_games(games: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(games, str):
        label = games.strip().lower()
        if label == "pilot":
            return PILOT_GAME_IDS
        if label in {"all", "public"}:
            return FROZEN_PUBLIC_GAME_IDS
        if label in {"search_pilot", "search-development", "search_dev"}:
            return SEARCH_PILOT_GAME_IDS
        if label in {"search_full", "search-heldout", "search_heldout"}:
            return SEARCH_FULL_GAME_IDS
        selected = tuple(part.strip() for part in games.split(",") if part.strip())
    else:
        selected = tuple(games)
    if not selected:
        raise ValueError("select at least one ARC-AGI-3 game")
    unknown = [game_id for game_id in selected if game_id not in FROZEN_PUBLIC_GAME_IDS]
    if unknown:
        raise ValueError(f"unknown frozen ARC-AGI-3 game(s): {', '.join(unknown)}")
    if len(set(selected)) != len(selected):
        raise ValueError("ARC-AGI-3 game selection contains duplicates")
    return selected


# Explicit names make manifests discoverable without instantiating a Dataset.
FULL_PUBLIC_GAME_IDS = FROZEN_PUBLIC_GAME_IDS
REPRESENTATIVE_PILOT_GAME_IDS = PILOT_GAME_IDS
ArcAgi3TaskGraphDataset = ArcAgi3Dataset

__all__ = [
    "FULL_PUBLIC_GAME_IDS",
    "REPRESENTATIVE_PILOT_GAME_IDS",
    "SEARCH_ACTION_BUDGET",
    "SEARCH_BASELINE_MAX_ACTIONS",
    "SEARCH_CANDIDATE_BUDGET",
    "SEARCH_ELIGIBLE_GAME_IDS",
    "SEARCH_FULL_GAME_IDS",
    "SEARCH_HELDOUT_GAME_IDS",
    "SEARCH_MANIFEST_REVISION",
    "SEARCH_PILOT_GAME_IDS",
    "SEARCH_RESTRAINT_GAME_IDS",
    "SEARCH_SELECTION_PROVENANCE",
    "SEARCH_TASK_PROMPT",
    "TASK_PROMPT",
    "ArcAgi3Dataset",
    "ArcAgi3TaskGraphDataset",
]
