"""Generated and Platoon-compatible TextCraft datasets.

The checked-in JSONL files contain only generation parameters. Recipes are
regenerated deterministically, keeping benchmark data in ``examples/`` without
shipping hundreds of redundant recipe documents in the library package.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from examples.research.textcraft.recipes import Recipe, RecipeBook, quantities
from rlmflow.rao import TaskSpec

DATA_DIR = Path(__file__).with_name("data")


@dataclass(frozen=True)
class TextCraftTask:
    id: str
    target_items: dict[str, int]
    initial_inventory: dict[str, int]
    recipe_book: RecipeBook
    max_steps: int = 64
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_spec(self) -> TaskSpec:
        target = ", ".join(f"{count}x {item}" for item, count in self.target_items.items())
        return TaskSpec(
            id=self.id,
            goal=f"""Craft these target items: {target}.

Use `await env_step(op="inspect", item="...")` to discover recipes,
`await env_step(op="inventory")` to inspect shared inventory, and
`await env_step(op="craft", item="...", quantity=N)` to craft. Every agent in
the tree shares one atomic inventory. Delegate independent ingredients with
`launch_agent(task, target_items={{"item": N}})`; structured
`target_items` are required for exact child verification. Retrieve useful child
results, then call `await env_step(op="finish")` and `submit(...)`.""",
            config={"textcraft_task": self},
            metadata={
                **self.metadata,
                "target_items": dict(self.target_items),
                "initial_inventory": dict(self.initial_inventory),
            },
        )


def generate_task(
    seed: int,
    *,
    depth: int = 3,
    branching: int = 2,
    max_steps: int = 64,
) -> TextCraftTask:
    """Generate an opaque balanced recipe tree with exact leaf inventory."""
    if depth < 1:
        raise ValueError("depth must be at least 1")
    if branching < 1:
        raise ValueError("branching must be at least 1")
    rng = random.Random(seed)
    serial = 0
    recipes: list[Recipe] = []
    inventory: Counter[str] = Counter()

    def build(level: int) -> str:
        nonlocal serial
        serial += 1
        name = f"item_{seed:05d}_{serial:03d}_{rng.randrange(10**6):06d}"
        if level == 0:
            inventory[name] += 1
            return name
        ingredients = Counter(build(level - 1) for _ in range(branching))
        recipes.append(Recipe.simple(name, ingredients))
        return name

    target = build(depth)
    return TextCraftTask(
        id=f"textcraft-synth-{seed:05d}-d{depth}",
        target_items={target: 1},
        initial_inventory=dict(inventory),
        recipe_book=RecipeBook.from_recipes(recipes),
        max_steps=max_steps,
        metadata={"seed": seed, "depth": depth, "branching": branching},
    )


def load_dataset(
    split: str = "train",
    *,
    data_dir: str | Path = DATA_DIR,
    limit: int | None = None,
) -> list[TextCraftTask]:
    path = Path(data_dir) / f"{split}.jsonl"
    tasks: list[TextCraftTask] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            tasks.append(
                generate_task(
                    int(row["seed"]),
                    depth=int(row.get("depth", 3)),
                    branching=int(row.get("branching", 2)),
                    max_steps=int(row.get("max_steps", 64)),
                )
            )
            if limit is not None and len(tasks) >= limit:
                break
    return tasks


def load_platoon_tasks(
    data_dir: str | Path,
    *,
    split: str = "train",
    limit: int | None = None,
) -> list[TextCraftTask]:
    """Load Platoon's task JSONL and adjacent ``recipes/`` directory."""
    root = Path(data_dir)
    book = RecipeBook.from_directory(root / "recipes")
    candidates = [
        root / f"textcraft_{split}.jsonl",
        root / f"textcraft_synth_{split}.jsonl",
        *sorted(root.glob(f"textcraft_synth_*_{split}.jsonl")),
    ]
    task_file = next((path for path in candidates if path.exists()), None)
    if task_file is None:
        raise FileNotFoundError(f"no TextCraft {split!r} JSONL found under {root}")
    tasks: list[TextCraftTask] = []
    with task_file.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            raw = json.loads(line)
            misc = raw.get("misc", {})
            targets = quantities(misc.get("target_items") or raw.get("target_items"))
            initial = quantities(misc.get("initial_inventory") or raw.get("initial_inventory"))
            if not targets:
                continue
            tasks.append(
                TextCraftTask(
                    id=str(raw.get("id") or raw.get("task_id") or f"{split}-{index}"),
                    target_items=targets,
                    initial_inventory=initial,
                    recipe_book=book,
                    max_steps=int(raw.get("max_steps", 64)),
                    metadata={"source": str(task_file), **misc},
                )
            )
            if limit is not None and len(tasks) >= limit:
                break
    return tasks


__all__ = [
    "DATA_DIR",
    "TextCraftTask",
    "generate_task",
    "load_dataset",
    "load_platoon_tasks",
]
