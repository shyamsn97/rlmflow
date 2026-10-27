"""Gymnasium TextCraft environment and exact RAO scorer."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

try:
    import gymnasium as gym
except ImportError as exc:  # pragma: no cover - optional example dependency
    raise ImportError(
        "The TextCraft example needs Gymnasium. Install `rlmflow[textcraft]` "
        "(environment only) or `rlmflow[rao]` (with Tinker)."
    ) from exc

from examples.research.textcraft.dataset import TextCraftTask
from examples.research.textcraft.recipes import item_name, quantities
from rlmflow import Graph, Node
from rlmflow.rao import RolloutTree, TaskSpec, agent_id


class MappingSpace(gym.Space[dict[str, Any]]):
    """A Gym space for the JSON-compatible mappings used by agent tools."""

    def sample(self, mask: Any = None, probability: Any = None) -> dict[str, Any]:
        return {}

    def contains(self, value: Any) -> bool:
        return isinstance(value, Mapping)


class TextCraftEnv(gym.Env):
    """Quantity-aware crafting over one inventory.

    ``EnvSession`` serializes calls from recursive agents. Gym owns environment
    state and termination; rlmflow only adapts the standard tuple API.
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 1}
    action_space = MappingSpace()
    observation_space = MappingSpace()

    def __init__(
        self,
        task: TextCraftTask,
        *,
        inventory: Counter[str] | None = None,
    ) -> None:
        super().__init__()
        self.task = task
        self.inventory = inventory if inventory is not None else Counter()
        self._owns_inventory = inventory is None
        self.initial_inventory: Counter[str] = Counter()
        self.steps = 0
        self.finished = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)
        if self._owns_inventory:
            self.inventory.clear()
            self.inventory.update(self.task.initial_inventory)
        self.initial_inventory = Counter(self.inventory)
        self.steps = 0
        self.finished = False
        return self._observation("episode started"), {"task_id": self.task.id}

    def fork(self, task: str, args: dict[str, Any]) -> TextCraftEnv:
        """Create a child episode with its own target over the shared inventory."""
        try:
            targets = quantities(args.get("target_items"))
        except (TypeError, ValueError) as exc:
            raise ValueError("child target_items must map item names to positive counts") from exc
        if not targets:
            raise ValueError("TextCraft subagents require non-empty target_items")
        child_task = TextCraftTask(
            id=f"{self.task.id}:child",
            target_items=targets,
            initial_inventory=dict(self.inventory),
            recipe_book=self.task.recipe_book,
            max_steps=self.task.max_steps,
            metadata={"parent_task_id": self.task.id, "task": task},
        )
        return TextCraftEnv(child_task, inventory=self.inventory)

    def step(
        self, action: Mapping[str, Any]
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self.finished:
            return self._observation("episode already finished"), 0.0, True, False, {}
        self.steps += 1
        op = str(action.get("op", "")).lower()
        if op == "inventory":
            return self._transition("inventory")
        if op in {"inspect", "get_info"}:
            item = item_name(str(action.get("item", "")))
            return self._transition(
                f"{len(self.task.recipe_book.options(item))} recipe(s) for {item}",
                recipes=self.task.recipe_book.describe(item),
            )
        if op == "craft":
            return self._craft(
                item_name(str(action.get("item", ""))),
                max(int(action.get("quantity", 1)), 1),
            )
        if op == "finish":
            self.finished = True
            success = self.root_success()
            return (
                self._observation("success" if success else "targets are incomplete"),
                float(success),
                True,
                False,
                {"success": success},
            )
        return self._transition(f"unknown operation: {op!r}")

    def _craft(
        self, item: str, quantity: int
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        options = self.task.recipe_book.options(item)
        if not options:
            return self._transition(f"no recipe for {item}")
        for recipe in options:
            crafts = (quantity + recipe.result_count - 1) // recipe.result_count
            resolved = self._resolve_ingredients(recipe.ingredients, crafts)
            if resolved is None:
                continue
            for ingredient, count in resolved.items():
                self.inventory[ingredient] -= count
            made = crafts * recipe.result_count
            self.inventory[item] += made
            return self._transition(f"crafted {made}x {item}", produced={item: made})
        return self._transition(
            f"missing ingredients for {quantity}x {item}",
            recipes=self.task.recipe_book.describe(item),
        )

    def _resolve_ingredients(
        self, ingredients: Mapping[str, int], crafts: int
    ) -> Counter[str] | None:
        needed: Counter[str] = Counter()
        available = self.inventory.copy()
        for ingredient, per_craft in ingredients.items():
            count = per_craft * crafts
            if ingredient.startswith("tag:"):
                candidates = self.task.recipe_book.tags.get(ingredient.removeprefix("tag:"), ())
                candidate = next((name for name in candidates if available[name] >= count), None)
                if candidate is None:
                    return None
                ingredient = candidate
            if available[ingredient] < count:
                return None
            needed[ingredient] += count
            available[ingredient] -= count
        return needed

    def _transition(
        self, message: str, **extra: Any
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        truncated = self.steps >= self.task.max_steps
        if truncated:
            self.finished = True
        return self._observation(message, **extra), 0.0, False, truncated, {}

    def _observation(self, message: str, **extra: Any) -> dict[str, Any]:
        return {
            "message": message,
            "inventory": {name: count for name, count in sorted(self.inventory.items()) if count},
            "target_items": dict(self.task.target_items),
            "steps": self.steps,
            **extra,
        }

    def root_success(self) -> bool:
        return all(
            self.inventory[item] - self.initial_inventory[item] >= count
            for item, count in self.task.target_items.items()
        )

    def render(self) -> str:
        return str(self._observation("render"))


def _environment_steps(graph: Graph, root: Node) -> list[dict[str, Any]]:
    receipts = [
        receipt
        for agent in graph.tree(root)
        for node in graph.nodes(agent=agent.id)
        for receipt in node.metadata.get("environment.steps", [])
        if isinstance(receipt, dict)
    ]
    return sorted(receipts, key=lambda receipt: int(receipt.get("index", -1)))


def _safe_quantities(value: Any) -> dict[str, int]:
    """Parse model-authored quantities without letting malformed args abort a batch."""
    try:
        return quantities(value)
    except (TypeError, ValueError):
        return {}


def textcraft_graph_rewards(graph: Graph, root: Node) -> dict[str, float]:
    """Compute exact rewards for ``root``'s tree from its tool metadata."""
    steps = _environment_steps(graph, root)
    task_record = root.metadata.get("rollout.task", {})
    task_metadata = task_record.get("metadata", {}) if isinstance(task_record, dict) else {}
    root_targets = _safe_quantities(task_metadata.get("target_items"))
    initial = _safe_quantities(task_metadata.get("initial_inventory"))
    final_observation = steps[-1].get("observation", {}) if steps else {}
    final_inventory = _safe_quantities(
        final_observation.get("inventory") if isinstance(final_observation, Mapping) else {}
    )
    scores: dict[str, float] = {}
    for agent in graph.tree(root):
        path = agent_id(graph, agent)
        if agent is root:
            scores[path] = float(
                bool(root_targets)
                and all(
                    final_inventory.get(item, 0) - initial.get(item, 0) >= count
                    for item, count in root_targets.items()
                )
            )
            continue
        raw = agent.args.get("target_items")
        targets = _safe_quantities(raw if isinstance(raw, (Mapping, Sequence, str)) else {})
        made: Counter[str] = Counter()
        prefix = path + "."
        for step in steps:
            producer = step.get("agent_id")
            if producer != path and not (isinstance(producer, str) and producer.startswith(prefix)):
                continue
            observation = step.get("observation")
            if isinstance(observation, Mapping):
                made.update(quantities(observation.get("produced")))
        scores[path] = float(
            bool(targets) and all(made[item] >= count for item, count in targets.items())
        )
    return scores


def textcraft_local_scores(tree: RolloutTree) -> dict[str, float]:
    """Compatibility adapter for the old tree-plus-session scorer interface."""
    return textcraft_graph_rewards(tree.graph, tree.root)


async def open_textcraft(task: TaskSpec) -> TextCraftEnv:
    value = task.config.get("textcraft_task")
    if not isinstance(value, TextCraftTask):
        raise TypeError("TaskSpec.config['textcraft_task'] must be a TextCraftTask")
    return TextCraftEnv(value)


__all__ = [
    "MappingSpace",
    "TextCraftEnv",
    "open_textcraft",
    "textcraft_graph_rewards",
    "textcraft_local_scores",
]
