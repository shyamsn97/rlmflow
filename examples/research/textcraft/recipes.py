"""Recipe parsing compatible with Platoon's TextCraft data."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def item_name(value: str) -> str:
    if value.startswith("tag:"):
        return "tag:" + value.removeprefix("tag:").split(":", 1)[-1]
    return value.split(":", 1)[-1]


def quantities(value: Any) -> dict[str, int]:
    if value is None:
        return {}
    if isinstance(value, str):
        return {item_name(value): 1}
    if isinstance(value, Mapping):
        return {item_name(str(name)): int(count) for name, count in value.items() if int(count) > 0}
    return dict(Counter(item_name(str(name)) for name in value))


def _ingredient(entry: Any) -> str | None:
    if isinstance(entry, str):
        return item_name(entry)
    if not isinstance(entry, Mapping):
        return None
    if "item" in entry:
        return item_name(str(entry["item"]))
    if "tag" in entry:
        return "tag:" + item_name(str(entry["tag"]))
    return None


@dataclass(frozen=True)
class Recipe:
    result: str
    result_count: int
    ingredients: dict[str, int]
    source: str = ""

    @classmethod
    def from_minecraft(cls, data: Mapping[str, Any], source: str = "") -> Recipe:
        raw_result = data.get("result")
        if isinstance(raw_result, list):
            raw_result = raw_result[0] if raw_result else None
        if isinstance(raw_result, str):
            result, result_count = item_name(raw_result), 1
        elif isinstance(raw_result, Mapping):
            raw_item = raw_result.get("item") or raw_result.get("id")
            if not raw_item:
                raise ValueError(f"recipe {source!r} has no result item")
            result = item_name(str(raw_item))
            result_count = int(raw_result.get("count", 1))
        else:
            raise ValueError(f"recipe {source!r} has an invalid result")

        ingredients: Counter[str] = Counter()
        if str(data.get("type", "")).endswith("crafting_shaped") or "pattern" in data:
            key = data.get("key", {})
            for row in data.get("pattern", []):
                for symbol in str(row):
                    ingredient = _ingredient(key.get(symbol)) if symbol != " " else None
                    if ingredient:
                        ingredients[ingredient] += 1
        else:
            for raw in data.get("ingredients", []):
                ingredient = _ingredient(raw)
                if ingredient:
                    ingredients[ingredient] += 1
        if not ingredients:
            raise ValueError(f"recipe {source!r} has no ingredients")
        return cls(result, result_count, dict(ingredients), source)

    @classmethod
    def simple(
        cls, result: str, ingredients: Mapping[str, int], *, result_count: int = 1
    ) -> Recipe:
        return cls(
            item_name(result),
            int(result_count),
            quantities(ingredients),
            source=f"generated:{item_name(result)}",
        )


@dataclass(frozen=True)
class RecipeBook:
    recipes: dict[str, tuple[Recipe, ...]]
    tags: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @classmethod
    def from_recipes(cls, recipes: Iterable[Recipe]) -> RecipeBook:
        grouped: dict[str, list[Recipe]] = defaultdict(list)
        all_items: set[str] = set()
        all_tags: set[str] = set()
        for recipe in recipes:
            grouped[recipe.result].append(recipe)
            all_items.add(recipe.result)
            for ingredient in recipe.ingredients:
                if ingredient.startswith("tag:"):
                    all_tags.add(ingredient.removeprefix("tag:"))
                else:
                    all_items.add(ingredient)
        tags = {
            tag: tuple(sorted(item for item in all_items if _matches_tag(item, tag)))
            for tag in all_tags
        }
        return cls({name: tuple(options) for name, options in grouped.items()}, tags)

    @classmethod
    def from_directory(cls, path: str | Path) -> RecipeBook:
        recipes: list[Recipe] = []
        for recipe_file in sorted(Path(path).glob("*.json")):
            try:
                recipes.append(
                    Recipe.from_minecraft(
                        json.loads(recipe_file.read_text(encoding="utf-8")),
                        recipe_file.name,
                    )
                )
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                # Platoon's folder includes non-crafting recipes; TextCraft
                # intentionally ignores documents it cannot execute.
                continue
        if not recipes:
            raise ValueError(f"no crafting recipes found in {path}")
        return cls.from_recipes(recipes)

    def options(self, item: str) -> tuple[Recipe, ...]:
        return self.recipes.get(item_name(item), ())

    def describe(self, item: str) -> list[dict[str, Any]]:
        return [
            {
                "result": recipe.result,
                "result_count": recipe.result_count,
                "ingredients": dict(recipe.ingredients),
            }
            for recipe in self.options(item)
        ]


def _matches_tag(item: str, tag: str) -> bool:
    tag = item_name(tag)
    if tag == "planks":
        return item.endswith("_planks")
    if tag == "logs":
        return item.endswith(("_log", "_stem")) or item in {"log", "stem"}
    if tag.endswith("_logs"):
        wood = tag.removesuffix("_logs")
        return item.startswith((wood + "_log", wood + "_stem"))
    if tag.endswith("_stems"):
        wood = tag.removesuffix("_stems")
        return item.startswith((wood + "_stem", wood + "_log"))
    if tag == "wool":
        return item.endswith("_wool")
    if tag == "coals":
        return item in {"coal", "charcoal"}
    return False


__all__ = ["Recipe", "RecipeBook", "item_name", "quantities"]
