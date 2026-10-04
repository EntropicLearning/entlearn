"""Translate estimator parameter paths into immutable Recipe replacements."""

from __future__ import annotations

from dataclasses import fields, replace
from typing import Any

from entlearn import Recipe


def _recipe_parameters(recipe: Recipe) -> dict[str, Any]:
    """Return every description field under its stable block or connection name."""
    return {
        f"recipe__{group}__{description.name}__{field.name}": getattr(description, field.name)
        for group in ("blocks", "connections")
        for description in getattr(recipe, group)
        for field in fields(description)
    }


def _replace_recipe_parameters(recipe: Recipe, changes: dict[str, Any]) -> Recipe:
    """Validate a complete batch of named replacements before returning a new Recipe."""
    available = _recipe_parameters(recipe)
    unknown = changes.keys() - available.keys()
    if unknown:
        raise ValueError(f"Invalid Recipe parameter(s): {sorted(unknown)}")
    groups = {}
    for group in ("blocks", "connections"):
        descriptions = []
        for description in getattr(recipe, group):
            prefix = f"recipe__{group}__{description.name}__"
            updates = {
                key.removeprefix(prefix): value
                for key, value in changes.items()
                if key.startswith(prefix)
            }
            descriptions.append(replace(description, **updates) if updates else description)
        groups[group] = tuple(descriptions)
    return replace(recipe, **groups)
