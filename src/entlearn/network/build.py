"""Private Recipe building."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

from entlearn.recipe import (
    Block,
    ClassificationHead,
    Connection,
    Input,
    ManifoldInput,
    Recipe,
    RegressionHead,
)


@dataclass(frozen=True)
class _BuiltRecipe:
    recipe: Recipe
    topological_blocks: tuple[Block, ...]
    connections: tuple[Connection, ...]
    incoming: dict[str, tuple[Connection, ...]]
    outgoing: dict[str, tuple[Connection, ...]]
    input: Input | ManifoldInput
    head: ClassificationHead | RegressionHead
    task: Literal["classification", "regression"]


def _build_recipe(recipe: Recipe) -> _BuiltRecipe:
    """Build one connected chain in stable topological order."""
    order = recipe.stable_order()
    edges = {(connection.source, connection.target) for connection in recipe.connections}
    if len(recipe.connections) != len(order) - 1 or edges != set(pairwise(order)):
        raise ValueError("the builder only supports one connected chain")
    by_name = {block.name: block for block in recipe.blocks}
    blocks = tuple(by_name[name] for name in order)
    incoming = {
        name: tuple(connection for connection in recipe.connections if connection.target == name)
        for name in order
    }
    outgoing = {
        name: tuple(connection for connection in recipe.connections if connection.source == name)
        for name in order
    }

    # Recipe validation makes a chain's unique source an input block and its unique sink a head.
    input_block, head = blocks[0], blocks[-1]
    assert isinstance(input_block, (Input, ManifoldInput))
    assert isinstance(head, (ClassificationHead, RegressionHead))
    task: Literal["classification", "regression"] = (
        "classification" if isinstance(head, ClassificationHead) else "regression"
    )
    return _BuiltRecipe(
        recipe=recipe,
        topological_blocks=blocks,
        connections=recipe.connections,
        incoming=incoming,
        outgoing=outgoing,
        input=input_block,
        head=head,
        task=task,
    )
