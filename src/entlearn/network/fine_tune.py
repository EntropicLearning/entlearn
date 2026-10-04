"""Shape-preserving objective replacement on copied fitted parameters."""

import math
from dataclasses import fields, replace

import torch

from entlearn.network.blocks.hidden import _HiddenBlock
from entlearn.network.blocks.regression import _RegressionBlock
from entlearn.network.build import _build_recipe
from entlearn.network.data import _RegressionSupervision, _StagedData
from entlearn.network.session import _CompiledGraph
from entlearn.network.state import _FittedNetwork
from entlearn.recipe import (
    ClassificationHead,
    Connection,
    Hidden,
    Input,
    ManifoldInput,
    Recipe,
    RegressionHead,
)

_OBJECTIVE_FIELDS: dict[type, set[str]] = {
    Input: {"epsilon", "epsilon_D", "epsilon_T", "delta_cat"},
    ManifoldInput: {"epsilon", "epsilon_T", "alpha"},
    Hidden: {"epsilon"},
    ClassificationHead: set(),
    RegressionHead: {"epsilon_M", "W_M"},
    Connection: {"delta", "theta_alpha"},
}


def validate_replacement(fitted: _FittedNetwork, recipe: Recipe) -> None:
    """Reject every replacement except the explicit shape-preserving objective fields."""
    old = fitted.graph.recipe
    if len(old.blocks) != len(recipe.blocks) or len(old.connections) != len(recipe.connections):
        raise ValueError("fine_tune cannot change topology")
    for before, after in zip(
        (*old.blocks, *old.connections), (*recipe.blocks, *recipe.connections), strict=True
    ):
        if type(before) is not type(after):
            raise ValueError("fine_tune cannot change block or connection kinds")
        allowed = _OBJECTIVE_FIELDS[type(before)]
        for field in fields(before):
            if field.name not in allowed and getattr(before, field.name) != getattr(
                after, field.name
            ):
                raise ValueError(
                    f"fine_tune cannot change {before.name!r} parameter {field.name!r}"
                )
    head = _build_recipe(recipe).head
    if (
        isinstance(head, RegressionHead)
        and head.W_M is not None
        and len(head.W_M) != fitted.schema.M
    ):
        raise ValueError("fixed W_M must match the fitted number of regression targets")


def replace_objective(graph: _CompiledGraph, recipe: Recipe, data: _StagedData) -> None:
    """Attach validated descriptions to an owned graph without resetting fitted weights."""
    built = _build_recipe(recipe)
    old_head = graph.head.description
    graph.recipe = recipe
    graph.incoming, graph.outgoing = built.incoming, built.outgoing
    connections = {connection.name: connection for connection in built.connections}
    graph.connections = tuple(connections[connection.name] for connection in graph.connections)
    for description in built.topological_blocks:
        block = graph.blocks[description.name]
        block = replace(block, description=description)
        graph.blocks[description.name] = block
        if isinstance(block, _HiddenBlock):
            block.incoming = {
                name: replace(state, description=connections[name])
                for name, state in block.incoming.items()
            }
    head = graph.head
    if isinstance(head, _RegressionBlock):
        assert isinstance(old_head, RegressionHead)
        supervision = data.supervision
        assert isinstance(supervision, _RegressionSupervision)
        if head.description.W_M is not None and head.description.W_M != old_head.W_M:
            assert supervision.output_weights is not None
            head.W_M = supervision.output_weights.detach().clone()
        elif head.W_M is None and math.isfinite(head.description.epsilon_M):
            head.W_M = torch.full(
                (data.schema.M,),
                1 / data.schema.M,
                dtype=data.X_cont.dtype,
                device=data.X_cont.device,
            )
