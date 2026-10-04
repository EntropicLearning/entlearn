"""Detached named snapshots of selected fitted parameter groups."""

from collections.abc import Sequence

import torch

from entlearn.network.blocks.hidden import _HiddenBlock
from entlearn.network.blocks.types import _OutputBlock
from entlearn.network.initialisation.geometry import capture_input_geometry
from entlearn.network.state import (
    InitialState,
    _DownstreamParameters,
    _FittedNetwork,
)


@torch.inference_mode()
def capture_current_state(fitted: _FittedNetwork, blocks: Sequence[str] | None) -> InitialState:
    """Copy selected complete groups in graph order, and not retained members or rows."""
    graph = fitted.graph
    if blocks is not None and (
        not isinstance(blocks, Sequence)
        or isinstance(blocks, str)
        or any(not isinstance(name, str) for name in blocks)
    ):
        raise ValueError("blocks must be a sequence of block names")
    names = graph.order if blocks is None else tuple(blocks)
    if len(names) != len(set(names)):
        raise ValueError("duplicate block names in capture")
    unknown = set(names) - set(graph.order)
    if unknown:
        raise ValueError(f"unknown capture block names: {sorted(unknown)}")
    parameters: list[_DownstreamParameters] = []
    for name in graph.order:
        if name not in names:
            continue
        block = graph.blocks[name]
        if isinstance(block, _HiddenBlock | _OutputBlock):
            parameters.append(block.capture_parameters())
    return InitialState(
        input_geometry=capture_input_geometry(graph) if graph.order[0] in names else None,
        parameters=tuple(parameters),
        connection_sub_seeds=graph.connection_sub_seeds,
    )
