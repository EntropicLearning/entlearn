"""Owned copying of active graph parameters and optional row coordinates.

Each block copies itself through ``copy_active``; the helpers here are the conversions
those copies share.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import torch

from entlearn.primitives.normalise import floored_log_

if TYPE_CHECKING:
    from entlearn.network.session import _CompiledGraph


def copy_tensor(value: torch.Tensor, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Return an owned copy of ``value`` in ``dtype`` on ``device``, rejecting narrowing overflow."""
    result = value.to(dtype=dtype, device=device, copy=True).detach()
    narrowed = dtype.itemsize < value.dtype.itemsize and bool(torch.isinf(result).any())
    if narrowed and int(torch.isinf(result).sum()) > int(torch.isinf(value).sum()):
        raise ValueError(f"fitted state must stay finite in {dtype}")
    return result


def copy_log(log_value: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Copy a cached log beside its already copied ``value``.

    A changed dtype recomputes the log from ``value`` rather than rounding the old one.
    """
    result = copy_tensor(log_value, value.dtype, value.device)
    if log_value.dtype != value.dtype:
        floored_log_(result, value)
    return result


def copy_active_graph(
    graph: _CompiledGraph,
    *,
    dtype: torch.dtype,
    device: torch.device,
    row_weights: torch.Tensor | None,
) -> _CompiledGraph:
    """Copy active parameters, retaining coordinates or allocating for new rows."""
    return replace(
        graph,
        blocks={
            name: block.copy_active(dtype, device, row_weights)
            for name, block in graph.blocks.items()
        },
        incoming=dict(graph.incoming),
        outgoing=dict(graph.outgoing),
        training_rows=graph.training_rows if row_weights is None else len(row_weights),
    )
