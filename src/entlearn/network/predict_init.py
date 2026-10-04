"""Validation and assignment of call-local prediction starting coordinates."""

from __future__ import annotations

from numbers import Integral

import torch

from entlearn.network.blocks.classification import _ClassificationBlock
from entlearn.network.config import PredictConfig
from entlearn.network.initialisation.seeds import _validated_seed
from entlearn.network.session import _CompiledGraph, _Workspace
from entlearn.network.validation import _validate_distribution_rows
from entlearn.recipe import Coupling


def validate_predict_init(
    graph: _CompiledGraph,
    config: PredictConfig,
    start: str | int | torch.Tensor | None,
) -> None:
    """Validate the prediction initialisation for iterative prediction."""
    if start is None:
        return
    if config.predict_mode != "iterative":
        raise ValueError("predict_init requires iterative prediction")
    if isinstance(start, torch.Tensor):
        # Shape, dtype, device, finiteness and the classification simplex are
        # checked by seed_prediction_ against the staged prediction buffer.
        return
    if isinstance(start, Integral) and not isinstance(start, bool):
        _validated_seed(start)
        return
    head = graph.head
    if isinstance(head, _ClassificationBlock):
        allowed, coupling = ("arithmetic", "uniform"), head.description.coupling
    else:
        allowed, coupling = ("mean",), None
    if not isinstance(start, str) or start not in allowed:
        raise ValueError(f"predict_init must be a tensor, integer seed or one of {allowed}")
    if start == "arithmetic" and coupling is Coupling.S:
        raise ValueError("arithmetic predict_init requires an M classification head")


def seed_prediction_(
    prediction: torch.Tensor,
    graph: _CompiledGraph,
    gamma: torch.Tensor,
    start: str | int | torch.Tensor,
    workspace: _Workspace,
) -> None:
    """Set the initial prediction, leaving affiliations unchanged."""
    head = graph.head
    classification = isinstance(head, _ClassificationBlock)
    if isinstance(start, torch.Tensor):
        if (
            start.shape != prediction.shape
            or start.dtype != prediction.dtype
            or start.device != prediction.device
        ):
            raise ValueError(
                "predict_init must match prediction shape, computation dtype and device"
            )
        if not bool(torch.isfinite(start).all()):
            raise ValueError("predict_init must contain only finite values")
        if classification:
            _validate_distribution_rows(start, name="predict_init", allow_zero=False)
        prediction.copy_(start)
    elif isinstance(start, Integral):
        generator = torch.Generator(device=prediction.device).manual_seed(int(start))
        # Normalised exponentials sample the simplex uniformly.
        # Regression maps that simplex through its output centroids.
        mixture = prediction if classification else torch.empty_like(gamma)
        mixture.exponential_(generator=generator)
        mixture.div_(mixture.sum(dim=1, keepdim=True))
        if not classification:
            head.readout_(prediction, mixture, graph.terminal.delta, workspace)
    elif start == "uniform":
        prediction.fill_(1 / prediction.shape[1])
    elif not classification:
        prediction.copy_(head.C_y.mean(dim=1))
    else:
        head.arithmetic_readout_(prediction, gamma, workspace)
