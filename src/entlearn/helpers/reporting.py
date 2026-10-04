"""State-free tensor reporting through the public fitted Network interface."""

from __future__ import annotations

import math
from numbers import Real
from typing import assert_never

import torch

from entlearn import (
    ClassificationHead,
    Coupling,
    Head,
    Hidden,
    Input,
    InputBlock,
    ManifoldInput,
    Network,
    RegressionHead,
)
from entlearn.primitives.statistics import effective_dimension

__all__ = [
    "active_features",
    "count_parameters",
    "effective_dimensions",
    "feature_importances",
    "target_weights",
]


def _require_network(model: Network) -> None:
    if not isinstance(model, Network):
        raise ValueError("reporting requires a fitted Network")


def _reporting_owners(model: Network) -> tuple[InputBlock, Head]:
    """Resolve the fitted chain's input and head independently of declaration order."""
    (first,) = (block for block in model.recipe.blocks if isinstance(block, InputBlock))
    (head,) = (block for block in model.recipe.blocks if isinstance(block, Head))
    return first, head


def _active_mask(model: Network, tol: float) -> torch.Tensor:
    """Select weights strictly above a multiple of the uniform feature share."""
    if isinstance(tol, bool) or not isinstance(tol, Real) or not math.isfinite(tol) or tol < 0:
        raise ValueError("tol must be finite and non-negative")
    weights = model.inspect("feature_weights")
    (wd,) = weights.values()
    if tol >= wd.numel():
        raise ValueError("tol must be below the feature count; tol/D must be below one")
    if torch.allclose(wd, torch.full_like(wd, 1.0 / wd.numel())):
        raise ValueError("active feature reporting requires non-uniform feature weights")
    return wd * wd.numel() > tol


@torch.inference_mode()
def active_features(model: Network, *, tol: float = 1.0) -> torch.Tensor:
    """Return sorted int64 indices of features whose weight exceeds ``tol / D``.

    Indices use tensor feature order and remain on the fitted device.
    This reports standard-input weights, not manifold tangent participation.
    A threshold is relative to
    the uniform share, not an absolute numerical floor. The default selects
    weights strictly above that uniform share.

    Raises:
        ValueError: If the model is not a fitted Network, has no feature weights,
            has uniform weights, or ``tol`` is outside ``[0, D)``.
    """
    _require_network(model)
    return _active_mask(model, tol).nonzero(as_tuple=True)[0]


@torch.inference_mode()
def count_parameters(
    model: Network,
    *,
    include_affiliations: bool = False,
    raw: bool = False,
    active_tol: float | None = None,
) -> int:
    """Count parameter entries or geometric degrees of freedom of a Network.

    Fixed feature, instance and output weights are excluded from the count. Training
    affiliations and instance weights (if learned) count only if ``include_affiliations=True``.

    By default (``raw=True``):
    - each vector simplex has D-1 free parameters.
    - A manifold basis represents a subspace, contributing ``d * (D - d)`` freedoms per
    active cluster instead of ``D * d``. With ``alpha=0``, the centroid also
    loses its ``d`` redundant tangent coordinates.

    ``active_tol`` retains only features whose weight exceeds ``active_tol / D``.
    Information needed for choosing the subset itself is not considered.

    Raises:
        ValueError: If the model or active threshold is invalid, or a requested
            fitted value is unavailable, including omitted row-bound state.
    """
    _require_network(model)
    mask = None if active_tol is None else _active_mask(model, active_tol)
    first, head = _reporting_owners(model)
    continuous = model.inspect("continuous_centroids")[first.name]
    total = (
        continuous.numel()
        if mask is None
        else (continuous.shape[0] * int(mask[: model.schema.D_cont].sum()))
    )
    match first:
        case Input():
            if model.schema.M_cat:
                categorical = model.inspect("categorical_centroids")[first.name]
                for index, centroid in enumerate(categorical):
                    if mask is None or bool(mask[model.schema.D_cont + index]):
                        total += centroid.numel() - (0 if raw else centroid.shape[0])
            if math.isfinite(first.epsilon_D):
                weights = model.inspect("feature_weights")[first.name]
                total += weights.numel() - (0 if raw else 1) if mask is None else int(mask.sum())
        case ManifoldInput():
            projectors = model.inspect("manifold_projectors")[first.name]
            K, D, d = projectors.shape
            total += projectors.numel() if raw else K * d * (D - d)
            if not raw and first.alpha == 0:
                total -= K * d
        case _:
            assert_never(first)
    if any(isinstance(block, Hidden) for block in model.recipe.blocks):
        transitions = model.inspect("transition_matrices")
        for connection in model.recipe.connections:
            if connection.name in transitions:
                theta = transitions[connection.name]
                total += theta.numel() - (
                    0 if raw else theta.shape[1 if connection.coupling is Coupling.M else 0]
                )
    parameters = model.inspect("head_parameters")[head.name]
    match head:
        case ClassificationHead():
            theta = parameters["theta"]
            total += theta.numel() - (
                0 if raw else theta.shape[1 if head.coupling is Coupling.M else 0]
            )
        case RegressionHead():
            total += parameters["C_y"].numel()
            if math.isfinite(head.epsilon_M):
                total += parameters["W_M"].numel() - (0 if raw else 1)
        case _:
            assert_never(head)
    if include_affiliations:
        affiliations = model.inspect("training_affiliations")
        for gamma in affiliations.values():
            total += gamma.numel() - (0 if raw else gamma.shape[0])
        if math.isfinite(first.epsilon_T):
            weights = model.inspect("training_instance_weights")[first.name]
            total += weights.numel() - (0 if raw else 1)
    return total


@torch.inference_mode()
def feature_importances(model: Network) -> torch.Tensor:
    """Return fitted feature weights or weighted manifold tangent participation.

    Standard input returns its learned or fixed feature simplex. Manifold input
    averages the diagonal tangent-projector leverage using training affiliations
    weighted by fitted training instance weights, then divides by subspace dimension.
    Those instance weights are learned, supplied, or uniform according to the fit.

    For orthonormal tangent bases Tₖ ∈ ℝᴰˣᵈ, training affiliations Γₜₖ and fitted
    instance weights wₜ, the weighted occupancy and feature participation are:

        ρₖ = (∑ₜ wₜ Γₜₖ) / (∑ₜ wₜ)
        ℓₖⱼ = (TₖTₖᵀ)ⱼⱼ = ∑ᵣ₌₁ᵈ (Tₖ)ⱼᵣ²
        importanceⱼ = (∑ₖ ρₖ ℓₖⱼ) / d

    Since ∑ₖ ρₖ = 1 and tr(TₖTₖᵀ) = d, ∑ⱼ importanceⱼ = 1.
    Rotating a tangent basis leaves its projector and participation unchanged.

    Manifold participation depends on feature coordinates and scaling; it is neither
    a learned feature-weight parameter nor a measure of predictive importance.
    It requires retained training affiliations and instance weights.
    Raises when row-bound values are unavailable.
    """
    _require_network(model)
    first, _ = _reporting_owners(model)
    match first:
        case Input():
            return model.inspect("feature_weights")[first.name]
        case ManifoldInput():
            basis = model.inspect("manifold_projectors")[first.name]
            gamma = model.inspect("training_affiliations")[first.name]
            weights = model.inspect("training_instance_weights")[first.name]
            occupancy = (weights / weights.sum()) @ gamma
            return occupancy @ basis.square().sum(dim=2) / basis.shape[2]
        case _:
            assert_never(first)


@torch.inference_mode()
def target_weights(model: Network) -> torch.Tensor:
    """Return detached regression target weights in fitted output order.

    Finite ``epsilon_M`` learns a relative preference for lower training residual
    costs. The costs average squared residuals
    against cluster output centroids under training affiliations and supervision
    weights. They depend on target scale, coupling strength and temperature.
    Infinite ``epsilon_M`` keeps supplied weights fixed, or uses an implicit
    uniform simplex when none were supplied.
    """
    _require_network(model)
    _, head = _reporting_owners(model)
    if not isinstance(head, RegressionHead):
        raise ValueError("target_weights requires a regression Network")
    parameters = model.inspect("head_parameters")[head.name]
    if "W_M" in parameters:
        return parameters["W_M"]
    return parameters["C_y"].new_full((model.schema.M,), 1.0 / model.schema.M)


@torch.inference_mode()
def effective_dimensions(
    model: Network, *, normalise: bool = True
) -> dict[str, dict[str, torch.Tensor]]:
    """Report realised spreads of fitted distributions.

    Each clustering block reports the mean row-wise ``affiliations`` spread.
    The input also reports ``instance_weights`` and, for standard input,
    ``feature_weights``. A regression head reports ``output_weights`` when
    explicitly represented, including fixed weights. ``normalise=True`` divides
    each effective dimension by the number of entries in its distribution.

    Raises:
        ValueError: If the model is not fitted or its training values are unavailable.
    """
    _require_network(model)
    first, _ = _reporting_owners(model)
    affiliations = model.inspect("training_affiliations")
    report = {
        name: {"affiliations": effective_dimension(gamma, normalise=normalise).mean()}
        for name, gamma in affiliations.items()
    }
    for request, field in (
        ("training_instance_weights", "instance_weights"),
        ("feature_weights", "feature_weights"),
    ):
        if field == "feature_weights" and not isinstance(first, Input):
            continue
        for name, weights in model.inspect(request).items():
            report[name][field] = effective_dimension(weights, normalise=normalise)
    for name, parameters in model.inspect("head_parameters").items():
        if "W_M" in parameters:
            report[name] = {
                "output_weights": effective_dimension(parameters["W_M"], normalise=normalise)
            }
    return report
