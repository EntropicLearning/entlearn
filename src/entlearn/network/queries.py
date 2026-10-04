"""Closed fitted-value queries, dispatched directly to their tensor owners."""

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal, overload

import torch

from entlearn._warnings import _warn
from entlearn.network.blocks.classification import _ClassificationBlock
from entlearn.network.blocks.hidden import _HiddenBlock
from entlearn.network.blocks.input import _StandardInputBlock
from entlearn.network.blocks.input_common import has_instance_weight_recovery
from entlearn.network.blocks.manifold import _ManifoldInputBlock
from entlearn.network.blocks.types import _ClusteringBlock
from entlearn.network.config import PredictConfig
from entlearn.network.fit import loss_noise_threshold
from entlearn.network.predict import predict
from entlearn.network.session import _CompiledGraph
from entlearn.primitives.statistics import inlier_scores_

if TYPE_CHECKING:
    from entlearn.network.model import Network

# The inspection names whose value is one tensor per owner, then the closed set of all names.
type TensorInspection = Literal[
    "continuous_centroids",
    "feature_weights",
    "manifold_projectors",
    "transition_matrices",
    "training_affiliations",
    "training_instance_weights",
]
type InspectionName = (
    TensorInspection | Literal["affiliation_regimes", "categorical_centroids", "head_parameters"]
)


def _continuous_centroids(graph: _CompiledGraph) -> dict[str, torch.Tensor]:
    """Return the input block's continuous centroids."""
    block = graph.input
    return {block.description.name: block.continuous_centroids.detach().clone()}


def _categorical_centroids(graph: _CompiledGraph) -> dict[str, tuple[torch.Tensor, ...]] | None:
    """Return a standard input's categorical centroids, or ``None`` when it has none."""
    block = graph.input
    if isinstance(block, _StandardInputBlock) and block.categorical_centroids:
        return {
            block.description.name: tuple(c.detach().clone() for c in block.categorical_centroids)
        }
    return None


def _feature_weights(graph: _CompiledGraph) -> dict[str, torch.Tensor] | None:
    """Return a standard input's feature weights, or ``None`` for a manifold input."""
    block = graph.input
    if isinstance(block, _StandardInputBlock):
        return {block.description.name: block.feature_weights.detach().clone()}
    return None


def _manifold_projectors(graph: _CompiledGraph) -> dict[str, torch.Tensor] | None:
    """Return a manifold input's projectors, or ``None`` for a standard input."""
    block = graph.input
    if isinstance(block, _ManifoldInputBlock):
        return {block.description.name: block.manifold_projectors.detach().clone()}
    return None


def _transition_matrices(graph: _CompiledGraph) -> dict[str, torch.Tensor] | None:
    """Return every hidden block's incoming transition matrices, or ``None`` when there are none."""
    matrices = {
        connection: state.theta.detach().clone()
        for target in graph.blocks.values()
        if isinstance(target, _HiddenBlock)
        for connection, state in target.incoming.items()
    }
    return matrices or None


def _head_parameters(graph: _CompiledGraph) -> dict[str, dict[str, torch.Tensor]]:
    """Return the head's parameters: ``theta``, or ``C_y`` with any ``W_M``."""
    head = graph.head
    parameters = (
        {"theta": head.theta.detach().clone()}
        if isinstance(head, _ClassificationBlock)
        else {"C_y": head.C_y.detach().clone()}
    )
    if not isinstance(head, _ClassificationBlock) and head.W_M is not None:
        parameters["W_M"] = head.W_M.detach().clone()
    return {head.description.name: parameters}


def _training_affiliations(graph: _CompiledGraph) -> dict[str, torch.Tensor] | None:
    """Return the clustering blocks' training affiliations, or ``None`` when they are not kept."""
    affiliations = {name: graph.affiliations(name) for name in graph.order[:-1]}
    if all(gamma.shape[0] == graph.training_rows for gamma in affiliations.values()):
        return {name: gamma.detach().clone() for name, gamma in affiliations.items()}
    return None


def _training_instance_weights(graph: _CompiledGraph) -> dict[str, torch.Tensor] | None:
    """Return the input's training instance weights, or ``None`` when they are not kept."""
    block = graph.input
    if block.instance_weights.shape == (graph.training_rows,):
        return {block.description.name: block.instance_weights.detach().clone()}
    return None


def _affiliation_regimes(graph: _CompiledGraph) -> dict[str, Literal["hard", "soft"]]:
    """Return whether each clustering block assigns hard or soft affiliations."""
    return {
        target.description.name: "soft" if target.soft_assignments else "hard"
        for target in graph.blocks.values()
        if isinstance(target, _ClusteringBlock)
    }


_INSPECTIONS: dict[str, Callable[[_CompiledGraph], dict[str, Any] | None]] = {
    "continuous_centroids": _continuous_centroids,
    "categorical_centroids": _categorical_centroids,
    "feature_weights": _feature_weights,
    "manifold_projectors": _manifold_projectors,
    "transition_matrices": _transition_matrices,
    "head_parameters": _head_parameters,
    "training_affiliations": _training_affiliations,
    "training_instance_weights": _training_instance_weights,
    "affiliation_regimes": _affiliation_regimes,
}


@overload
def inspect(graph: _CompiledGraph, name: TensorInspection) -> dict[str, torch.Tensor]: ...
@overload
def inspect(
    graph: _CompiledGraph, name: Literal["categorical_centroids"]
) -> dict[str, tuple[torch.Tensor, ...]]: ...
@overload
def inspect(
    graph: _CompiledGraph, name: Literal["head_parameters"]
) -> dict[str, dict[str, torch.Tensor]]: ...
@overload
def inspect(
    graph: _CompiledGraph, name: Literal["affiliation_regimes"]
) -> dict[str, Literal["hard", "soft"]]: ...
def inspect(
    graph: _CompiledGraph, name: str
) -> (
    dict[str, torch.Tensor]
    | dict[str, tuple[torch.Tensor, ...]]
    | dict[str, dict[str, torch.Tensor]]
    | dict[str, Literal["hard", "soft"]]
):
    """Return detached fitted values under stable block or connection names.

    Raises:
        ValueError: If the name is unknown or unavailable for this fitted model.
    """
    if not isinstance(name, str) or name not in _INSPECTIONS:
        raise ValueError(f"unknown inspection name: {name!r}")
    values = _INSPECTIONS[name](graph)
    if values is None:
        raise ValueError(f"inspection {name!r} is unavailable for this fitted model")
    return values


def build_scoring_reference(
    graph: _CompiledGraph,
    X_cont: torch.Tensor,
    X_cat: Sequence[torch.Tensor],
    config: PredictConfig,
) -> torch.Tensor | None:
    """Get the reference instance weights on the training data, to be used by the scoring empirical CDF.

    The fitted prediction policy must already be resolved. The result contains
    recovered weights.

    Raises:
        RuntimeError: If the fitted recovery produces invalid reference values.
    """
    if not has_instance_weight_recovery(graph.input):
        return None
    reference = predict(
        graph, X_cont, X_cat_new=X_cat, config=config, details=("instance_weights",)
    ).instance_weights
    if reference is None or not bool(torch.isfinite(reference).all() & (reference >= 0).all()):
        raise RuntimeError("fitted recovery produced an invalid scoring reference")
    return reference


def _scores_available(network: "Network") -> bool:
    """Return whether ``network.score_samples`` can score queries."""
    return has_instance_weight_recovery(network._graph.input)


def score_samples(
    graph: _CompiledGraph,
    reference: torch.Tensor | None,
    X_cont: torch.Tensor,
    X_cat: Sequence[torch.Tensor] | None,
    config: PredictConfig,
) -> torch.Tensor:
    """Rank a query recovery against the retained fitted-config reference.

    Raises:
        ValueError: If the model has no instance-weight recovery or no scoring reference.
    """
    if not has_instance_weight_recovery(graph.input):
        raise ValueError("score_samples requires fitted instance-weight recovery")
    if reference is None or not reference.numel():
        raise ValueError("score_samples requires a scoring reference")
    weights = predict(
        graph, X_cont, X_cat_new=X_cat, config=config, details=("instance_weights",)
    ).instance_weights
    assert weights is not None
    spread = float(reference.max() - reference.min())
    band = loss_noise_threshold(float(reference.max()), reference.dtype)
    if spread <= band:
        _warn(
            "the scoring reference has collapsed to a single value "
            f"(spread {spread:.3g} within the noise band {band:.3g}); "
            "score_samples has a step-like reference rather than an informative ranking",
            UserWarning,
        )
    out = torch.empty_like(weights)
    inlier_scores_(out, reference, weights)
    return out
