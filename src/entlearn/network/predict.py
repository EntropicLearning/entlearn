"""One private prediction engine for primary values and selected query details."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from entlearn.network.blocks.classification import _ClassificationBlock
from entlearn.network.blocks.hidden import _HiddenBlock
from entlearn.network.blocks.input_common import (
    has_instance_weight_recovery,
    recover_instance_weights_,
)
from entlearn.network.config import PredictConfig
from entlearn.network.predict_init import seed_prediction_, validate_predict_init
from entlearn.network.reconstruct import reconstruct_from_affiliations
from entlearn.network.refine import refine_prediction_
from entlearn.network.session import _CompiledGraph, _Workspace
from entlearn.network.state import PredictionResult
from entlearn.primitives.reductions import compute_wt_cost_
from entlearn.primitives.softmax import assign_simplex_


@dataclass
class _PredictionInputCache:
    """Local cache of input affiliations, modality costs and shared scratch."""

    X: torch.Tensor
    gamma: torch.Tensor
    total: torch.Tensor
    sqdist: torch.Tensor | None
    categorical_cost: torch.Tensor | None
    instance_cost: torch.Tensor
    workspace: _Workspace

    def refresh_instance_cost_(self) -> None:
        """Reduce both input modalities at the current affiliations."""
        compute_wt_cost_(
            self.instance_cost,
            self.gamma,
            self.sqdist,
            self.categorical_cost,
            self.workspace.matrix[: self.X.shape[0], : self.gamma.shape[1]],
        )


def _input_affiliations(
    graph: _CompiledGraph,
    X_cont_new: object,
    *,
    X_cat_new: Sequence[torch.Tensor] | None,
) -> _PredictionInputCache:
    """Validate query rows and return their input affiliations, costs and workspace."""
    block = graph.input
    X, X_cat = block.validate_prediction_input(X_cont_new, X_cat_new)
    T = X.shape[0]
    max_clusters = max(graph.affiliations(name).shape[1] for name in graph.order[:-1])
    workspace = _Workspace.allocate(
        T, max_clusters, graph.head.output_width, dtype=X.dtype, device=X.device
    )
    cost, sqdist, categorical_cost = block.price_query_rows_(X, X_cat, workspace)
    gamma = torch.empty_like(cost)
    # The fit assembles a Wt-weighted cost and assigns at ``epsilon / T``. Prediction
    # assembles an unweighted cost and assigns at ``epsilon``. The test Wt is the uniform
    # ``1 / T_new``, so the two ``1 / T`` factors cancel and the exponent matches the fit's.
    # The fitted regime depends on ``epsilon`` and the dtype alone, and is not influenced by the batch size
    assign_simplex_(
        gamma,
        cost,
        block.effective_epsilon,
        1,
        workspace.row_keepdim,
        workspace.row_indices,
    )
    return _PredictionInputCache(
        X,
        gamma,
        cost,
        sqdist,
        categorical_cost,
        torch.empty(T, dtype=X.dtype, device=X.device),
        workspace,
    )


def _forward(
    graph: _CompiledGraph,
    X_cont_new: object,
    *,
    X_cat_new: Sequence[torch.Tensor] | None,
) -> tuple[_PredictionInputCache, dict[str, torch.Tensor]]:
    """Return all clustering affiliations and scratch from a read-only forward pass."""
    query = _input_affiliations(graph, X_cont_new, X_cat_new=X_cat_new)
    X, gamma, workspace = query.X, query.gamma, query.workspace
    T = X.shape[0]
    gammas = {graph.input.description.name: gamma}
    for name in graph.order[1:-1]:
        hidden = graph.blocks[name]
        assert isinstance(hidden, _HiddenBlock)
        (connection,) = graph.incoming[name]
        propagated = torch.empty(
            T,
            hidden.K,
            dtype=X.dtype,
            device=X.device,
        )
        hidden.incoming[connection.name].propagate_(
            propagated,
            gamma,
            hidden.effective_epsilon,
            workspace,
        )
        gamma = propagated
        gammas[name] = gamma
    return query, gammas


def _head_logits(
    graph: _CompiledGraph,
    X_cont: torch.Tensor,
    X_cat: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Return unscaled geometric logits using single-pass affiliations."""
    head = graph.head
    assert isinstance(head, _ClassificationBlock)
    _, gammas = _forward(graph, X_cont, X_cat_new=X_cat)
    gamma = gammas[graph.connection_into(head).source]
    logits = torch.empty(gamma.shape[0], head.output_width, dtype=gamma.dtype, device=gamma.device)
    head.logits_(logits, gamma)
    return logits


def predict(
    graph: _CompiledGraph,
    X_cont_new: object,
    *,
    X_cat_new: Sequence[torch.Tensor] | None,
    config: PredictConfig,
    predict_init: str | int | torch.Tensor | None = None,
    details: Sequence[str] = (),
) -> PredictionResult:
    """Run a prediction and copy the requested details."""
    supported = {"affiliations", "instance_weights", "diagnostics", "reconstruction"}
    if (
        not isinstance(details, Sequence)
        or isinstance(details, str)
        or any(not isinstance(name, str) or name not in supported for name in details)
    ):
        raise ValueError("unknown prediction detail")
    recovering = has_instance_weight_recovery(graph.input)
    if "instance_weights" in details and not recovering:
        raise ValueError(
            "recovering instance_weights requires learning them during training with finite "
            "epsilon_T and retaining a finite log_partition. Frozen or hard instance weights "
            "allow no instance-weight recovery"
        )
    validate_predict_init(graph, config, predict_init)
    query, gammas = _forward(graph, X_cont_new, X_cat_new=X_cat_new)
    workspace = query.workspace
    head = graph.head
    gamma = gammas[graph.connection_into(head).source]
    delta = graph.connection_into(head).delta
    output = torch.empty(gamma.shape[0], head.output_width, dtype=gamma.dtype, device=gamma.device)
    if isinstance(head, _ClassificationBlock):
        if config.output_mode == "arithmetic":
            head.arithmetic_readout_(output, gamma, workspace)
        else:
            epsilon_P = delta if config.epsilon_P is None else config.epsilon_P
            head.geometric_readout_(output, gamma, delta, epsilon_P, workspace)
    else:
        head.readout_(output, gamma, delta, workspace)
    weights = None
    if recovering and ("instance_weights" in details or config.predict_mode == "iterative"):
        query.refresh_instance_cost_()
        weights = torch.empty_like(query.instance_cost)
        recover_instance_weights_(graph.input, weights, query.instance_cost)
    history, converged = (), False
    if predict_init is not None:
        seed_prediction_(output, graph, gamma, predict_init, workspace)
    if config.predict_mode == "iterative":
        history, converged = refine_prediction_(
            graph,
            query,
            gammas,
            output,
            weights,
            config,
            seeded=predict_init is not None,
        )
        if isinstance(head, _ClassificationBlock) and config.output_mode == "arithmetic":
            # Refinement solves for a geometric latent prediction coordinate.
            # An arithmetic policy instead serves the linear read-out of the final
            # affiliations. This does not change the latent objective or its history.
            head.arithmetic_readout_(output, gamma, workspace)
    return PredictionResult(
        prediction=output.detach().clone(),
        affiliations={name: value.detach().clone() for name, value in gammas.items()}
        if "affiliations" in details
        else None,
        instance_weights=weights.detach().clone()
        if weights is not None and "instance_weights" in details
        else None,
        n_iter=len(history) if "diagnostics" in details else None,
        converged=converged if "diagnostics" in details else None,
        loss_history=history if "diagnostics" in details else None,
        reconstruction=reconstruct_from_affiliations(
            graph.input, query.X, gammas[graph.input.description.name]
        )
        if "reconstruction" in details
        else None,
    )
