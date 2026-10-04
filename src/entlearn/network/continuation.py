"""Resume and fine-tuning of retained trajectories without repeating selection."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import replace

import torch

from entlearn._warnings import _warn
from entlearn.network.blocks.hidden import _HiddenBlock
from entlearn.network.blocks.input_common import update_affiliations_
from entlearn.network.blocks.types import _ClusteringBlock
from entlearn.network.build import _build_recipe
from entlearn.network.data import _stage_data, _StagedData
from entlearn.network.finalise import finalise
from entlearn.network.fine_tune import replace_objective, validate_replacement
from entlearn.network.fit import ConvergenceWarning, _optimise
from entlearn.network.session import _CompiledGraph, _FitSession
from entlearn.network.state import (
    InitialState,
    _copy_parameter_group,
    _FittedNetwork,
    _input_width,
    _NetworkRecord,
)
from entlearn.network.state_validation import resume_error, validate_parameter_state
from entlearn.network.transfer import copy_active_graph
from entlearn.recipe import Recipe


def _copy_initial_state(
    state: InitialState, dtype: torch.dtype, *, device: torch.device | None = None
) -> InitialState:
    """Copy original geometry in the computation dtype, independently of fitted state."""
    return replace(
        state,
        input_geometry=None
        if state.input_geometry is None
        else _copy_parameter_group(state.input_geometry, dtype, device),
        parameters=tuple(_copy_parameter_group(group, dtype, device) for group in state.parameters),
    )


def _continuation_session(
    graph: _CompiledGraph, data: _StagedData, initial_state: InitialState
) -> _FitSession:
    """Open a fit session over the copied graph, retaining every fitted coordinate."""
    geometry = initial_state.input_geometry
    assert geometry is not None
    # Original capacities preserve the strided arithmetic used after pruning.
    # A shared original state may be wider than the prefix used by this fit.
    capacities = {
        name: (
            _input_width(geometry, graph.input.description)
            if name == graph.order[0]
            else block.description.K
        )
        for name, block in graph.blocks.items()
        if isinstance(block, _ClusteringBlock)
    }
    return _FitSession.open(graph, data, capacities=capacities)


def _continue_one(
    fitted: _FittedNetwork,
    data: _StagedData,
    state: InitialState,
    *,
    max_iter: int,
    tol: float,
    logger: logging.Logger,
    verbose: int,
    replacement: Recipe | None,
) -> _FittedNetwork:
    """Continue one copied active graph and complete its prediction policy."""
    rebuild_rows = replacement is not None
    graph = copy_active_graph(
        fitted.graph,
        dtype=data.X_cont.dtype,
        device=data.X_cont.device,
        row_weights=data.sample_weights if rebuild_rows else None,
    )
    if replacement is not None:
        replace_objective(graph, replacement, data)
    graph.resolve_affiliation_regimes()
    session = _continuation_session(graph, data, state)
    if rebuild_rows:
        update_affiliations_(graph.input, session)
        for name in graph.order[1:-1]:
            block = graph.blocks[name]
            assert isinstance(block, _HiddenBlock)
            block.initialise_(session)
    trajectory = _optimise(
        session,
        max_iter=max_iter,
        tol=tol,
        logger=logger,
        verbose=verbose,
        previous=None if rebuild_rows else fitted,
    )
    if not trajectory.converged:
        message = (
            f"the solver has not converged after {trajectory.n_iter} iterations "
            f"(max_iter={max_iter}); increase max_iter or loosen tol"
        )
        _warn(message, ConvergenceWarning)
        trajectory = replace(trajectory, warnings=(*trajectory.warnings, message))
    policy = fitted.predict_config
    if fitted.epsilon_P_source == "derived":
        policy = replace(policy, epsilon_P=None)
    return finalise(
        graph,
        data,
        trajectory,
        policy,
        state,
        scoring_reference=True,
        max_iter=max_iter,
        tol=tol,
    )


@torch.inference_mode()
def continue_fit(
    record: _NetworkRecord,
    X_cont: torch.Tensor,
    y: torch.Tensor,
    *,
    X_cat: Sequence[torch.Tensor] | None,
    sample_weights: torch.Tensor | None,
    class_weights: torch.Tensor | None,
    task_weights: torch.Tensor | Callable[[torch.Tensor], torch.Tensor] | None,
    computation_dtype: torch.dtype | None,
    max_iter: int,
    tol: float,
    logger: logging.Logger,
    verbose: int,
    replacement: Recipe | None = None,
) -> _NetworkRecord:
    """Validate once, continue retained trajectories, and preserve the selection history.

    Fine-tuning resets the selection's warnings.
    """
    fitted, states, members = record.fitted, record.states, record.members
    rebuild_rows = replacement is not None
    for candidate in (fitted, *(members or ())):
        if replacement is not None:
            validate_parameter_state(candidate)
            validate_replacement(candidate, replacement)
        else:
            error = resume_error(candidate)
            if error is not None:
                raise ValueError(error)
    built = _build_recipe(fitted.graph.recipe if replacement is None else replacement)
    if (
        not rebuild_rows
        and isinstance(X_cont, torch.Tensor)
        and X_cont.ndim == 2
        and X_cont.shape[0] != fitted.graph.training_rows
    ):
        raise ValueError("resume requires the fitted training row count")
    data = _stage_data(
        built,
        X_cont,
        y,
        X_cat=X_cat,
        sample_weights=sample_weights,
        class_weights=class_weights,
        task_weights=task_weights,
        computation_dtype=computation_dtype,
        fitted_schema=fitted.schema,
    )
    winner = record.winner
    original_states = (fitted.initial_state,) if states is None else states
    copied_states = tuple(
        _copy_initial_state(state, data.X_cont.dtype, device=data.X_cont.device)
        for state in original_states
    )
    continued = tuple(
        _continue_one(
            candidate,
            data,
            state,
            max_iter=max_iter,
            tol=tol,
            logger=logger,
            verbose=verbose,
            replacement=replacement,
        )
        for candidate, state in zip(
            (fitted,) if members is None else members,
            (copied_states[winner],) if members is None else copied_states,
            strict=True,
        )
    )
    return _NetworkRecord(
        continued[0 if members is None else winner],
        replace(record.selection, warnings=(), warnings_at=0) if rebuild_rows else record.selection,
        None if states is None else copied_states,
        None if members is None else continued,
    )
