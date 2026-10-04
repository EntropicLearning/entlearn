"""One independently seeded fit and its process-local logging configuration."""

from __future__ import annotations

import logging
from dataclasses import replace
from numbers import Integral

import torch

from entlearn.network.build import _BuiltRecipe
from entlearn.network.config import PredictConfig
from entlearn.network.data import _StagedData, _training_subset
from entlearn.network.finalise import finalise
from entlearn.network.fit import _fit
from entlearn.network.initialisation.geometry import _seed_initial_state
from entlearn.network.initialisation.resolve_state import resolve_state
from entlearn.network.state import InitialState, _FittedNetwork


def _fit_logger(logger: logging.Logger | None, verbose: int) -> logging.Logger:
    """Validate logging controls and resolve an instance-owned logger.

    A caller's logger is never reconfigured. The default is unregistered, so its
    handler and level cannot affect another Network or the application's loggers.

    Raises:
        ValueError: If the logger or verbosity is outside its domain.
    """
    if isinstance(verbose, bool) or not isinstance(verbose, Integral) or verbose < 0:
        raise ValueError("verbose must be a non-negative integer")
    if logger is not None:
        if not isinstance(logger, logging.Logger):
            raise ValueError("logger must be a logging.Logger or None")
        return logger
    resolved = logging.Logger("entlearn.network", level=logging.DEBUG)  # noqa: LOG001
    resolved.propagate = False
    if verbose:
        resolved.addHandler(logging.StreamHandler())
    return resolved


@torch.inference_mode()
def _fit_candidate(
    seed: int,
    *,
    built: _BuiltRecipe,
    data: _StagedData,
    initial_state: InitialState | None,
    allow_conversion: bool,
    predict_config: PredictConfig,
    max_iter: int,
    tol: float,
    logger: logging.Logger | None,
    verbose: int,
    init_rows: torch.Tensor | None = None,
    rows: torch.Tensor | None = None,
    provided: bool = False,
) -> _FittedNetwork:
    """Complete one independent fit and its prediction policy before selection.

    ``provided`` marks ``initial_state`` as the caller's own, fitted once without selection.
    """
    state = initial_state
    if state is None:
        state = _seed_initial_state(
            built, data, allow_conversion=allow_conversion, seed=seed, rows=init_rows
        )
    if rows is not None:
        data = _training_subset(built, data, rows)
    state, messages = resolve_state(
        built,
        data,
        state,
        seed=seed,
        allow_conversion=allow_conversion,
        provided=provided,
    )
    graph, trajectory = _fit(
        built,
        data,
        state,
        max_iter=max_iter,
        tol=tol,
        logger=_fit_logger(logger, verbose),
        verbose=verbose,
    )
    return finalise(
        graph,
        data,
        replace(trajectory, warnings=messages + trajectory.warnings),
        predict_config,
        state,
        # Selection scores a fold fit and discards it, so it needs no scoring reference.
        scoring_reference=rows is None,
        max_iter=max_iter,
        tol=tol,
    )
