"""One independently seeded fit and its process-local logging configuration."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import replace
from numbers import Integral

import torch

from entlearn.network.build import _BuiltRecipe
from entlearn.network.config import PredictConfig
from entlearn.network.data import _StagedData, _training_subset
from entlearn.network.finalise import complete, finalise
from entlearn.network.fit import _fit
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
    seeded: Mapping[int, InitialState],
    rows: torch.Tensor | None = None,
    provided: bool = False,
    training_prediction: bool = False,
) -> tuple[_FittedNetwork, torch.Tensor | None]:
    """Fit one independent candidate for selection, completed now if selection predicts with it.

    ``provided`` marks ``initial_state`` as the caller's own, fitted once without selection.
    Otherwise, ``seeded[seed]`` supplies the state generated before candidate fitting.
    """
    state = seeded[seed] if initial_state is None else initial_state
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
    fitted = finalise(
        graph,
        data,
        replace(trajectory, warnings=messages + trajectory.warnings),
        predict_config,
        state,
        max_iter=max_iter,
        tol=tol,
    )
    if rows is None and not training_prediction:
        # Selection completes an unscored full-data fit only if it keeps it.
        return fitted, None
    # Selection scores a fold fit and discards it, so it needs no scoring reference.
    return complete(
        fitted, data, scoring_reference=rows is None, training_prediction=training_prediction
    )
