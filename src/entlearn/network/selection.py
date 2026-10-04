"""Network-owned scoring and retention of original states and fitted members."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from functools import partial
from numbers import Real
from typing import Literal, Protocol

import torch

from entlearn._warnings import _warn
from entlearn.network.config import PredictConfig
from entlearn.network.data import _StagedData
from entlearn.network.dispatch import _map_seeds
from entlearn.network.fit import (
    ConvergenceWarning,
    _report_loss,
    _report_summary,
)
from entlearn.network.folds import _Fold
from entlearn.network.persistence.snapshot import decode_snapshot, encode_snapshot
from entlearn.network.predict import predict
from entlearn.network.state import (
    InitialState,
    InitOutcome,
    _FittedNetwork,
    _NetworkRecord,
    _SelectionHistory,
    _Trajectory,
    _winning_outcome,
)


class _SelectionLoss(Protocol):
    """A tensor callback shared by in-sample and explicit-fold selection."""

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        sample_weights: torch.Tensor | None,
        class_weights: torch.Tensor | None,
        task_weights: torch.Tensor | None,
        fold: tuple[torch.Tensor, torch.Tensor] | None,
        partition: Literal["training", "validation"],
    ) -> float | torch.Tensor: ...


class _FitCandidate(Protocol):
    """Fit and calibrate selected rows."""

    def __call__(
        self,
        seed: int,
        *,
        initial_state: InitialState | None,
        predict_config: PredictConfig,
        logger: logging.Logger | None,
        verbose: int,
        rows: torch.Tensor | None = None,
    ) -> _FittedNetwork: ...


@dataclass(frozen=True)
class _CandidateEvaluation:
    """An evaluated original geometry and the full-data fit, when already available."""

    initial_state: InitialState
    trajectories: tuple[_Trajectory, ...]
    fitted: _FittedNetwork | None
    score: float
    train_score: float | None = None

    def __reduce__(self) -> tuple[Callable[..., _CandidateEvaluation], tuple[object, ...]]:
        """Transport fitted candidates through the canonical model schema."""
        if self.fitted is None:
            return type(self), (
                self.initial_state,
                self.trajectories,
                None,
                self.score,
                self.train_score,
            )
        # Selection has not finished, so the candidate has no selection history yet.
        metadata, tensors = encode_snapshot(_NetworkRecord(self.fitted, _SelectionHistory(())))
        return _restore_candidate, (
            metadata,
            tensors,
            str(self.fitted.graph.input.continuous_centroids.device),
            self.score,
        )


def _restore_candidate(
    metadata: str,
    tensors: dict[str, torch.Tensor],
    device: str,
    score: float,
) -> _CandidateEvaluation:
    """Reconstruct one worker result."""
    fitted = decode_snapshot(metadata, tensors, device=device).fitted
    return _CandidateEvaluation(fitted.initial_state, (fitted.trajectory,), fitted, score)


@torch.inference_mode()
def _predict_and_score(
    candidate: _FittedNetwork,
    data: _StagedData,
    selection_loss: _SelectionLoss,
    *,
    rows: torch.Tensor | None = None,
    fold: _Fold | None = None,
    partition: Literal["training", "validation"] = "training",
) -> float:
    """Predict requested rows and score labelled responses with raw weights."""
    prediction = predict(
        candidate.graph,
        data.X_cont if rows is None else data.X_cont[rows],
        X_cat_new=data.X_cat if rows is None else tuple(feature[rows] for feature in data.X_cat),
        config=candidate.predict_config,
    ).prediction
    labelled = data.labelled if rows is None else data.labelled[rows]
    selected = labelled if rows is None else rows[labelled]
    # Callback inputs must not expose shared mutable storage. Boolean indexing allocates
    # new storage for predictions, targets and row weights. Class weights need a clone.
    loss = selection_loss(
        prediction[labelled],
        data.target[selected],
        sample_weights=(
            None if data.raw_sample_weights is None else data.raw_sample_weights[selected]
        ),
        class_weights=None if data.class_weights is None else data.class_weights.clone(),
        task_weights=None if data.task_weights is None else data.task_weights[selected],
        fold=None if fold is None else (fold[0].clone(), fold[1].clone()),
        partition=partition,
    )
    if isinstance(loss, torch.Tensor):
        if loss.ndim != 0 or loss.is_complex() or loss.dtype == torch.bool:
            raise ValueError("selection_loss must return one finite real scalar")
        loss = float(loss)
    if isinstance(loss, bool) or not isinstance(loss, Real) or not math.isfinite(loss):
        raise ValueError("selection_loss must return one finite real scalar")
    return float(loss)


@torch.inference_mode()
def _evaluate_candidate(
    seed: int,
    *,
    fit_one: _FitCandidate,
    data: _StagedData,
    initial_state: InitialState | None,
    folds: Sequence[_Fold] | None,
    selection_loss: _SelectionLoss | None,
    predict_config: PredictConfig,
    return_train_score: bool,
    logger: logging.Logger | None,
    verbose: int,
) -> _CandidateEvaluation:
    """Evaluate a initialisation in-sample or across independent training folds."""
    if folds is None:
        candidate = fit_one(
            seed,
            initial_state=initial_state,
            predict_config=predict_config,
            logger=logger,
            verbose=verbose,
        )
        score = (
            candidate.trajectory.loss_history[-1]
            if selection_loss is None
            else _predict_and_score(candidate, data, selection_loss)
        )
        return _CandidateEvaluation(
            candidate.initial_state, (candidate.trajectory,), candidate, score
        )

    assert selection_loss is not None
    scores, train_scores, trajectories = [], [], []
    fold_config = replace(predict_config, predict_mode="single")
    state = initial_state
    for fold in folds:
        train, validation = fold
        candidate = fit_one(
            seed,
            initial_state=state,
            predict_config=fold_config,
            logger=logger,
            verbose=verbose,
            rows=train,
        )
        # The first fit validates, copies and converts original geometry as authorised.
        # Reuse that state thereafter.
        if not trajectories:
            state = candidate.initial_state
        trajectories.append(candidate.trajectory)
        if return_train_score:
            train_scores.append(
                _predict_and_score(
                    candidate, data, selection_loss, rows=train, fold=fold, partition="training"
                )
            )
        scores.append(
            _predict_and_score(
                candidate, data, selection_loss, rows=validation, fold=fold, partition="validation"
            )
        )
        del candidate
    assert state is not None
    return _CandidateEvaluation(
        state,
        tuple(trajectories),
        None,
        math.fsum(score / len(folds) for score in scores),
        math.fsum(score / len(folds) for score in train_scores) if return_train_score else None,
    )


def _report_trajectory(
    trajectory: _Trajectory,
    identity: str,
    *,
    parallel: bool,
    logger: logging.Logger,
    verbose: int,
    non_converged: list[str],
) -> None:
    """Log a parallel fit's trajectory afterwards, and record the fit if it did not converge."""
    if parallel:
        for iteration, loss in enumerate(trajectory.loss_history):
            _report_loss(logger, verbose, iteration, loss)
        _report_summary(logger, verbose, trajectory)
    if not trajectory.converged:
        non_converged.append(f"{identity} within max_iter={trajectory.n_iter}")


def _finalise(
    record: InitOutcome,
    evaluation: _CandidateEvaluation,
    *,
    fit_one: _FitCandidate,
    predict_config: PredictConfig,
    logger: logging.Logger | None,
    verbose: int,
    report: Callable[[_Trajectory, str], None],
) -> _FittedNetwork:
    """Return the candidate's full-data fit, refitting it first when only folds were fitted."""
    fitted = evaluation.fitted
    if fitted is None:
        fitted = fit_one(
            record.seed,
            initial_state=evaluation.initial_state,
            predict_config=predict_config,
            logger=logger,
            verbose=verbose,
        )
        report(fitted.trajectory, f"init {record.index} (seed={record.seed}), full-data refit")
    return replace(fitted, initial_state=evaluation.initial_state)


@torch.inference_mode()
def _select_candidate(
    fit_one: _FitCandidate,
    seeds: tuple[int, ...],
    *,
    data: _StagedData,
    initial_state: InitialState | None,
    folds: Sequence[_Fold] | None,
    selection_loss: _SelectionLoss | None,
    predict_config: PredictConfig,
    return_train_score: bool,
    n_jobs: int | None,
    parallel_backend: Literal["threads", "processes"],
    retain: Literal["winner", "states", "members"],
    logger: logging.Logger,
    verbose: int,
) -> _NetworkRecord:
    """Select and finalise original geometries with shared dispatch, reporting and retention."""
    parallel = len(seeds) > 1 and n_jobs not in (None, 1)
    # Parallel fits log nothing live; report() logs each one afterwards, in seed order.
    fit_logger, fit_verbose = (None, 0) if parallel else (logger, verbose)
    records: list[InitOutcome] = []
    non_converged: list[str] = []
    states: list[InitialState] | None = None if retain == "winner" else []
    retained: list[_CandidateEvaluation] | None = [] if retain == "members" else None
    report = partial(
        _report_trajectory,
        parallel=parallel,
        logger=logger,
        verbose=verbose,
        non_converged=non_converged,
    )

    backend, evaluations = _map_seeds(
        partial(
            _evaluate_candidate,
            fit_one=fit_one,
            data=data,
            initial_state=initial_state,
            folds=folds,
            selection_loss=selection_loss,
            predict_config=predict_config,
            return_train_score=return_train_score,
            logger=fit_logger,
            verbose=fit_verbose,
        ),
        seeds,
        n_jobs=n_jobs,
        backend=parallel_backend if parallel else "serial",
    )
    best: InitOutcome | None = None
    best_evaluation: _CandidateEvaluation | None = None
    try:
        for index, (seed, evaluation) in enumerate(zip(seeds, evaluations, strict=True)):
            record = InitOutcome(
                index=index,
                seed=seed,
                score=evaluation.score,
                train_score=evaluation.train_score,
                requested_backend=parallel_backend,
                effective_backend=backend,
            )
            records.append(record)
            for ordinal, trajectory in enumerate(evaluation.trajectories):
                identity = f"init {index} (seed={seed})"
                if folds is not None:
                    identity += f", fold {ordinal}"
                report(trajectory, identity)
            if states is not None:
                states.append(evaluation.initial_state)
            if retained is not None:
                retained.append(evaluation)
            # Keep only the best evaluation unless members are retained.
            if best is None or _winning_outcome((best, record)) is record:
                best, best_evaluation = record, evaluation
    finally:
        # Stopping early still releases the runner's held warnings, on this line.
        evaluations.close()
    assert best is not None and best_evaluation is not None
    finalise = partial(
        _finalise,
        fit_one=fit_one,
        predict_config=predict_config,
        logger=fit_logger,
        verbose=fit_verbose,
        report=report,
    )
    members = None if retained is None else tuple(map(finalise, records, retained))
    winner = finalise(best, best_evaluation) if members is None else members[best.index]
    return _NetworkRecord(
        winner,
        _selection_history(
            tuple(records),
            non_converged,
            best_index=best.index,
            warnings_at=len(winner.trajectory.warnings),
            logger=logger,
            verbose=verbose,
        ),
        None if states is None else tuple(states),
        members,
    )


def _selection_history(
    records: tuple[InitOutcome, ...],
    non_converged: list[str],
    *,
    best_index: int,
    warnings_at: int,
    logger: logging.Logger,
    verbose: int,
) -> _SelectionHistory:
    """Report ordered selection and aggregate non-convergence before publication."""
    if verbose >= 1 and len(records) > 1:
        logger.info(
            "multi-init: selected init %d (score=%.6g)", best_index, records[best_index].score
        )
        logger.info("multi-init: per-init scores = %s", [record.score for record in records])
    if not non_converged:
        return _SelectionHistory(records, best_index)
    message = (
        "the solver did not converge for "
        + "; ".join(non_converged)
        + "; increase max_iter or loosen tol"
    )
    # Report once, in the parent, after every candidate has returned but before
    # any public Network exists. Warning-as-error therefore aborts publication.
    _warn(message, ConvergenceWarning)
    return _SelectionHistory(records, best_index, (message,), warnings_at)
