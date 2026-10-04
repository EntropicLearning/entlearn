"""Scikit-learn scoring views over Network candidate predictions."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Literal, cast

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import get_scorer
from sklearn.model_selection import BaseCrossValidator, check_cv
from sklearn.utils import _safe_indexing
from sklearn.utils.validation import check_consistent_length

type _CV = int | BaseCrossValidator | Iterable[tuple[Sequence[int], Sequence[int]]] | None
type _InitRows = (
    Sequence[int] | Callable[[tuple[tuple[np.ndarray, np.ndarray], ...]], Sequence[int]] | None
)


class _FrozenClassifierPredictions(ClassifierMixin, BaseEstimator):
    """Serve cached classifier responses through the estimator protocol."""

    probabilities: np.ndarray
    classes_: np.ndarray

    def __init__(self, probabilities: np.ndarray, classes: np.ndarray) -> None:
        self.probabilities = probabilities
        self.classes_ = classes

    def predict(self, X: object) -> np.ndarray:
        """Return cached decoded labels."""
        del X
        return self.classes_[self.probabilities.argmax(axis=1)]

    def predict_proba(self, X: object) -> np.ndarray:
        """Return cached class probabilities."""
        del X
        return self.probabilities


class _FrozenRegressorPredictions(RegressorMixin, BaseEstimator):
    """Serve cached regression responses through the estimator protocol."""

    prediction: np.ndarray

    def __init__(self, prediction: np.ndarray) -> None:
        self.prediction = prediction

    def predict(self, X: object) -> np.ndarray:
        """Return cached real responses."""
        del X
        return self.prediction


def _scorer_accepts_sample_weight(scorer: Callable[..., float]) -> bool:
    """Return whether a scorer explicitly accepts sample weights."""
    try:
        parameters = inspect.signature(scorer).parameters
    except (TypeError, ValueError):
        return False
    return "sample_weight" in parameters


def _scorer_selection_requested(
    n_inits: int, cv: _CV, scoring: str | Callable[..., float] | None
) -> bool:
    """Return whether fit controls request scorer-based candidate selection."""
    return n_inits != 1 or cv is not None or scoring is not None


def _scoring_rows(
    labelled: np.ndarray,
    fold: tuple[torch.Tensor, torch.Tensor] | None,
    partition: Literal["training", "validation"],
) -> np.ndarray:
    """Return the original row indices represented by callback predictions."""
    if fold is None:
        rows = np.arange(labelled.shape[0])
    else:
        index = 0 if partition == "training" else 1
        rows = fold[index].detach().cpu().numpy()
    return rows[labelled[rows]]


def _score_cached_response(
    scorer: Callable[..., float],
    response: BaseEstimator,
    X: object,
    target: np.ndarray,
    sample_weights: torch.Tensor | None,
) -> float:
    """Invoke one scorer and adapt its higher-is-better result to a loss."""
    kwargs = {}
    if sample_weights is not None and _scorer_accepts_sample_weight(scorer):
        kwargs["sample_weight"] = sample_weights.detach().cpu().numpy()
    return -float(scorer(response, X, target, **kwargs))


def _regression_view(was_1d: bool, response: np.ndarray) -> _FrozenRegressorPredictions:
    """Serve a cached regression response with the fit target's dimensionality."""
    return _FrozenRegressorPredictions(response[:, 0] if was_1d else response)


@dataclass(frozen=True)
class _ScorerSelectionLoss:
    """Translate cached tensor predictions into a smaller-is-better scorer result.

    ``labelled`` marks the rows a scorer sees, and ``truth`` is read only on those
    rows. ``view`` wraps one cached response as the estimator the scorer calls.
    """

    scorer: Callable[..., float]
    X: object
    truth: np.ndarray
    labelled: np.ndarray
    view: Callable[[np.ndarray], BaseEstimator]

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
    ) -> float:
        del target, class_weights, task_weights
        rows = _scoring_rows(self.labelled, fold, partition)
        view = self.view(prediction.detach().cpu().numpy())
        return _score_cached_response(
            self.scorer, view, _safe_indexing(self.X, rows), self.truth[rows], sample_weights
        )


def _classification_selection_loss(
    codes: np.ndarray,
    classes: np.ndarray,
    X: object,
    scoring: str | Callable[..., float] | None,
    *,
    n_inits: int,
    cv: _CV,
) -> _ScorerSelectionLoss | None:
    """Build a classification scorer callback."""
    if not _scorer_selection_requested(n_inits, cv, scoring):
        return None
    return _ScorerSelectionLoss(
        get_scorer("accuracy" if scoring is None else scoring),
        X,
        classes[codes],
        codes != -1,
        partial(_FrozenClassifierPredictions, classes=classes),
    )


def _regression_selection_loss(
    target: np.ndarray,
    was_1d: bool,
    X: object,
    scoring: str | Callable[..., float] | None,
    *,
    n_inits: int,
    cv: _CV,
) -> _ScorerSelectionLoss | None:
    """Build a regression scorer callback."""
    if not _scorer_selection_requested(n_inits, cv, scoring):
        return None
    labelled = ~np.isnan(target) if was_1d else ~np.isnan(target).all(axis=1)
    return _ScorerSelectionLoss(
        get_scorer("r2" if scoring is None else scoring),
        X,
        target,
        labelled,
        partial(_regression_view, was_1d),
    )


def _materialise_splits(
    cv: _CV,
    X: object,
    target: np.ndarray,
    groups: object,
    *,
    classifier: bool,
    device: torch.device,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...] | None:
    """Resolve and snapshot a splitter partitioning as Network row-index tensors."""
    if cv is None:
        return None
    check_consistent_length(X, groups)
    splitter = check_cv(cv, y=target if classifier else None, classifier=classifier)
    splits = (
        splitter.split(X, target) if groups is None else splitter.split(X, target, groups=groups)
    )
    pairs = []
    for ordinal, (training, validation) in enumerate(splits):
        converted = []
        for name, rows in (("training", training), ("validation", validation)):
            array = np.asarray(rows)
            if array.ndim != 1 or array.dtype.kind not in "iu":
                raise ValueError(
                    f"cv fold {ordinal} {name} indices must be a one-dimensional integer array"
                )
            array = array.astype(np.int64, copy=True)
            converted.append(torch.as_tensor(array, dtype=torch.int64, device=device))
        pairs.append((converted[0], converted[1]))
    return tuple(pairs)


def _resolve_init_rows(
    init_rows: _InitRows,
    pairs: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None,
    *,
    device: torch.device,
) -> torch.Tensor | None:
    """Resolve explicit or fold-derived initialisation rows."""
    if init_rows is None:
        return None
    resolved: object = init_rows
    if callable(init_rows):
        if pairs is None:
            raise ValueError("a callable init_rows requires cv to define materialised folds")
        folds = tuple(
            (training.cpu().numpy().copy(), validation.cpu().numpy().copy())
            for training, validation in pairs
        )
        resolver = cast(
            "Callable[[tuple[tuple[np.ndarray, np.ndarray], ...]], Sequence[int]]", init_rows
        )
        resolved = resolver(folds)
    array = np.asarray(resolved)
    if array.ndim != 1 or array.dtype.kind not in "iu":
        raise ValueError("init_rows must resolve to a one-dimensional integer array")
    return torch.as_tensor(array.astype(np.int64, copy=True), dtype=torch.int64, device=device)


def common_train_rows(folds: Sequence[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Return the sorted intersection of every training partition.

    An ordinary KFold partition has an empty intersection; expanding-window time
    series folds often share an initial training core. The helper returns an empty
    array when there is no common row. Fitting rejects that selection before sampling.
    This only addresses geometry leakage, not external preprocessing.

    Args:
        folds: Non-empty sequence of training/validation integer-index pairs.

    Returns:
        One-dimensional int64 array of common training row indices.

    Raises:
        ValueError: If no folds are supplied or training indices are malformed.
    """
    if not folds:
        raise ValueError("common_train_rows needs at least one fold")
    core = None
    for training, _ in folds:
        indices = np.asarray(training)
        if indices.ndim != 1 or indices.dtype.kind not in "iu":
            raise ValueError("training indices must be one-dimensional integer arrays")
        core = np.unique(indices) if core is None else np.intersect1d(core, indices)
    return np.asarray(core, dtype=np.int64)
