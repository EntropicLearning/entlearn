"""Optional statistical feature-weight initialiser.

The helper scores continuous and categorical features on labelled rows, then
normalises the scores into the standard input's continuous-then-categorical
feature order. The correlation method uses correlation ratio, Cramér's V or
absolute Pearson correlation according to the feature modality and task. The
mutual-information method delegates to scikit-learn's classification or
regression estimator. Statistical calculations use float64; the requested
tensor dtype is applied to the result. Degenerate scores fall back to uniform
weights.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import numpy as np
import torch


def _prepare_labelled_arrays(
    X_cont: object,
    categorical_codes: Sequence[object],
    target: object,
    task: Literal["classification", "regression"],
    labelled_mask: object | None,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """Return float64, row-aligned arrays restricted to labelled rows."""
    if task not in ("classification", "regression"):
        raise ValueError("task must be 'classification' or 'regression'")
    X_array = np.asarray(X_cont, dtype=np.float64)
    if X_array.ndim != 2:
        raise ValueError("X_cont must be a two-dimensional array")
    rows = X_array.shape[0]
    codes_arrays = [np.asarray(codes) for codes in categorical_codes]
    if any(codes.ndim != 1 or codes.shape[0] != rows for codes in codes_arrays):
        raise ValueError("every categorical code array must have one entry per X_cont row")
    target_array = np.asarray(target)
    if target_array.ndim not in (1, 2) or target_array.shape[0] != rows:
        raise ValueError("target must have one row per X_cont row")
    if task == "classification":
        if target_array.ndim != 1:
            raise ValueError("classification target must be one-dimensional class codes")
    else:
        target_array = target_array.astype(np.float64, copy=False)  # pragma: no mutate
        target_array = _as_2d_target(target_array)
        if target_array.shape[1] == 0:
            raise ValueError("regression target must have at least one output")
    if X_array.shape[1] + len(codes_arrays) == 0:
        raise ValueError("at least one continuous or categorical feature is required")
    if labelled_mask is None:
        mask = np.ones(rows, dtype=bool)
    else:
        mask = np.asarray(labelled_mask, dtype=bool)
        if mask.ndim != 1 or mask.shape[0] != rows:
            raise ValueError("labelled_mask must have one entry per X_cont row")
    return X_array[mask], [codes[mask] for codes in codes_arrays], target_array[mask]


def _correlation_ratio(values: np.ndarray, groups: np.ndarray) -> float:
    """Return the correlation ratio of continuous values across groups."""
    if values.shape[0] == 0:
        return 0.0
    grand_mean = values.mean()
    total = float(np.sum((values - grand_mean) ** 2))
    if not total > 0.0:
        return 0.0
    _, codes = np.unique(groups, return_inverse=True)
    counts = np.bincount(codes)
    sums = np.bincount(codes, weights=values)
    group_means = sums / counts
    between = float(np.sum(counts * (group_means - grand_mean) ** 2))
    ratio = between / total
    return math.sqrt(max(0.0, min(1.0, ratio))) if math.isfinite(ratio) else 0.0


def _cramers_v(first: np.ndarray, second: np.ndarray) -> float:
    """Return Cramér's V between two categorical variables."""
    from scipy.stats.contingency import association, crosstab

    if first.shape[0] == 0:
        return 0.0
    table = crosstab(first, second).count
    rows, columns = table.shape
    if rows < 2 or columns < 2:
        return 0.0
    value = float(association(table, method="cramer", correction=False))
    return value if math.isfinite(value) else 0.0


def _absolute_pearson(first: np.ndarray, second: np.ndarray) -> float:
    """Return the absolute Pearson correlation between continuous variables."""
    if first.shape[0] < 2:
        return 0.0
    if not (first.std() > 0.0 and second.std() > 0.0):
        return 0.0
    value = float(np.corrcoef(first, second)[0, 1])
    return abs(value) if math.isfinite(value) else 0.0


def _as_2d_target(target: np.ndarray) -> np.ndarray:
    """Return a regression target with one column per output."""
    return target.reshape(-1, 1) if target.ndim == 1 else target


def _resolve_output_weights(
    output_weights: object | None,
    outputs: int,
) -> np.ndarray | None:
    """Return validated output weights scaled to keep their weighted mean finite."""
    if output_weights is None:
        return None
    weights = np.asarray(output_weights, dtype=np.float64).reshape(-1)
    if weights.shape[0] != outputs:
        raise ValueError(
            f"output_weights has {weights.shape[0]} entries but the target has "
            f"{outputs} output dimension(s)"
        )
    if not bool(np.isfinite(weights).all()):
        raise ValueError("output_weights must contain only finite values")
    if bool((weights < 0).any()):
        raise ValueError("output_weights must be non-negative")
    maximum = float(weights.max())
    if maximum == 0.0:
        raise ValueError("output_weights must have positive total mass")
    return weights / maximum


def _finalise_scores(scores: np.ndarray) -> np.ndarray:
    """Return a float64 feature-weight simplex, uniform when no score is positive."""
    total = float(scores.sum())
    if not total > 0.0:
        return np.full(scores.shape[0], 1.0 / scores.shape[0])
    return scores / total


def _to_tensor(
    scores: np.ndarray,
    *,
    dtype: torch.dtype,
    device: torch.device | str | None,
) -> torch.Tensor:
    """Return detached feature weights on the requested placement."""
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("dtype must be torch.float32 or torch.float64")
    return torch.tensor(scores, dtype=dtype, device=device)


def _correlation_scores(
    X_labelled: np.ndarray,
    codes_labelled: list[np.ndarray],
    target_labelled: np.ndarray,
    task: Literal["classification", "regression"],
    resolved_output_weights: np.ndarray | None,
) -> np.ndarray:
    """Return unnormalised correlation scores for at least two labelled rows."""
    continuous_width = X_labelled.shape[1]
    scores = np.zeros(continuous_width + len(codes_labelled))
    if task == "classification":
        class_codes = target_labelled.astype(np.int64, copy=False)
        for index in range(continuous_width):
            scores[index] = _correlation_ratio(X_labelled[:, index], class_codes)
        for index, codes in enumerate(codes_labelled, start=continuous_width):
            scores[index] = _cramers_v(codes, class_codes)
    else:
        outputs = target_labelled.shape[1]
        for index in range(continuous_width):
            scores[index] = float(
                np.average(
                    [
                        _absolute_pearson(X_labelled[:, index], target_labelled[:, output])
                        for output in range(outputs)
                    ],
                    weights=resolved_output_weights,
                )
            )
        for index, codes in enumerate(codes_labelled, start=continuous_width):
            scores[index] = float(
                np.average(
                    [
                        _correlation_ratio(target_labelled[:, output], codes)
                        for output in range(outputs)
                    ],
                    weights=resolved_output_weights,
                )
            )
    return scores


def _mutual_info_scores(
    X_labelled: np.ndarray,
    codes_labelled: list[np.ndarray],
    target_labelled: np.ndarray,
    task: Literal["classification", "regression"],
    resolved_output_weights: np.ndarray | None,
    random_state: int | None,
) -> np.ndarray:
    """Return unnormalised mutual-information scores for at least two labelled rows."""
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

    continuous_width = X_labelled.shape[1]
    code_block = (
        np.column_stack(codes_labelled) if codes_labelled else np.empty((X_labelled.shape[0], 0))
    )
    codes = code_block.astype(np.float64, copy=False)  # pragma: no mutate
    design = np.hstack((X_labelled, codes))
    discrete = np.array([False] * continuous_width + [True] * len(codes_labelled))
    if task == "classification":
        class_codes = target_labelled.astype(np.int64, copy=False)
        return mutual_info_classif(
            design,
            class_codes,
            discrete_features=discrete,
            random_state=random_state,
        )
    per_target = [
        mutual_info_regression(
            design,
            target_labelled[:, output],
            discrete_features=discrete,
            random_state=random_state,
        )
        for output in range(target_labelled.shape[1])
    ]
    return np.average(per_target, axis=0, weights=resolved_output_weights)


def _feature_weight_scores(
    X_cont: object,
    categorical_codes: Sequence[object],
    target: object,
    task: Literal["classification", "regression"],
    method: Literal["correlation", "mutual_info"],
    labelled_mask: object | None,
    output_weights: object | None,
    random_state: int | None,
) -> np.ndarray:
    """Return the float64 feature-weight simplex scored by ``method``."""
    X_labelled, codes_labelled, target_labelled = _prepare_labelled_arrays(
        X_cont,
        categorical_codes,
        target,
        task,
        labelled_mask,
    )
    resolved_output_weights = (
        _resolve_output_weights(output_weights, target_labelled.shape[1])
        if task == "regression"
        else None
    )
    if X_labelled.shape[0] < 2:
        scores = np.zeros(X_labelled.shape[1] + len(codes_labelled))
    elif method == "correlation":
        scores = _correlation_scores(
            X_labelled, codes_labelled, target_labelled, task, resolved_output_weights
        )
    else:
        scores = _mutual_info_scores(
            X_labelled,
            codes_labelled,
            target_labelled,
            task,
            resolved_output_weights,
            random_state,
        )
    return _finalise_scores(scores)


def feature_weights(
    X_cont: object,
    categorical_codes: Sequence[object],
    target: object,
    *,
    task: Literal["classification", "regression"],
    method: Literal["correlation", "mutual_info"],
    labelled_mask: object | None = None,
    output_weights: object | None = None,
    random_state: int | None = None,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return feature weights scored from each feature's relationship with the target.

    Continuous features precede categorical features in the returned tensor.
    ``"correlation"`` scores a feature by correlation ratio, Cramér's V or absolute
    Pearson correlation according to its modality and the task. ``"mutual_info"``
    scores it with scikit-learn's mutual-information estimator. When fewer than two
    rows are labelled or no score is positive, the weights are uniform.

    Args:
        X_cont: Two-dimensional continuous input accepted by NumPy.
        categorical_codes: One one-dimensional code array per categorical feature.
        target: One-dimensional classification codes or one- or two-dimensional
            regression targets.
        task: Statistical relationship to measure.
        method: Statistic that scores each feature, ``"correlation"`` or
            ``"mutual_info"``.
        labelled_mask: Optional boolean array selecting labelled rows.
        output_weights: Optional regression-output weights used when averaging scores.
        random_state: Seed passed to scikit-learn's mutual-information estimator.
            It must be ``None`` for ``"correlation"``, which draws no random numbers.
        dtype: Floating dtype of the returned tensor. Statistical calculations use
            float64 regardless of this output dtype.
        device: Device of the returned tensor.

    Returns:
        Detached, non-negative feature weights with unit total mass.

    Raises:
        ValueError: If ``method`` is unknown, ``random_state`` accompanies
            ``"correlation"``, the arrays are incompatible or the tensor placement
            is unsupported.
    """
    if method not in ("correlation", "mutual_info"):
        raise ValueError("method must be 'correlation' or 'mutual_info'")
    if method == "correlation" and random_state is not None:
        raise ValueError("random_state must be None for correlation")
    scores = _feature_weight_scores(
        X_cont,
        categorical_codes,
        target,
        task,
        method,
        labelled_mask,
        output_weights,
        random_state,
    )
    return _to_tensor(scores, dtype=dtype, device=device)
