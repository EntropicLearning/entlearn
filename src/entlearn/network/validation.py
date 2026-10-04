"""Tensor data validation for the Network boundary."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from entlearn._warnings import _warn
from entlearn.primitives.normalise import _eps

_FLOAT_DTYPES = (torch.float32, torch.float64)
_SIMPLEX_SUM_GUARD = 8


def _validate_prediction_features(
    X_cont: object,
    X_cat: Sequence[torch.Tensor] | None,
    centroids: torch.Tensor,
    cardinalities: tuple[int, ...],
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Check query storage and values against the fitted feature geometry."""
    X = _validate_continuous(X_cont, None)
    if X.dtype != centroids.dtype:
        raise ValueError("prediction input must use the fitted computation dtype")
    if X.device != centroids.device:
        raise ValueError("prediction input must use the fitted device")
    if X.shape[1] != centroids.shape[1]:
        raise ValueError("prediction input has the wrong continuous feature count")
    categorical, _ = _validate_categorical(
        X_cat,
        rows=X.shape[0],
        dtype=X.dtype,
        device=X.device,
        allow_conversion=False,
        expected=cardinalities,
    )
    return X, categorical


def _validate_storage(values: torch.Tensor, *, name: str) -> None:
    """Require ordinary strided storage on a device that contains data."""
    if values.layout != torch.strided:
        raise ValueError(f"{name} must use strided tensor storage")
    if values.device.type == "meta":
        raise ValueError(f"{name} must be on a device that contains data")


def _validate_distribution_rows(
    values: torch.Tensor,
    *,
    name: str,
    allow_zero: bool,
    precision: torch.dtype | None = None,
) -> None:
    """Require simplex rows, allowing an explicit precision for converted fitted state."""
    if not bool(torch.isfinite(values).all()):
        raise ValueError(f"{name} must contain only finite values")
    if bool((values < 0).any()):
        raise ValueError(f"{name} must be non-negative")
    row_sums = values.sum(dim=1)
    # A sum over M values accumulates O(M eps) round-off
    tolerance = (
        _SIMPLEX_SUM_GUARD
        * values.shape[1]
        * _eps(values.dtype if precision is None else precision)
    )
    valid = (row_sums - 1).abs() <= tolerance
    if allow_zero:
        valid |= (values == 0).all(dim=1)
    if not bool(valid.all()):
        raise ValueError(f"{name} rows must sum to one within the dtype tolerance")


def _validate_continuous(
    X_cont: object,
    computation_dtype: torch.dtype | None,
) -> torch.Tensor:
    """Return a valid continuous input tensor."""
    if not isinstance(X_cont, torch.Tensor):
        raise ValueError("X_cont must be a non-empty finite float32 or float64 matrix")
    _validate_storage(X_cont, name="X_cont")
    if X_cont.ndim != 2 or X_cont.shape[0] == 0 or X_cont.dtype not in _FLOAT_DTYPES:
        raise ValueError("X_cont must be a non-empty finite float32 or float64 matrix")
    if computation_dtype is not None and computation_dtype not in _FLOAT_DTYPES:
        raise ValueError("computation_dtype must be torch.float32 or torch.float64")
    staged = X_cont if computation_dtype is None else X_cont.to(dtype=computation_dtype)
    if not bool(torch.isfinite(staged).all()):
        raise ValueError("X_cont must be a non-empty finite float32 or float64 matrix")
    return staged


def _stage_classification_target(
    target: object,
    *,
    rows: int,
    dtype: torch.dtype,
    device: torch.device,
    n_classes: int | None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return a dense classification target, its labelled mask, and number of classes."""
    if not isinstance(target, torch.Tensor) or target.device != device:
        raise ValueError("classification y must be a tensor on the X_cont device")
    _validate_storage(target, name="classification y")
    if target.ndim == 1:
        if target.shape[0] != rows or target.dtype is not torch.int64:
            raise ValueError("classification y must be a length-T int64 tensor")
        if bool((target < -1).any()):
            raise ValueError("classification y permits -1 as its only negative code")
        labelled = target != -1
        if not bool(labelled.any()):
            raise ValueError("classification y must contain a labelled row")
        labelled_codes = target[labelled]
        if n_classes is None:
            width = int(labelled_codes.max().item()) + 1
            if not torch.equal(
                torch.unique(labelled_codes),
                torch.arange(width, dtype=torch.int64, device=device),
            ):
                raise ValueError(
                    "n_classes is not specified and classification y codes contain gaps"
                )
        else:
            width = n_classes
            if bool((labelled_codes >= width).any()):
                raise ValueError("classification y exceeds the declared number of classes")
        dense = torch.zeros(rows, width, dtype=dtype, device=device)
        dense[labelled, labelled_codes] = 1
        return dense, labelled, width

    if (
        target.ndim != 2
        or target.shape[0] != rows
        or target.shape[1] == 0
        or target.dtype not in _FLOAT_DTYPES
    ):
        raise ValueError("classification y must be int64 codes or a floating (T, M) tensor")
    if n_classes is not None and target.shape[1] != n_classes:
        raise ValueError("classification y columns must match the declared number of classes")
    dense = target.to(dtype=dtype)
    _validate_distribution_rows(dense, name="classification y", allow_zero=True)
    labelled = (dense != 0).any(dim=1)
    if not bool(labelled.any()):
        raise ValueError("classification y must contain a labelled row")
    return dense, labelled, dense.shape[1]


def _stage_weights(
    weights: object,
    *,
    name: str,
    length: int,
    dtype: torch.dtype,
    device: torch.device,
    positive: bool = False,
) -> torch.Tensor:
    """Return one finite, non-negative floating weight vector, strictly positive if requested."""
    if not isinstance(weights, torch.Tensor):
        raise ValueError(f"{name} must be a length-{length} floating tensor on the X_cont device")
    _validate_storage(weights, name=name)
    if (
        weights.ndim != 1
        or weights.shape[0] != length
        or weights.dtype not in _FLOAT_DTYPES
        or weights.device != device
    ):
        raise ValueError(f"{name} must be a length-{length} floating tensor on the X_cont device")
    staged = weights.to(dtype=dtype)
    if not bool(torch.isfinite(staged).all()):
        raise ValueError(f"{name} must contain only finite values")
    if positive and not bool((staged > 0).all()):
        raise ValueError(f"{name} must be positive; drop zero-weight rows instead")
    if bool((staged < 0).any()):
        raise ValueError(f"{name} must be non-negative")
    return staged


def _stage_sample_weights(
    weights: object, *, length: int, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the raw and normalised sample weights, each strictly positive in ``dtype``."""
    raw = _stage_weights(
        weights, name="sample_weights", length=length, dtype=dtype, device=device, positive=True
    )
    normalised = _normalise_weights(raw, name="sample_weights")
    if not bool((normalised > 0).all()):
        raise ValueError("sample_weights span too wide a range for the computation dtype")
    return raw, normalised


def _normalise_weights(weights: torch.Tensor, *, name: str) -> torch.Tensor:
    """Normalise a finite non-negative weight vector without sum overflow."""
    total = weights.sum()
    if bool(torch.isfinite(total)):
        if not bool(total > 0):
            raise ValueError(f"{name} must have positive total mass")
        return weights / total
    scaled = weights / weights.max()
    return scaled / scaled.sum()


def _stage_regression_target(
    target: object,
    *,
    rows: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return a two-dimensional regression target and its labelled mask."""
    if not isinstance(target, torch.Tensor) or target.device != device:
        raise ValueError("regression y must be a tensor on the X_cont device")
    _validate_storage(target, name="regression y")
    if target.dtype not in _FLOAT_DTYPES or target.ndim not in (1, 2) or target.shape[0] != rows:
        raise ValueError("regression y must be a floating length-T vector or (T, M) tensor")
    staged = target.to(dtype=dtype)
    if staged.ndim == 1:
        staged = staged[:, None]
    if staged.shape[1] == 0:
        raise ValueError("regression y must have a positive output dimension")

    missing = torch.isnan(staged)
    any_missing = missing.any(dim=1)
    partially_missing = any_missing & ~missing.all(dim=1)
    partial_count = int(partially_missing.sum())
    if partial_count:
        _warn(
            f"{partial_count} regression target row(s) are partially missing; "
            "each row is treated as wholly unlabelled",
            UserWarning,
        )
    labelled = ~any_missing
    if bool(torch.isinf(staged[labelled]).any()):
        raise ValueError("labelled regression y values must be finite")
    staged = torch.where(labelled[:, None], staged, torch.zeros((), dtype=dtype, device=device))
    return staged, labelled, staged.shape[1]


def _validate_categorical(
    X_cat: Sequence[torch.Tensor] | None,
    *,
    rows: int,
    dtype: torch.dtype,
    device: torch.device,
    allow_conversion: bool,
    expected: tuple[int, ...] | None = None,
) -> tuple[tuple[torch.Tensor, ...], tuple[int, ...]]:
    """Return valid categorical tensors and their cardinalities.

    ``expected`` names the fitted cardinality of each feature.

    Raises:
        ValueError: If a feature is malformed, or disagrees with ``expected``.
    """
    if X_cat is None:
        if expected:
            raise ValueError(f"X_cat must contain {len(expected)} fitted categorical feature(s)")
        return (), ()
    if isinstance(X_cat, (str, bytes)) or not isinstance(X_cat, Sequence):
        raise ValueError("X_cat must be a sequence of tensors")
    if expected is not None and len(X_cat) != len(expected):
        raise ValueError(f"X_cat must contain {len(expected)} fitted categorical feature(s)")

    staged_features = []
    cardinalities = []
    for position, feature in enumerate(X_cat):
        expected_cardinality = None if expected is None else expected[position]
        if not isinstance(feature, torch.Tensor):
            raise ValueError(f"X_cat[{position}] must be a tensor")
        if feature.device != device:
            raise ValueError(f"X_cat[{position}] must be on the X_cont device")
        _validate_storage(feature, name=f"X_cat[{position}]")
        if feature.ndim == 2:
            if (
                feature.shape[0] != rows
                or feature.shape[1] < 2
                or (expected_cardinality is not None and feature.shape[1] != expected_cardinality)
                or feature.dtype not in _FLOAT_DTYPES
                or (not allow_conversion and feature.dtype is not dtype)
            ):
                raise ValueError(
                    f"X_cat[{position}] must be a (T, M_i) computation-dtype distribution"
                )
            feature = feature.to(dtype=dtype)
            _validate_distribution_rows(
                feature,
                name=f"X_cat[{position}]",
                allow_zero=False,
            )
            staged_features.append(feature)
            cardinalities.append(feature.shape[1])
            continue
        if feature.ndim != 1 or feature.shape[0] != rows or feature.dtype is not torch.int64:
            raise ValueError(f"X_cat[{position}] must be a length-T int64 code tensor")
        if bool((feature < 0).any()):
            raise ValueError(f"X_cat[{position}] codes must be non-negative")
        highest = int(feature.max().item())
        if expected_cardinality is None:
            cardinality = highest + 1
            if cardinality < 2:
                raise ValueError(f"X_cat[{position}] must contain at least two levels")
        else:
            cardinality = expected_cardinality
            if highest >= cardinality:
                raise ValueError(
                    f"X_cat[{position}] holds code {highest}, outside the fitted "
                    f"[0, {cardinality}) levels"
                )
        staged_features.append(feature)
        cardinalities.append(cardinality)
    return tuple(staged_features), tuple(cardinalities)
