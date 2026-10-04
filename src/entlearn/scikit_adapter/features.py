"""State-free tabular layout learning, vocabulary replay, tensor staging and reconstruction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import narwhals.stable.v2 as nw
import numpy as np
import torch
from narwhals.dependencies import (
    is_into_dataframe,
    is_narwhals_dataframe,
    is_narwhals_lazyframe,
    is_polars_lazyframe,
)
from sklearn.utils import _safe_indexing
from sklearn.utils.validation import _num_features, check_array

from entlearn import ReconstructionResult
from entlearn._warnings import _warn

try:
    import pandas as pd
except ImportError:
    pd = None


@dataclass(frozen=True, eq=False)
class FeatureLayout:
    """The fitted column split and categorical vocabularies.

    Every original column appears in exactly one position tuple, each sorted in
    original feature order: ``continuous_indices``, kept ``categorical_indices``
    with their sorted level vocabularies in ``categories``, or ``dropped_indices``
    for single-level categorical columns with their one-level vocabularies in
    ``dropped_categories``. ``tensor_positions`` gives the original position of
    each Network feature: continuous positions followed by kept categorical
    positions. Dropped columns never reach the Network: ``n_columns`` still counts
    them, so reports can restore the original feature order.
    """

    continuous_indices: tuple[int, ...]
    categorical_indices: tuple[int, ...]
    dropped_indices: tuple[int, ...]
    categories: tuple[np.ndarray, ...]
    dropped_categories: tuple[np.ndarray, ...]

    @property
    def tensor_positions(self) -> tuple[int, ...]:
        """Return the original column position of each Network feature, in tensor order."""
        return self.continuous_indices + self.categorical_indices

    @property
    def n_columns(self) -> int:
        """Return the original column count, dropped columns included."""
        return len(self.tensor_positions) + len(self.dropped_indices)

    def as_tabular(
        self, result: ReconstructionResult, feature_names: tuple[str, ...] | None
    ) -> TabularReconstruction:
        """Convert values and attach fitted tabular meanings."""
        continuous = result.continuous.cpu().numpy()
        columns = {
            position: (probability.cpu().numpy(), levels.copy())
            for position, probability, levels in zip(
                self.categorical_indices, result.categorical, self.categories, strict=True
            )
        }
        for position, levels in zip(self.dropped_indices, self.dropped_categories, strict=True):
            columns[position] = (
                np.ones((continuous.shape[0], 1), dtype=continuous.dtype),
                levels.copy(),
            )
        indices = tuple(sorted(columns))
        return TabularReconstruction(
            continuous=continuous,
            categorical=tuple(columns[index][0] for index in indices),
            continuous_indices=self.continuous_indices,
            categorical_indices=indices,
            categories=tuple(columns[index][1] for index in indices),
            dropped_indices=self.dropped_indices,
            feature_names=feature_names,
        )


@dataclass(frozen=True, eq=False)
class TabularReconstruction:
    """Reconstructed values with original feature positions and categorical meanings.

    ``continuous`` columns correspond to ``continuous_indices``. Each matrix in
    ``categorical`` contains probabilities over its matching ``categories`` vector,
    and occupies the matching original ``categorical_indices`` position. Both
    position sequences are sorted in original feature order.

    A continuous-only layout has empty ``categorical``, ``categorical_indices`` and
    ``categories``. Dropped single-level columns have one-column distributions of
    ones over their fitted singleton vocabulary. ``dropped_indices`` distinguishes
    these constants from model reconstructions. Query values in dropped columns are
    ignored. ``feature_names`` contains fitted names when available, otherwise ``None``.
    """

    continuous: np.ndarray
    categorical: tuple[np.ndarray, ...]
    continuous_indices: tuple[int, ...]
    categorical_indices: tuple[int, ...]
    categories: tuple[np.ndarray, ...]
    dropped_indices: tuple[int, ...]
    feature_names: tuple[str, ...] | None


def _is_dataframe(X: object) -> bool:
    return is_into_dataframe(X)


def _frame(X: Any) -> nw.DataFrame[Any]:
    return nw.from_native(X, eager_only=True)


def _column_count(X: object) -> int:
    try:
        return _num_features(X)
    except TypeError as exc:
        raise ValueError(str(exc)) from exc


def _column_values(X: object, index: int) -> np.ndarray:
    if _is_dataframe(X):
        frame = _frame(X)
        return frame.get_column(frame.columns[index]).to_numpy()
    return np.asarray(_safe_indexing(X, index, axis=1))


def _has_missing(values: np.ndarray) -> bool:
    if values.dtype == object:
        if pd is not None:
            mask = np.asarray(pd.isna(values))
            if mask.dtype == bool:
                return bool(mask.any())
        return any(
            value is None
            or (isinstance(value, (float, np.floating)) and np.isnan(value))
            or (isinstance(value, (np.datetime64, np.timedelta64)) and np.isnat(value))
            for value in values
        )
    if np.issubdtype(values.dtype, np.floating):
        return bool(np.isnan(values).any())
    if np.issubdtype(values.dtype, np.datetime64) or np.issubdtype(values.dtype, np.timedelta64):
        return bool(np.isnat(values).any())
    return False


def _is_categorical_dtype(dtype: object) -> bool:
    return dtype in (nw.Categorical, nw.Enum, nw.String, nw.Object)


def _resolve_categorical_indices(
    X: object, categorical_features: Sequence[int] | Literal["from_dtype"] | None
) -> list[int]:
    n_cols = _column_count(X)
    if categorical_features is None:
        return []
    if isinstance(categorical_features, str):
        if categorical_features != "from_dtype":
            raise ValueError(
                f"categorical_features string must be 'from_dtype', got {categorical_features!r}"
            )
        if _is_dataframe(X):
            return [
                position
                for position, dtype in enumerate(_frame(X).schema.values())
                if _is_categorical_dtype(dtype)
            ]
        if np.asarray(X).dtype == object:
            raise ValueError(
                "categorical_features='from_dtype' cannot discriminate the columns of "
                "an object-dtype array; pass explicit integer column indices, or an "
                "eager DataFrame whose columns carry per-column dtypes"
            )
        return []
    indices = list(categorical_features)
    seen: set[int] = set()
    for index in indices:
        if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)):
            raise TypeError(
                f"categorical feature entry {index!r} is not an integer column index; "
                "pass integer column indices or 'from_dtype'"
            )
        if index < 0 or index >= n_cols:
            raise ValueError(f"categorical feature index {index} is outside [0, {n_cols})")
        if index in seen:
            raise ValueError(f"duplicate categorical feature index {index}")
        seen.add(index)
    return sorted(indices)


_NUMPY_STAGING_DTYPE = {torch.float32: np.float32, torch.float64: np.float64}


def _staging_dtype(dtype: torch.dtype) -> type:
    return _NUMPY_STAGING_DTYPE[dtype]


def _continuous_block(X: object, continuous_indices: Sequence[int], dtype: type) -> np.ndarray:
    n_rows = len(_frame(X)) if _is_dataframe(X) else np.asarray(X).shape[0]
    if not continuous_indices:
        return np.empty((n_rows, 0), dtype=dtype)
    if not _is_dataframe(X):
        arr = np.asarray(X)
        if arr.dtype != object and np.issubdtype(arr.dtype, np.number):
            if np.iscomplexobj(arr):
                raise ValueError("Complex data not supported")
            selected = (
                arr
                if tuple(continuous_indices) == tuple(range(arr.shape[1]))
                else arr[:, list(continuous_indices)]
            )
            with np.errstate(over="ignore"):
                return np.array(selected, dtype=dtype, order="C")
    columns = []
    for index in continuous_indices:
        raw = _column_values(X, index)
        if np.iscomplexobj(raw):
            raise ValueError("Complex data not supported")
        try:
            with np.errstate(over="ignore"):
                columns.append(raw.astype(dtype))
        except ValueError as exc:
            raise ValueError(
                f"continuous column at position {index} is not numeric; cast it to a "
                "numeric dtype or mark it categorical"
            ) from exc
    return np.column_stack(columns)


def _fit_feature_layout(
    X: object,
    categorical_features: Sequence[int] | Literal["from_dtype"] | None,
    dtype: type,
) -> tuple[FeatureLayout, np.ndarray, list[np.ndarray]]:
    """Learn all modality positions and sorted level vocabularies."""
    n_cols = _column_count(X)
    cat_indices = _resolve_categorical_indices(X, categorical_features)
    cat_set = set(cat_indices)
    continuous_indices = tuple(i for i in range(n_cols) if i not in cat_set)
    kept_indices: list[int] = []
    dropped_indices: list[int] = []
    categories: list[np.ndarray] = []
    dropped_categories: list[np.ndarray] = []
    codes: list[np.ndarray] = []
    for index in cat_indices:
        raw = _column_values(X, index)
        if _has_missing(raw):
            raise ValueError(f"categorical column at position {index} contains missing values")
        levels = np.unique(raw)
        if levels.shape[0] == 1:
            _warn(
                f"categorical column at position {index} has a single level "
                f"({levels[0]!r}) and is dropped, a constant column carries no signal",
                UserWarning,
            )
            dropped_indices.append(index)
            dropped_categories.append(levels)
            continue
        kept_indices.append(index)
        categories.append(levels)
        codes.append(np.searchsorted(levels, raw).astype(np.int64))
    layout = FeatureLayout(
        continuous_indices=continuous_indices,
        categorical_indices=tuple(kept_indices),
        dropped_indices=tuple(dropped_indices),
        categories=tuple(categories),
        dropped_categories=tuple(dropped_categories),
    )
    return layout, _continuous_block(X, continuous_indices, dtype), codes


def _apply_feature_layout(
    X: object, layout: FeatureLayout, dtype: type, *, check_dropped: bool = False
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Replay the fitted positions and vocabularies, rejecting unseen categorical values."""
    n_cols = _column_count(X)
    if n_cols != layout.n_columns:
        raise ValueError(
            f"X has {n_cols} column(s) but the fitted layout expects {layout.n_columns}"
        )
    codes: list[np.ndarray] = []
    indices = layout.categorical_indices + (layout.dropped_indices if check_dropped else ())
    categories = layout.categories + (layout.dropped_categories if check_dropped else ())
    for index, levels in zip(indices, categories, strict=True):
        raw = _column_values(X, index)
        if _has_missing(raw):
            raise ValueError(f"categorical column at position {index} contains missing values")
        positions = np.clip(np.searchsorted(levels, raw), 0, levels.shape[0] - 1)
        unknown_mask = levels[positions] != raw
        if unknown_mask.any():
            offending = np.unique(np.asarray(raw)[unknown_mask])
            raise ValueError(
                f"categorical column at position {index} contains level(s) unseen at "
                f"fit time: {list(offending)}; novel categorical levels are not supported"
            )
        if index in layout.categorical_indices:
            codes.append(positions.astype(np.int64))
    return _continuous_block(X, layout.continuous_indices, dtype), codes


def _resolve_device_dtype(
    device: str | torch.device | None, dtype: torch.dtype | None
) -> tuple[torch.device, torch.dtype]:
    resolved_dtype = torch.float64 if dtype is None else dtype
    if resolved_dtype is not torch.float32 and resolved_dtype is not torch.float64:
        raise ValueError("dtype must be torch.float32 or torch.float64")
    resolved_device = torch.device("cpu") if device is None else torch.device(device)
    if resolved_device.type != "cpu" and resolved_device.index is None:
        resolved_device = torch.empty(0, device=resolved_device, dtype=resolved_dtype).device
    return resolved_device, resolved_dtype


def _check_tabular_input(X: object) -> object:
    """Validate tables, preserving DataFrame dtypes and normalising other array-likes."""
    if is_narwhals_lazyframe(X) or is_polars_lazyframe(X):
        raise TypeError("Lazy input is not supported; call .collect() before fitting or prediction")
    if is_narwhals_dataframe(X):
        X = nw.to_native(X, pass_through=True)
    if not _is_dataframe(X):
        return check_array(X, dtype=None, ensure_all_finite=False)
    n_cols = _column_count(X)
    n_rows = len(_frame(X))
    if n_rows == 0 or n_cols == 0:
        check_array(X, dtype=None)
    return X


def _feature_tensors(
    X_cont: np.ndarray, codes: list[np.ndarray], *, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Check continuous values in computation precision and stage both modalities."""
    X_cont = check_array(X_cont, dtype=_staging_dtype(dtype), ensure_min_features=0)
    return (
        torch.as_tensor(X_cont, dtype=dtype, device=device),
        tuple(torch.as_tensor(code, dtype=torch.int64, device=device) for code in codes),
    )
