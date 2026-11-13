"""Optional starting value for the categorical scale ``delta_cat``.

The helper prices every column as if the clusters knew nothing about it: a continuous
column then costs its variance, and a categorical column costs ``delta_cat`` times its
cardinality-adjusted level entropy. The returned value makes the two typical costs equal.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch

from entlearn.network.blocks.input import _guess_delta_cat
from entlearn.network.validation import _stage_weights, _validate_continuous


def _has_missing(codes: np.ndarray) -> bool:
    """Return whether a code array holds NaN or ``None``."""
    if codes.dtype.kind in "fc":
        return bool(np.isnan(codes).any())
    if codes.dtype == object:
        return any(
            value is None or (isinstance(value, float | np.floating) and math.isnan(value))
            for value in codes
        )
    return False


def delta_cat_guess(
    X_cont: object,
    categorical_codes: Sequence[object],
    *,
    sample_weights: object | None = None,
) -> float:
    """Return a starting value for ``delta_cat``.

    Errors in a continuous feature have magnitude similar to its variance. Categorical
    features, on the other hand, have potentially unbounded cross-entropy errors, depending
    on their entropy and number of levels.

    This helper returns a scaling factor that brings the latter to a comparable range, using
    the median variance of the continuous features and the median cross-entropy of the
    categorical ones, both weighted by the sample weights.

    Every column is treated as uninformative, so the returned value is a heuristic that can
    be used as a starting point. Fitting uses it when ``Input.delta_cat`` is ``None``.

    Args:
        X_cont: Two-dimensional continuous input accepted by NumPy, scaled as for the fit.
            The calculation keeps a float32 or float64 input's dtype; other inputs use float64.
        categorical_codes: One one-dimensional code array per categorical feature. A
            feature's levels are the distinct codes it holds.
        sample_weights: Optional positive weights for the rows.

    Returns:
        The positive starting value.

    Raises:
        ValueError: If no continuous feature varies or no categorical feature has at least
            two levels, if ``X_cont`` holds a non-finite value or a code is NaN or ``None``,
            or if the arrays or weights are malformed.
    """
    array = np.asarray(X_cont)
    if array.dtype not in (np.float32, np.float64):
        array = array.astype(np.float64)
    # The checks Network.fit applies to the same inputs.
    X = _validate_continuous(torch.from_numpy(array), None)
    rows = X.shape[0]
    weights = None
    if sample_weights is not None:
        weights = _stage_weights(
            torch.from_numpy(np.asarray(sample_weights, dtype=np.float64)),
            name="sample_weights",
            length=rows,
            dtype=X.dtype,
            device=X.device,
            positive=True,
        )

    codes = []
    for position, column in enumerate(categorical_codes):
        array = np.asarray(column)
        if array.shape != (rows,):
            raise ValueError(f"categorical_codes[{position}] must have one entry per X_cont row")
        if _has_missing(array):
            raise ValueError(f"categorical_codes[{position}] contains NaN or None")
        codes.append(torch.from_numpy(np.unique(array, return_inverse=True)[1]))
    guess = _guess_delta_cat(X, codes, weights)
    if guess is None:
        raise ValueError(
            "delta_cat_guess needs a varying continuous feature and a categorical feature "
            "with at least two levels"
        )
    return guess
