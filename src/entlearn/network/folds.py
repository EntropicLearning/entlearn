"""Validation of caller-owned row partitions, without splitter or search policy."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from entlearn.network.data import _StagedData

type _Fold = tuple[torch.Tensor, torch.Tensor]


def _row_indices(rows: object, data: _StagedData, name: str) -> torch.Tensor:
    """Validate and copy a non-empty ordered set of indices into the fit rows."""
    if (
        not isinstance(rows, torch.Tensor)
        or rows.dtype != torch.int64
        or rows.ndim != 1
        or rows.device != data.X_cont.device
    ):
        raise ValueError(f"{name} must be a one-dimensional int64 tensor on the data device")
    if not rows.numel() or bool(((rows < 0) | (rows >= len(data.X_cont))).any()):
        raise ValueError(f"{name} must contain indices in [0, {len(data.X_cont)})")
    if rows.unique().numel() != rows.numel():
        raise ValueError(f"{name} must not contain duplicate indices")
    return rows.detach().clone()


def _validation_folds(pairs: object, data: _StagedData) -> tuple[_Fold, ...]:
    """Snapshot materialised, disjoint train-validation pairs with scorable partitions."""
    if not isinstance(pairs, Sequence) or not pairs:
        raise ValueError("validation_pairs must be a non-empty materialised sequence of pairs")
    folds = []
    for ordinal, pair in enumerate(pairs):
        name = f"validation_pairs[{ordinal}]"
        if not isinstance(pair, Sequence) or len(pair) != 2:
            raise ValueError(f"{name} must contain training and validation indices")
        train = _row_indices(pair[0], data, name)
        validation = _row_indices(pair[1], data, name)
        if bool(torch.isin(train, validation).any()):
            raise ValueError(f"{name} training and validation rows must be disjoint")
        for partition, rows in (("training", train), ("validation", validation)):
            positive = data.labelled[rows].clone()
            if data.task_weights is not None:
                positive &= data.task_weights[rows] > 0
            if data.class_weights is not None:
                positive &= ((data.target[rows] > 0) & (data.class_weights > 0)).any(dim=1)
            if not bool(positive.any()):
                raise ValueError(f"{name} {partition} needs positive-weight labelled rows")
        folds.append((train, validation))
    return tuple(folds)
