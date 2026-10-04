"""Common validation of retained floating parameters and stochastic axes."""

import torch

from entlearn.network.validation import _validate_distribution_rows, _validate_storage


def validate_tensor(
    value: object,
    name: str,
    shape: tuple[int, ...] | None = None,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    finite: bool = True,
) -> torch.Tensor:
    """Require concrete floating storage, dimensions, placement and finite values."""
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a tensor")
    _validate_storage(value, name=name)
    if value.device.type == "meta" or value.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"{name} must be a concrete float32 or float64 tensor")
    if shape is not None and value.shape != shape:
        raise ValueError(f"{name} has invalid shape; expected {shape}")
    if dtype is not None and value.dtype != dtype:
        raise ValueError(f"{name} must use the computation dtype")
    if device is not None and value.device != device:
        raise ValueError(f"{name} must use the computation device")
    if finite and not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")
    return value


def validate_simplex_tensor(value: torch.Tensor, name: str, dim: int = -1) -> None:
    """Validate a retained stochastic axis without renormalising promoted values."""
    rows = value.movedim(dim, -1)
    if rows.ndim == 1:
        rows = rows[None]
    # Retained float32 rounding survives promotion and subsequent frozen updates.
    _validate_distribution_rows(rows, name=name, allow_zero=False, precision=torch.float32)
