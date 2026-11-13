"""Stochastic normalisation along either axis, the objective's logarithm, and scratch views."""

from __future__ import annotations

import torch


def _eps(dtype: torch.dtype) -> float:
    """Return the machine precision of ``dtype``, the package's relative-precision unit.

    It compares a quantity with another on the same scale: round-off tolerances, the guard
    on a relative-change denominator and the relative ridge on a rescaled manifold
    covariance or Gram matrix.
    """
    return torch.finfo(dtype).eps


def _tiny(dtype: torch.dtype) -> float:
    """Return the smallest positive normal number of ``dtype``, the package's range guard.

    A computed mass or probability at or below it is numerically zero.
    It guards divisors and logarithms, and it is the empty-mass threshold.
    """
    return torch.finfo(dtype).tiny


def _is_soft(temp: float) -> bool:
    """Return whether ``temp`` assigns softly.

    Zero is the hard regime and any positive setting is soft. A softmax subtracts the minimum
    cost before scaling.
    """
    return temp > 0.0


def normalise_(theta: torch.Tensor, dim: int, scratch_K: torch.Tensor) -> None:
    """Project ``theta`` onto a stochastic simplex along ``dim``, in-place.

    A slice whose mass is at or below ``finfo(dtype).tiny`` counts as empty and becomes
    uniform, as an empty categorical centroid does.

    Args:
        theta: ``(K_target, K_source)``, normalised in place; must be non-negative.
        dim: Axis summed over: ``0`` normalises the columns, ``1`` the rows.
        scratch_K: Workspace for the sums: ``(K_source,)`` for ``dim=0``,
            ``(K_target,)`` for ``dim=1``.
    """
    uniform = 1.0 / theta.shape[dim]
    torch.sum(theta, dim=dim, out=scratch_K)  # slice sums over dim
    torch.threshold_(scratch_K, _tiny(theta.dtype), 0.0)  # an empty slice divides by zero
    theta.div_(scratch_K.unsqueeze(dim))
    theta.nan_to_num_(nan=uniform, posinf=uniform)  # an empty slice (0/0 or x/0) -> uniform


def floored_log_(out: torch.Tensor, source: torch.Tensor) -> None:
    """Write ``log(max(source, tiny))`` into ``out``, the objective's logarithm of a probability.

    ``tiny`` is the smallest positive normal number of ``out``'s dtype. Every normal
    probability keeps its exact logarithm, even below machine precision. A
    numerically zero probability, zero or subnormal, receives
    ``log(tiny)``: about ``-708`` in float64 and ``-87`` in float32. The logarithm is
    therefore finite, and a zero weight on it contributes zero to a matrix
    product.

    Args:
        out: Output buffer, overwritten.
        source: Non-negative tensor, read-only.
    """
    torch.clamp_min(source, _tiny(out.dtype), out=out)  # copy + floor in one op
    out.log_()


def _scratch_view(buffer: torch.Tensor, columns: int, rows: int | None = None) -> torch.Tensor:
    """Return a contiguous ``(rows, columns)`` matrix over the first elements of ``buffer``.

    ``rows`` defaults to all of ``buffer``'s rows, whatever its width.

    Raises:
        ValueError: If ``buffer`` is not contiguous.
    """
    if not buffer.is_contiguous():
        raise ValueError(
            f"scratch buffer of shape {tuple(buffer.shape)} and strides {buffer.stride()} "
            "is not contiguous"
        )
    rows = buffer.shape[0] if rows is None else rows
    return buffer.as_strided((rows, columns), (columns, 1))
