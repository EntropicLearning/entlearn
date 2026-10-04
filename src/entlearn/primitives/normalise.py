"""Stochastic normalisation along either axis, and log-cache refresh."""

from __future__ import annotations

import torch


def _eps(dtype: torch.dtype) -> float:
    """Return the machine precision of ``dtype``.

    The package's single precision floor. It serves as the safe-log floor, the
    empty-cluster threshold, the hard/soft gate on a temperature setting such as
    ``epsilon`` (never on a per-row temperature such as ``epsilon / T``), and the guard on
    a relative-change denominator.
    """
    return torch.finfo(dtype).eps


def _is_soft(temp: float, dtype: torch.dtype) -> bool:
    """Return whether temperature setting ``temp`` assigns softly rather than by hard minimum."""
    return temp > _eps(dtype)


def normalise_(theta: torch.Tensor, dim: int, scratch_K: torch.Tensor) -> None:
    """Project ``theta`` onto a stochastic simplex along ``dim``, in-place.

    Args:
        theta: ``(K_target, K_source)``, normalised in place; must be non-negative.
        dim: Axis summed over: ``0`` normalises the columns, ``1`` the rows.
        scratch_K: Workspace for the sums: ``(K_source,)`` for ``dim=0``,
            ``(K_target,)`` for ``dim=1``.
    """
    uniform = 1.0 / theta.shape[dim]
    torch.sum(theta, dim=dim, out=scratch_K)  # slice sums over dim
    theta.div_(scratch_K.unsqueeze(dim))
    theta.nan_to_num_(nan=uniform, posinf=uniform, neginf=uniform)  # empty slice (0/0) -> uniform


def floored_log_(out: torch.Tensor, source: torch.Tensor) -> None:
    """Write ``log(max(source, eps))`` into ``out``.

    ``eps`` is the machine precision of ``out``'s dtype, so zero entries give a large
    negative number rather than ``-inf``.

    Args:
        out: Output buffer, overwritten.
        source: Non-negative tensor, read-only.
    """
    eps = _eps(out.dtype)
    torch.clamp_min(source, eps, out=out)  # copy + floor in one op
    out.log_()
