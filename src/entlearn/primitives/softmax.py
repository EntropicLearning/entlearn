"""Temperature softmax, hard argmin assignment, and log-partition."""

from __future__ import annotations

import torch

from entlearn.primitives.normalise import _is_soft


def softmax_with_temp_(
    out: torch.Tensor,
    cost: torch.Tensor,
    temp: float,
    dim: int,
    scratch_keepdim: torch.Tensor,
) -> None:
    """Write ``softmax(-cost/temp, dim)`` into ``out``.

    Every slice along ``dim`` sums to 1. A slice whose exponentials are all
    non-finite becomes the uniform ``1/N``, ``N = cost.shape[dim]``.

    Args:
        out: ``(..., N)`` output, overwritten.
        cost: ``(..., N)`` cost tensor, read-only.
        temp: Softmax temperature; must be finite and ``> 0``.
        dim: Axis to softmax over.
        scratch_keepdim: ``(..., 1)`` workspace for the slice max and the normaliser.
    """
    n = cost.shape[dim]
    uniform = 1.0 / n
    torch.mul(cost, -1.0 / temp, out=out)  # -cost/temp
    torch.amax(out, dim=dim, keepdim=True, out=scratch_keepdim)  # row/col max (shift)
    out.sub_(scratch_keepdim).exp_()  # exp(shifted)
    torch.sum(out, dim=dim, keepdim=True, out=scratch_keepdim)  # Z
    out.div_(scratch_keepdim)  # softmax
    out.nan_to_num_(nan=uniform, posinf=uniform, neginf=uniform)  # all-inf slice -> uniform 1/N


def argmin_assign_(
    out: torch.Tensor,
    cost: torch.Tensor,
    dim: int,
    scratch_idx: torch.Tensor,
) -> None:
    """Write a one-hot at the minimum of ``cost`` along ``dim`` into ``out``.

    The ``temp -> 0`` limit of :func:`softmax_with_temp_`. Ties go to the first index.

    Args:
        out: ``(..., N)`` output, overwritten.
        cost: ``(..., N)`` cost tensor, read-only.
        dim: Axis to take the argmin over.
        scratch_idx: ``(..., 1)`` int64 workspace for the argmin index.
    """
    torch.argmin(cost, dim=dim, keepdim=True, out=scratch_idx)
    out.zero_()
    out.scatter_(dim, scratch_idx, 1.0)


def compute_log_partition_(
    out_scalar: torch.Tensor,
    b: torch.Tensor,
    temp: float,
    scratch_T: torch.Tensor,
    scratch_scalar: torch.Tensor,
) -> None:
    """Write the log-partition ``logsumexp_t(-b/temp)`` into ``out_scalar``.

    Computes ``m + log Σ_t exp(u[t] - m)`` with ``u = -b/temp`` and ``m = max_t u``,
    over the ``(T,)`` vector ``b``. ``out_scalar`` is overwritten, not added into.

    Args:
        out_scalar: ``()`` output, overwritten.
        b: ``(T,)`` cost vector, read-only. ``T >= 1``.
        temp: Softmax temperature; must be finite and ``> 0``.
        scratch_T: ``(T,)`` workspace for the shifted exponentials.
        scratch_scalar: ``()`` workspace for the max ``m``.
    """
    torch.mul(b, -1.0 / temp, out=scratch_T)  # u = -b/temp
    torch.amax(scratch_T, dim=0, out=scratch_scalar)  # m (())
    scratch_T.sub_(scratch_scalar).exp_()  # exp(u - m)
    torch.sum(scratch_T, dim=0, out=out_scalar)  # Σ exp(u - m)
    out_scalar.log_()  # log Σ
    out_scalar.add_(scratch_scalar)  # + m  => log Z


def assign_simplex_(
    out: torch.Tensor,
    cost: torch.Tensor,
    temp: float,
    dim: int,
    scratch_keepdim: torch.Tensor,
    scratch_idx: torch.Tensor,
) -> None:
    """Assign ``cost`` onto the simplex along ``dim``: soft softmax, or hard one-hot.

    Takes :func:`softmax_with_temp_` when the temperature exceeds the dtype machine
    precision, and :func:`argmin_assign_` when it does not.

    Args:
        out: ``(..., N)`` output, overwritten.
        cost: ``(..., N)`` cost tensor, read-only.
        temp: Softmax temperature; must be finite and ``> 0`` on the soft route.
        dim: Axis to assign along.
        scratch_keepdim: ``(..., 1)`` workspace for the soft route.
        scratch_idx: ``(..., 1)`` int64 workspace for the hard route.
    """
    if _is_soft(temp, out.dtype):
        softmax_with_temp_(out, cost, temp, dim, scratch_keepdim)
        return
    argmin_assign_(out, cost, dim, scratch_idx)


def weighted_assign_simplex_(
    out: torch.Tensor,
    cost: torch.Tensor,
    temp: float,
    row_weights: torch.Tensor,
    scratch_keepdim: torch.Tensor,
    scratch_idx: torch.Tensor,
) -> None:
    """Assign rows onto the simplex when row ``t``'s entropy carries weight ``row_weights[t]``.

    Row ``t`` receives ``softmax(-cost[t] / (temp · row_weights[t]))``. A temperature at or
    below the dtype machine precision assigns every row by hard minimum cost.

    Args:
        out: ``(T, K)`` row-stochastic output (modified).
        cost: ``(T, K)`` weighted assignment cost, read-only.
        temp: Raw affiliation temperature. Must be finite and ``> 0`` on the soft route.
        row_weights: ``(T,)`` positive row weights.
        scratch_keepdim: ``(T, 1)`` workspace (modified).
        scratch_idx: ``(T, 1)`` int64 workspace (modified).
    """
    if not _is_soft(temp, out.dtype):
        argmin_assign_(out, cost, 1, scratch_idx)
        return
    torch.div(cost, row_weights.unsqueeze(1), out=out)
    softmax_with_temp_(out, out, temp, 1, scratch_keepdim)
