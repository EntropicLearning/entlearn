"""Temperature softmax, hard argmin assignment, and the log-partition of a softmax."""

from __future__ import annotations

import math

import torch

from entlearn.primitives.normalise import _is_soft, _tiny


def _scale_ratio_(out: torch.Tensor, numerator: float, denominator: float) -> None:
    """Scale ``out`` by a ratio of positive scalars.

    If the ratio would overflow, it is calculated in a way that ``out`` times the ratio is
    still representable in the computation dtype.
    """
    tiny = _tiny(out.dtype)
    ratio = numerator / denominator
    if tiny <= ratio <= 1.0 / tiny:
        out.mul_(ratio)
        return
    a, ea = math.frexp(numerator)
    b, eb = math.frexp(denominator)
    mantissa, exponent = a / b, ea - eb
    if mantissa < 1.0:
        mantissa *= 2.0
        exponent -= 1
    limit = -math.frexp(tiny)[1]
    while abs(exponent) > limit:
        step = limit if exponent > 0 else -limit
        out.mul_(math.ldexp(1.0, step))
        exponent -= step
    out.mul_(math.ldexp(mantissa, exponent))


def softmax_with_temp_(
    out: torch.Tensor,
    cost: torch.Tensor,
    temp: float,
    dim: int,
    scratch_keepdim: torch.Tensor,
    row_weights: torch.Tensor | float = 1.0,
) -> None:
    """Write ``softmax(-cost / (temp · row_weights), dim)`` into ``out``.

    The minimum cost is subtracted before scaling. A slice whose minimum is not finite
    becomes the uniform ``1/N``, ``N = cost.shape[dim]``. With a tensor of row weights, a
    temperature of at least one divides before them and a smaller one after them, so with
    weights at most one no intermediate quotient exceeds the final exponent.

    Args:
        out: ``(..., N)`` output, overwritten. It may alias ``cost``.
        cost: ``(..., N)`` cost tensor, read-only unless aliased.
        temp: Softmax temperature; must be finite and ``> 0``.
        dim: Axis to softmax over.
        scratch_keepdim: Workspace shaped like ``cost`` with ``dim`` collapsed to length one.
        row_weights: Weights in ``(0, 1]`` on the temperature: one for every slice, or a
            ``(T,)`` tensor when ``dim=1``.
    """
    torch.amin(cost, dim=dim, keepdim=True, out=scratch_keepdim)
    torch.sub(scratch_keepdim, cost, out=out)  # -(cost - min), zero at the minimum
    if not isinstance(row_weights, torch.Tensor):
        _scale_ratio_(out, 1.0 / row_weights, temp)
    elif temp >= 1.0:
        _scale_ratio_(out, 1.0, temp)
        out.div_(row_weights.unsqueeze(1))
    else:
        out.div_(row_weights.unsqueeze(1))
        _scale_ratio_(out, 1.0, temp)
    torch.softmax(out, dim, out=out)
    out.nan_to_num_(nan=1.0 / cost.shape[dim])  # an infinite minimum (inf - inf) -> uniform


def argmin_assign_(
    out: torch.Tensor,
    cost: torch.Tensor,
    dim: int,
    scratch_keepdim: torch.Tensor,
    scratch_idx: torch.Tensor,
) -> None:
    """Write a one-hot at the minimum of ``cost`` along ``dim`` into ``out``.

    The ``temp -> 0`` limit of :func:`softmax_with_temp_`. Ties go to the first index.

    Args:
        out: ``(..., N)`` output, overwritten.
        cost: ``(..., N)`` cost tensor, read-only.
        dim: Axis to take the minimum over.
        scratch_keepdim: ``(..., 1)`` workspace for the minimum.
        scratch_idx: ``(..., 1)`` int64 workspace for its index.
    """
    torch.min(cost, dim=dim, keepdim=True, out=(scratch_keepdim, scratch_idx))
    out.zero_()
    out.scatter_(dim, scratch_idx, 1.0)


def compute_log_partition_(
    out_log_sum: torch.Tensor,
    out_shift: torch.Tensor,
    b: torch.Tensor,
    temp: float,
    scratch_T: torch.Tensor,
) -> None:
    """Write the log-partition ``log Σ_t exp(-b[t]/temp)`` in two finite parts.

    The log-partition is ``out_log_sum - out_shift/temp``, with the shift
    ``out_shift = min_t b[t]`` and ``out_log_sum = log Σ_t exp(-(b[t] - out_shift)/temp)``
    in ``[0, log T]``. Both parts are finite at any positive temperature, where the
    log-partition itself can overflow, and the softmax weight of a cost ``b`` is
    ``exp(-(b - out_shift)/temp - out_log_sum)``. Both outputs are overwritten.

    Args:
        out_log_sum: ``()`` output, overwritten with the shifted log-sum.
        out_shift: ``()`` output, overwritten with the smallest cost.
        b: ``(T,)`` finite cost vector, read-only. ``T >= 1``.
        temp: Softmax temperature; must be finite and ``> 0``.
        scratch_T: ``(T,)`` workspace for the shifted exponentials.
    """
    torch.amin(b, dim=0, out=out_shift)
    torch.sub(out_shift, b, out=scratch_T)  # -(b - min), zero at the minimum
    _scale_ratio_(scratch_T, 1.0, temp)
    scratch_T.exp_()
    torch.sum(scratch_T, dim=0, out=out_log_sum)  # Σ_t exp(-(b[t] - min)/temp), in [1, T]
    out_log_sum.log_()


def assign_simplex_(
    out: torch.Tensor,
    cost: torch.Tensor,
    temp: float,
    dim: int,
    scratch_keepdim: torch.Tensor,
    scratch_idx: torch.Tensor,
) -> None:
    """Assign ``cost`` onto the simplex along ``dim``: soft softmax, or hard one-hot.

    Applies :func:`softmax_with_temp_` at a positive temperature and :func:`argmin_assign_`
    at zero.

    Args:
        out: ``(..., N)`` output, overwritten.
        cost: ``(..., N)`` cost tensor, read-only.
        temp: Finite, non-negative softmax temperature.
        dim: Axis to assign along.
        scratch_keepdim: ``(..., 1)`` workspace for the minimum cost.
        scratch_idx: ``(..., 1)`` int64 workspace for the hard route.
    """
    if _is_soft(temp):
        softmax_with_temp_(out, cost, temp, dim, scratch_keepdim)
        return
    argmin_assign_(out, cost, dim, scratch_keepdim, scratch_idx)


def weighted_assign_simplex_(
    out: torch.Tensor,
    cost: torch.Tensor,
    temp: float,
    row_weights: torch.Tensor | float,
    scratch_keepdim: torch.Tensor,
    scratch_idx: torch.Tensor,
) -> None:
    """Assign rows onto the simplex when row ``t``'s entropy carries weight ``row_weights[t]``.

    Row ``t`` receives ``softmax(-cost[t] / (temp · row_weights[t]))``; a float
    ``row_weights`` is one weight shared by every row. A zero temperature assigns every
    row by hard minimum.

    Args:
        out: ``(T, K)`` row-stochastic output (modified).
        cost: ``(T, K)`` weighted assignment cost, read-only.
        temp: Finite, non-negative affiliation temperature.
        row_weights: ``(T,)`` row weights in ``(0, 1]``, or one such weight for every row.
        scratch_keepdim: ``(T, 1)`` workspace for the row minimum (modified).
        scratch_idx: ``(T, 1)`` int64 workspace for the hard assignment.
    """
    if not _is_soft(temp):
        argmin_assign_(out, cost, 1, scratch_keepdim, scratch_idx)
        return
    softmax_with_temp_(out, cost, temp, 1, scratch_keepdim, row_weights)
