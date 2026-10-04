"""Tensor-native statistical calculations."""

from __future__ import annotations

import torch

from entlearn.primitives.normalise import _eps, floored_log_


def entropy(p: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Return Shannon entropy along ``dim``."""
    log_p = p.clamp_min(_eps(p.dtype)).log()
    return -(p * log_p).sum(dim=dim)


def effective_dimension(
    p: torch.Tensor,
    dim: int = -1,
    *,
    normalise: bool = True,
) -> torch.Tensor:
    """Return the exponential of entropy along ``dim``.

    Args:
        p: A probability tensor.
        dim: Axis that contains the probability vectors.
        normalise: Divide by the axis length when true.

    Returns:
        The effective dimension with ``dim`` removed.
    """
    result = entropy(p, dim=dim).exp()
    return result / p.shape[dim] if normalise else result


def inlier_scores_(
    out: torch.Tensor,
    Wt_train: torch.Tensor,
    Wt_test: torch.Tensor,
) -> None:
    """Evaluate the right-continuous empirical CDF of ``Wt_train`` at ``Wt_test``."""
    if Wt_train.numel() == 0:
        raise ValueError("Wt_train is empty: the empirical CDF needs at least one reference value")
    sorted_reference = torch.sort(Wt_train).values
    counts = torch.searchsorted(sorted_reference, Wt_test, right=True)
    out.copy_(counts.to(Wt_test.dtype) / sorted_reference.numel())


def entropy_penalty_(
    out_scalar: torch.Tensor,
    p: torch.Tensor,
    coefficient: float,
    log_buffer: torch.Tensor,
    scratch_scalar: torch.Tensor,
    *,
    row_weights: torch.Tensor | None = None,
) -> None:
    """Subtract a Shannon-entropy reward from the loss in ``out_scalar``.

    With ``H = -Σ p·log p`` over every element of ``p``, applies
    ``out_scalar -= coefficient·H``. When ``row_weights`` is given, each first-axis
    slice contributes its weight. The log is floored at the dtype machine precision,
    so zero entries do not contribute. ``log_buffer`` must match ``p``. The caller
    must zero ``out_scalar`` first.

    Args:
        out_scalar: ``()`` loss accumulator, decreased in place.
        p: Probability tensor of any shape, read-only.
        coefficient: Non-negative finite weight.
        log_buffer: Workspace shaped like ``p``.
        scratch_scalar: ``()`` workspace.
        row_weights: Optional ``(p.shape[0],)`` first-axis weights for a matrix ``p``.
    """
    floored_log_(log_buffer, p)  # floor + safe-log into log_buffer (p untouched)
    log_buffer.mul_(p)  # p · log p  (mutates log_buffer)
    if row_weights is not None:
        log_buffer.mul_(row_weights.unsqueeze(1))
    torch.sum(log_buffer, dim=tuple(range(log_buffer.ndim)), out=scratch_scalar)  # Σ p·log p = -H
    out_scalar.add_(scratch_scalar, alpha=coefficient)  # out += coef·Σ p·log p  ==  out -= coef·H
