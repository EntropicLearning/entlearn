"""Tensor-native statistical calculations."""

from __future__ import annotations

import torch


def entropy(p: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Return the Shannon entropy ``-Σ p·log p`` along ``dim``, with ``0·log 0 = 0``.

    The logarithm is exact for every positive entry, including subnormal entries.
    """
    return torch.special.entr(p).sum(dim=dim)


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
    """Add a negative-entropy penalty to the loss in ``out_scalar``.

    With ``H = -Σ p·log p`` over every element of ``p``, applies
    ``out_scalar -= coefficient·H``. When ``row_weights`` is given, each first-axis
    slice contributes its weight. A zero entry contributes
    ``0·log 0 = 0`` and a positive entry contributes ``-p·log p``.
    ``log_buffer`` must match ``p``.

    Args:
        out_scalar: ``()`` loss accumulator, decreased in place.
        p: Probability tensor of any shape, read-only.
        coefficient: Non-negative finite weight.
        log_buffer: Workspace shaped like ``p``.
        scratch_scalar: ``()`` workspace.
        row_weights: Optional ``(p.shape[0],)`` first-axis weights for a matrix ``p``.
    """
    torch.special.entr(p, out=log_buffer)  # -p·log p, zero where p = 0 (p untouched)
    if row_weights is not None:
        log_buffer.mul_(row_weights.unsqueeze(1))
    torch.sum(log_buffer, dim=tuple(range(log_buffer.ndim)), out=scratch_scalar)  # H
    out_scalar.sub_(scratch_scalar, alpha=coefficient)  # out -= coef·H
