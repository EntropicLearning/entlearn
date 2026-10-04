"""Connection transition-matrix update."""

from __future__ import annotations

import torch

from entlearn.primitives.normalise import floored_log_, normalise_


def update_theta_(
    theta: torch.Tensor,
    log_theta: torch.Tensor,
    gamma_target: torch.Tensor,
    gamma_source: torch.Tensor,
    scratch_K_norm: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
    scratch_TK_source: torch.Tensor | None = None,
    pseudocount: float = 0.0,
    norm_dim: int = 0,
) -> None:
    """Set ``theta`` to the normalised co-occurrence of target and source affiliations.

    Computes ``θ ← normalise(Γ_targetᵀ @ Γ_source, norm_dim)`` in place and refreshes
    ``log_theta = log(max(θ, eps))``. ``sample_weights``, if given, weights the source
    affiliations first; ``pseudocount`` adds a constant ``a`` to every count, giving
    ``(counts + a)/(slicesum + N·a)`` with ``N = theta.shape[norm_dim]``.

    ``norm_dim`` picks the axis that sums to one: ``0`` normalises columns,
    ``1`` normalises rows. A slice with no mass becomes uniform.

    Args:
        theta: ``(K_target, K_source)``, overwritten.
        log_theta: ``(K_target, K_source)``, overwritten.
        gamma_target: ``(T, K_target)`` target-block affiliations.
        gamma_source: ``(T, K_source)`` source-block affiliations.
        scratch_K_norm: Workspace for the slice sums: ``(K_source,)`` at ``norm_dim=0``,
            ``(K_target,)`` at ``norm_dim=1``.
        sample_weights: ``(T,)`` per-instance weights, or ``None``.
        scratch_TK_source: ``(T, K_source)`` workspace, required iff ``sample_weights``
            is given.
        pseudocount: Added to every count before normalising; ``0.0`` is a no-op.
        norm_dim: Axis that sums to one after normalising, ``0`` (columns) or ``1`` (rows).
    """
    if sample_weights is None:
        torch.mm(gamma_target.transpose(0, 1), gamma_source, out=theta)  # Γ_targetᵀ @ Γ_source
    else:
        assert scratch_TK_source is not None, (
            "scratch_TK_source is required when sample_weights are given"
        )
        torch.mul(gamma_source, sample_weights.unsqueeze(1), out=scratch_TK_source)  # sw ∘ Γ_source
        torch.mm(gamma_target.transpose(0, 1), scratch_TK_source, out=theta)
    if pseudocount != 0.0:
        theta.add_(pseudocount)  # Dirichlet pseudocount: counts + a before normalising
    normalise_(theta, norm_dim, scratch_K_norm)  # stochastic along the constrained axis
    floored_log_(log_theta, theta)
