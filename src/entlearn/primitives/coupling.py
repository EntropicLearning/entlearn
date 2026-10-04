"""Cross-block coupling accumulation and partial loss."""

from __future__ import annotations

import torch


def accumulate_coupling_(
    cost: torch.Tensor,
    log_theta: torch.Tensor,
    gamma: torch.Tensor,
    delta_eff: float,
    sample_weights: torch.Tensor | None = None,
    scratch_TK: torch.Tensor | None = None,
) -> None:
    """Add the coupling cost for one end of a transition to ``cost``.

    Adds ``-delta_eff·w[t]·Σ_k gamma[t, k]·log_theta[k, k_own]`` to
    ``cost[t, k_own]``, with ``w[t] = sample_weights[t]`` when given, else 1. The
    caller must zero ``cost`` first.

    The same call serves either end of a transition. The caller picks which by how it
    orients the arguments:

    - Target side: pass ``log_theta.transpose(0, 1)`` and the source's affiliations.
    - Source side: pass ``log_theta`` and the target's affiliations.

    Args:
        cost: ``(T, K_own)`` cost buffer, accumulated into.
        log_theta: ``(K, K_own)`` log of the transition matrix, oriented as explained above.
        gamma: ``(T, K)`` the other block's affiliations.
        delta_eff: Coupling strength, including the ``1/T`` divisor.
        sample_weights: ``(T,)`` per-instance weights, or ``None``.
        scratch_TK: ``(T, K)`` workspace, required iff ``sample_weights`` is given.
    """
    if sample_weights is None:
        cost.addmm_(gamma, log_theta, alpha=-delta_eff)
    else:
        assert scratch_TK is not None, "scratch_TK is required when sample_weights are provided"
        torch.mul(gamma, sample_weights.unsqueeze(1), out=scratch_TK)  # w ∘ gamma
        cost.addmm_(scratch_TK, log_theta, alpha=-delta_eff)


def transition_partial_loss_(
    out_scalar: torch.Tensor,
    log_theta: torch.Tensor,
    gamma_target: torch.Tensor,
    gamma_source: torch.Tensor,
    delta_eff: float,
    scratch_TK_source: torch.Tensor,
    scratch_scalar: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> None:
    """Add a connection's coupling loss to ``out_scalar``.

    Adds ``-delta_eff·Σ_{t,k_target,k_source} w[t]·gamma_target[t,k_target]·
    gamma_source[t,k_source]·log_theta[k_target,k_source]``, with
    ``w[t] = sample_weights[t]`` when given, else 1. The caller must zero
    ``out_scalar`` first.

    Args:
        out_scalar: ``()`` loss accumulator, added into.
        log_theta: ``(K_target, K_source)`` log of the transition matrix.
        gamma_target: ``(T, K_target)`` target-block affiliations.
        gamma_source: ``(T, K_source)`` source-block affiliations.
        delta_eff: Coupling strength, including the ``1/T`` divisor.
        scratch_TK_source: ``(T, K_source)`` workspace.
        scratch_scalar: ``()`` workspace.
        sample_weights: ``(T,)`` per-instance weights, or ``None``.
    """
    torch.mm(gamma_target, log_theta, out=scratch_TK_source)  # P = Γ_target @ logθ, (T, K_source)
    scratch_TK_source.mul_(gamma_source)  # P ∘ Γ_source
    if sample_weights is not None:
        scratch_TK_source.mul_(sample_weights.unsqueeze(1))  # w[t] · (P ∘ Γ_source)
    torch.sum(
        scratch_TK_source, dim=(0, 1), out=scratch_scalar
    )  # full reduction into the () scratch
    out_scalar.add_(scratch_scalar, alpha=-delta_eff)  # += -δ_eff · Σ_{t,k_s}
