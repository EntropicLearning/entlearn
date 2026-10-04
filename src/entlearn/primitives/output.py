"""Head parameter updates, cost accumulation, loss, and propagation."""

from __future__ import annotations

import torch

from entlearn.primitives.centroids import compute_wd_cost_euclidean_
from entlearn.primitives.distance import weighted_sq_distance_
from entlearn.primitives.normalise import _eps, _is_soft, floored_log_, normalise_
from entlearn.primitives.softmax import assign_simplex_


def scale_logits_(
    out: torch.Tensor,
    logits: torch.Tensor,
    beta: float,
    scratch_keepdim: torch.Tensor,
) -> None:
    """Write row-centred logits times ``beta`` without an uncentred scaled intermediate.

    ``out`` and ``logits`` have shape ``(T, M)`` and may alias. The ``(T, 1)``
    scratch receives each row's maximum. Inputs must be finite and ``beta`` positive.
    """
    torch.amax(logits, dim=1, keepdim=True, out=scratch_keepdim)
    torch.sub(logits, scratch_keepdim, out=out)
    out.mul_(beta)
    # A beta beyond the dtype range sends negative gaps to -inf. Exact maxima
    # must remain zero, including ties, rather than becoming 0 * inf = NaN.
    out.nan_to_num_(nan=0.0, neginf=-torch.inf)


def geometric_probabilities_(
    out: torch.Tensor,
    logits: torch.Tensor,
    delta: float,
    epsilon_P: float,
    scratch_keepdim: torch.Tensor,
    scratch_idx: torch.Tensor,
) -> None:
    """Write the geometric read-out from unscaled ``(T, M)`` logits.

    ``epsilon_P <= eps(dtype)`` explicitly chooses the first maximum. Otherwise
    the soft read-out uses the combined inverse temperature ``delta / epsilon_P``.
    Output may alias logits; all scratch is caller-owned with shape ``(T, 1)``.
    """
    if not _is_soft(epsilon_P, out.dtype):
        torch.argmax(logits, dim=1, keepdim=True, out=scratch_idx)
        out.zero_()
        out.scatter_(1, scratch_idx, 1.0)
        return
    scale_logits_(out, logits, delta / epsilon_P, scratch_keepdim)
    out.exp_()
    torch.sum(out, dim=1, keepdim=True, out=scratch_keepdim)
    out.div_(scratch_keepdim)


def apply_output_gate_(
    out: torch.Tensor,
    source: torch.Tensor,
    effective_output_weights: torch.Tensor | None,
    labelled: torch.Tensor,
) -> None:
    """Scale each row of ``source`` by its output weight and zero the unlabelled rows.

    ``out`` may be ``source`` itself. Both operations are element-wise, so the aliased
    call is the in-place form. A caller whose weights are already included into ``source``
    passes ``None`` and gets only the mask.

    Args:
        out: ``(T, N)`` output (modified).
        source: ``(T, N)`` rows to gate; read-only unless aliased with ``out``.
        effective_output_weights: ``(T,)`` per-instance output·sample weights (``eow``),
            or ``None`` when ``source`` already carries them.
        labelled: ``(T,)`` boolean mask of labelled instances.
    """
    if effective_output_weights is None:
        torch.mul(source, labelled.unsqueeze(1), out=out)
        return
    torch.mul(source, effective_output_weights.unsqueeze(1), out=out)
    out.mul_(labelled.unsqueeze(1))


def update_classification_output_(
    theta_out: torch.Tensor,
    log_theta_out: torch.Tensor,
    Pi: torch.Tensor,
    gamma_source: torch.Tensor,
    effective_output_weights: torch.Tensor | None,
    labelled: torch.Tensor,
    scratch_norm: torch.Tensor,
    scratch_TK_source: torch.Tensor,
    norm_dim: int = 0,
) -> None:
    """Set a classification head's ``theta_out`` and refresh its log cache.

    Normalises the joint counts ``Piᵀ @ (labelled·eow ∘ gamma_source)`` along
    ``norm_dim``, then refreshes ``log_theta_out = log(max(θ_out, eps))``.

    ``norm_dim=0`` normalises columns, so each column is a distribution over the ``M``
    classes. A cluster with no labelled mass takes the uniform ``1/M``, and a wholly
    unlabelled batch makes the whole matrix uniform. ``norm_dim=1`` normalises rows, so
    each row is a distribution over the ``K_source`` clusters; a class with no mass
    takes the uniform ``1/K_source``.

    Args:
        theta_out: ``(M, K_source)`` output, overwritten.
        log_theta_out: ``(M, K_source)`` output, overwritten with the floored log.
        Pi: ``(T, M)`` classification target, one-hot or per-instance distribution.
        gamma_source: ``(T, K_source)`` source-block affiliations.
        effective_output_weights: ``(T,)`` per-instance class·sample weights (``eow``),
            or ``None`` when ``Pi`` already carries them.
        labelled: ``(T,)`` boolean mask of labelled instances.
        scratch_norm: Workspace for the slice sums: ``(K_source,)`` at ``norm_dim=0``,
            ``(M,)`` at ``norm_dim=1``.
        scratch_TK_source: ``(T, K_source)`` workspace.
        norm_dim: Axis that sums to one, ``0`` (columns) or ``1`` (rows).
    """
    apply_output_gate_(scratch_TK_source, gamma_source, effective_output_weights, labelled)
    torch.mm(Pi.transpose(0, 1), scratch_TK_source, out=theta_out)
    normalise_(theta_out, norm_dim, scratch_norm)
    floored_log_(log_theta_out, theta_out)  # floor + log into log_theta_out


def update_regression_output_(
    Cy: torch.Tensor,
    Y: torch.Tensor,
    gamma_source: torch.Tensor,
    effective_output_weights: torch.Tensor,
    labelled: torch.Tensor,
    Y_mean_weighted: torch.Tensor,
    denom_buf: torch.Tensor,
    scratch_MK: torch.Tensor,
    scratch_TK_source: torch.Tensor,
) -> None:
    """Set a regression head's centroids ``Cy``.

    Writes ``Cy[m, k] = numer[m, k] / denom[k]`` with
    ``numer = Yᵀ @ (labelled·eow ∘ gamma_source)`` and ``denom`` its column sums. A
    cluster with mass at or below the dtype machine precision takes the weighted
    labelled mean ``Y_mean_weighted[m]`` instead. Denominators are protected before
    division, a wholly unlabelled batch therefore makes every column that mean.

    ``scratch_TK_source``, ``scratch_MK`` and ``Cy`` must be distinct buffers. ``Cy``
    briefly holds the protected divisors.

    Args:
        Cy: ``(M, K_source)`` output, overwritten.
        Y: ``(T, M)`` regression target.
        gamma_source: ``(T, K_source)`` source-block affiliations.
        effective_output_weights: ``(T,)`` per-instance output·sample weights (``eow``).
        labelled: ``(T,)`` boolean mask of labelled instances.
        Y_mean_weighted: ``(M,)`` ``eow``-weighted labelled mean of ``Y``, the
            zero-mass fallback.
        denom_buf: ``(K_source,)`` workspace.
        scratch_MK: ``(M, K_source)`` workspace.
        scratch_TK_source: ``(T, K_source)`` workspace.
    """
    # mask + weight the source affiliations
    apply_output_gate_(scratch_TK_source, gamma_source, effective_output_weights, labelled)
    torch.mm(Y.transpose(0, 1), scratch_TK_source, out=scratch_MK)  # numer = Yᵀ @ masked weights
    torch.sum(scratch_TK_source, dim=0, out=denom_buf)  # denom[k]
    Cy.copy_(denom_buf.unsqueeze(0))
    Cy.clamp_min_(_eps(Cy.dtype))
    torch.div(scratch_MK, Cy, out=Cy)
    torch.gt(denom_buf, _eps(Cy.dtype), out=denom_buf)
    Cy.mul_(denom_buf.unsqueeze(0))
    denom_buf.neg_().add_(1.0)  # now 1 for an empty column, else 0
    Cy.addcmul_(
        Y_mean_weighted.unsqueeze(1), denom_buf.unsqueeze(0)
    )  # empty cols <- Y_mean_weighted


def classification_output_accumulate_into_source_cost_(
    cost_source: torch.Tensor,
    log_theta_out: torch.Tensor,
    Pi: torch.Tensor,
    delta: float,
    effective_output_weights: torch.Tensor | None,
    labelled: torch.Tensor,
    scratch_TM: torch.Tensor,
) -> None:
    """Add the classification output term to the source block's coupling cost.

    Adds ``-delta·labelled[t]·eow[t]·Σ_m Pi[t, m]·log_theta_out[m, k_source]`` to
    ``cost_source[t, k_source]``. The caller must zero ``cost_source`` first, and
    ``scratch_TM`` must not share storage with it.

    Args:
        cost_source: ``(T, K_source)`` cost buffer, accumulated into.
        log_theta_out: ``(M, K_source)`` log of the transition matrix.
        Pi: ``(T, M)`` classification target, one-hot or per-instance distribution.
        delta: Raw output coupling strength, no ``1/T``.
        effective_output_weights: ``(T,)`` per-instance class·sample weights (``eow``),
            or ``None`` when ``Pi`` already carries them.
        labelled: ``(T,)`` boolean mask of labelled instances.
        scratch_TM: ``(T, M)`` workspace.
    """
    apply_output_gate_(scratch_TM, Pi, effective_output_weights, labelled)
    cost_source.addmm_(
        scratch_TM, log_theta_out, alpha=-delta
    )  # += -δ · (masked eow∘Pi) @ logθ_out


def _regression_output_residual_(
    scratch_TK: torch.Tensor,
    Cy: torch.Tensor,
    Y: torch.Tensor,
    target_sq: torch.Tensor,
    effective_output_weights: torch.Tensor,
    labelled: torch.Tensor,
    scratch_K: torch.Tensor,
    scratch_KD: torch.Tensor,
    *,
    gamma_source: torch.Tensor | None = None,
    Wm: torch.Tensor | None = None,
) -> None:
    """Write the regression head's weighted squared residual into ``scratch_TK``.

    Computes ``‖Y - Cy‖²``, ``Wm``-weighted per dimension when ``Wm`` is given, clamps
    it at zero, multiplies by ``gamma_source`` when given, then gates the rows through
    :func:`apply_output_gate_`.
    """
    weighted_sq_distance_(
        scratch_TK, Y, Cy.transpose(0, 1), target_sq, scratch_K, scratch_KD, feature_weights=Wm
    )
    scratch_TK.clamp_min_(0.0)  # clamp negative round-off residuals
    if gamma_source is not None:
        scratch_TK.mul_(gamma_source)  # Γ_source ∘ ‖Y-Cy‖² (loss path only)
    apply_output_gate_(scratch_TK, scratch_TK, effective_output_weights, labelled)


def regression_output_accumulate_into_source_cost_(
    cost_source: torch.Tensor,
    Cy: torch.Tensor,
    Y: torch.Tensor,
    target_sq: torch.Tensor,
    delta: float,
    effective_output_weights: torch.Tensor,
    labelled: torch.Tensor,
    scratch_TK: torch.Tensor,
    scratch_K: torch.Tensor,
    scratch_KD: torch.Tensor,
    *,
    Wm: torch.Tensor | None = None,
) -> None:
    """Add the regression output term to the source block's coupling cost.

    Adds ``delta·labelled[t]·eow[t]·‖Y_t - Cy_k‖²_Wm`` to ``cost_source[t, k]``.
    Both ``scratch_TK`` and ``scratch_KD`` must be disjoint from ``cost_source`` so
    already accumulated coupling costs remain intact.

    Args:
        cost_source: ``(T, K_source)`` cost buffer, accumulated into.
        Cy: ``(M, K_source)`` regression centroids of the head.
        Y: ``(T, M)`` regression target.
        target_sq: ``(T,)`` precomputed ``Σ_m Y[t, m]²``, or ``Σ_m Wm[m]·Y[t, m]²``
            when ``Wm`` is given.
        delta: Raw output coupling strength, no ``1/T``.
        effective_output_weights: ``(T,)`` per-instance output·sample weights (``eow``).
        labelled: ``(T,)`` boolean mask of labelled instances.
        scratch_TK: ``(T, K_source)`` workspace.
        scratch_K: ``(K_source,)`` workspace.
        scratch_KD: ``(K_source, M)`` workspace.
        Wm: ``(M,)`` output-dimension weights, or ``None`` for all 1.
    """
    _regression_output_residual_(
        scratch_TK,
        Cy,
        Y,
        target_sq,
        effective_output_weights,
        labelled,
        scratch_K,
        scratch_KD,
        Wm=Wm,
    )
    cost_source.add_(scratch_TK, alpha=delta)  # cost_source += δ · (masked eow ∘ ‖Y-Cy‖²)


def classification_output_loss_(
    out_scalar: torch.Tensor,
    log_theta_out: torch.Tensor,
    Pi: torch.Tensor,
    gamma_source: torch.Tensor,
    delta: float,
    effective_output_weights: torch.Tensor | None,
    labelled: torch.Tensor,
    scratch_TM: torch.Tensor,
    scratch_TK_source: torch.Tensor,
    scratch_scalar: torch.Tensor,
) -> None:
    """Add a classification head's loss to ``out_scalar``.

    Adds ``-delta·Σ_{t,m,k} labelled[t]·eow[t]·Pi[t,m]·gamma_source[t,k]·
    log_theta_out[m,k]``.

    Args:
        out_scalar: ``()`` loss accumulator, added into.
        log_theta_out: ``(M, K_source)`` log of the head's transition matrix.
        Pi: ``(T, M)`` classification target, one-hot or per-instance distribution.
        gamma_source: ``(T, K_source)`` source-block affiliations.
        delta: Raw output coupling strength, no ``1/T``.
        effective_output_weights: ``(T,)`` per-instance class·sample weights (``eow``),
            or ``None`` when ``Pi`` already carries them.
        labelled: ``(T,)`` boolean mask of labelled instances.
        scratch_TM: ``(T, M)`` workspace.
        scratch_TK_source: ``(T, K_source)`` workspace.
        scratch_scalar: ``()`` workspace.
    """
    apply_output_gate_(scratch_TM, Pi, effective_output_weights, labelled)
    torch.mm(scratch_TM, log_theta_out, out=scratch_TK_source)  # S = (masked eow∘Pi) @ logθ_out
    scratch_TK_source.mul_(gamma_source)  # S ∘ Γ_source
    torch.sum(
        scratch_TK_source, dim=(0, 1), out=scratch_scalar
    )  # full reduction into the scalar scratch
    out_scalar.add_(scratch_scalar, alpha=-delta)  # += -δ · Σ_{t,m,k}


def regression_output_loss_(
    out_scalar: torch.Tensor,
    Cy: torch.Tensor,
    Y: torch.Tensor,
    target_sq: torch.Tensor,
    gamma_source: torch.Tensor,
    delta: float,
    effective_output_weights: torch.Tensor,
    labelled: torch.Tensor,
    scratch_TK: torch.Tensor,
    scratch_K: torch.Tensor,
    scratch_KD: torch.Tensor,
    scratch_scalar: torch.Tensor,
    *,
    Wm: torch.Tensor | None = None,
) -> None:
    """Add a regression head's loss to ``out_scalar``.

    Adds ``delta·Σ_{t,k} labelled[t]·eow[t]·gamma_source[t,k]·‖Y_t - Cy_k‖²_Wm``.

    Args:
        out_scalar: ``()`` loss accumulator, added into.
        Cy: ``(M, K_source)`` regression centroids of the head.
        Y: ``(T, M)`` regression target.
        target_sq: ``(T,)`` precomputed ``Σ_m Y[t, m]²``, or ``Σ_m Wm[m]·Y[t, m]²``
            when ``Wm`` is given.
        gamma_source: ``(T, K_source)`` source-block affiliations.
        delta: Raw output coupling strength, no ``1/T``.
        effective_output_weights: ``(T,)`` per-instance output·sample weights (``eow``).
        labelled: ``(T,)`` boolean mask of labelled instances.
        scratch_TK: ``(T, K_source)`` workspace.
        scratch_K: ``(K_source,)`` workspace.
        scratch_KD: ``(K_source, M)`` workspace.
        scratch_scalar: ``()`` workspace.
        Wm: ``(M,)`` output-dimension weights, or ``None`` for all 1.
    """
    _regression_output_residual_(
        scratch_TK,
        Cy,
        Y,
        target_sq,
        effective_output_weights,
        labelled,
        scratch_K,
        scratch_KD,
        gamma_source=gamma_source,
        Wm=Wm,
    )
    torch.sum(scratch_TK, dim=(0, 1), out=scratch_scalar)  # full reduction into the scalar scratch
    out_scalar.add_(scratch_scalar, alpha=delta)  # += δ · Σ_{t,k_s}


def propagate_classification_output_(
    out_TM: torch.Tensor,
    gamma_source: torch.Tensor,
    theta_out: torch.Tensor,
    scratch_T1: torch.Tensor,
) -> None:
    """Write the predicted class distribution ``out_TM = gamma_source @ θ_outᵀ``.

    Every row is then divided by its sum, so it is a distribution over the ``M``
    classes. The row sums are floored at the dtype machine precision, so an all-zero
    row stays zero instead of becoming ``NaN``.

    ``scratch_T1`` must not share storage with ``out_TM``, which the shapes permit at
    ``M == 1``.

    Args:
        out_TM: ``(T, M)`` output, overwritten.
        gamma_source: ``(T, K_source)`` source-block affiliations.
        theta_out: ``(M, K_source)`` left-stochastic transition matrix of the head.
        scratch_T1: ``(T, 1)`` workspace for the row sums.
    """
    eps = _eps(out_TM.dtype)
    torch.mm(gamma_source, theta_out.transpose(0, 1), out=out_TM)  # out ← gamma_source @ θ_outᵀ
    torch.sum(out_TM, dim=1, keepdim=True, out=scratch_T1)  # row sums Σ_m out[t,m]
    scratch_T1.clamp_min_(eps)
    out_TM.div_(scratch_T1)


def propagate_regression_output_(
    out_TM: torch.Tensor,
    gamma_source: torch.Tensor,
    Cy: torch.Tensor,
) -> None:
    """Write the predicted targets ``out_TM = gamma_source @ Cyᵀ``.

    Args:
        out_TM: ``(T, M)`` output, overwritten.
        gamma_source: ``(T, K_source)`` source-block affiliations.
        Cy: ``(M, K_source)`` regression centroids of the head.
    """
    torch.mm(gamma_source, Cy.transpose(0, 1), out=out_TM)  # out ← gamma_source @ Cyᵀ


def update_output_weights_(
    Wm: torch.Tensor,
    Y: torch.Tensor,
    Y_sq: torch.Tensor,
    Cy: torch.Tensor,
    gamma_source: torch.Tensor,
    effective_output_weights: torch.Tensor,
    labelled: torch.Tensor,
    delta: float,
    epsilon_M: float,
    Wm_cost: torch.Tensor,
    masked_TK: torch.Tensor,
    scratch_MK: torch.Tensor,
    scratch_K: torch.Tensor,
    scratch_T: torch.Tensor,
    scratch_keepdim: torch.Tensor,
    scratch_idx: torch.Tensor,
    scratch_KM: torch.Tensor,
    scratch_M: torch.Tensor,
) -> None:
    """Set a regression head's output-dimension weights ``Wm``.

    Forms the per-dimension weighted residuals
    ``Wm_cost[m] = delta·Σ_t labelled[t]·eow[t]·Σ_k gamma_source[t, k]·
    (Y[t, m] - Cy[m, k])²``, then assigns them onto the simplex with
    :func:`assign_simplex_` at temperature ``epsilon_M``.

    ``masked_TK``, ``scratch_MK`` and ``scratch_KM`` must be distinct buffers.

    Args:
        Wm: ``(M,)`` output, overwritten.
        Y: ``(T, M)`` regression target.
        Y_sq: ``(T, M)`` element-wise ``Y²`` cache.
        Cy: ``(M, K_source)`` regression centroids of the head.
        gamma_source: ``(T, K_source)`` source-block affiliations.
        effective_output_weights: ``(T,)`` per-instance output·sample weights (``eow``).
        labelled: ``(T,)`` boolean mask of labelled instances.
        delta: Raw output coupling strength, not scaled by ``1/T``.
        epsilon_M: Finite non-negative softmax temperature.
        Wm_cost: ``(M,)`` workspace receiving the per-dimension residuals.
        masked_TK: ``(T, K_source)`` workspace.
        scratch_MK: ``(M, K_source)`` workspace.
        scratch_K: ``(K_source,)`` workspace.
        scratch_T: ``(T,)`` workspace.
        scratch_keepdim: ``(1,)`` workspace for the softmax reduction.
        scratch_idx: ``(1,)`` int64 workspace for the hard-assignment index.
        scratch_KM: ``(K_source, M)`` workspace for the Euclidean cost expansion.
        scratch_M: ``(M,)`` workspace for the per-dimension reductions.
    """
    apply_output_gate_(masked_TK, gamma_source, effective_output_weights, labelled)
    torch.sum(masked_TK, dim=0, out=scratch_K)
    compute_wd_cost_euclidean_(
        Wm_cost,
        Y,
        Cy.transpose(0, 1),
        masked_TK,
        scratch_K,
        Y_sq,
        scratch_MK.transpose(0, 1),
        scratch_T,
        scratch_KM,
        scratch_M,
    )
    Wm_cost.mul_(delta)
    assign_simplex_(Wm, Wm_cost, epsilon_M, 0, scratch_keepdim, scratch_idx)
