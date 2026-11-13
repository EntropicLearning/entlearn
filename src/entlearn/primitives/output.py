"""Head parameter updates, cost accumulation, loss, and propagation."""

from __future__ import annotations

import torch

from entlearn.primitives.distance import weighted_sq_distance_
from entlearn.primitives.normalise import _is_soft, _tiny, floored_log_, normalise_
from entlearn.primitives.softmax import _scale_ratio_, assign_simplex_


def scale_logits_(
    out: torch.Tensor,
    logits: torch.Tensor,
    numerator: float,
    denominator: float,
    scratch_keepdim: torch.Tensor,
) -> None:
    """Write row-centred logits times ``numerator / denominator``.

    ``out`` and ``logits`` have shape ``(T, M)`` and may alias. The ``(T, 1)`` scratch
    receives each row's maximum. The positive finite factors are applied without forming
    an overflowing ratio, so each row's maxima stay exactly zero, ties included. A row
    whose logits are all ``-inf`` is centred to zeros, i.e., uniform.
    """
    torch.amax(logits, dim=1, keepdim=True, out=scratch_keepdim)
    torch.sub(logits, scratch_keepdim, out=out)
    out.nan_to_num_(nan=0.0, neginf=-torch.inf)  # an all -inf row: -inf - (-inf) -> 0
    _scale_ratio_(out, numerator, denominator)


def geometric_probabilities_(
    out: torch.Tensor,
    logits: torch.Tensor,
    delta: float,
    epsilon_P: float,
    scratch_keepdim: torch.Tensor,
    scratch_idx: torch.Tensor,
) -> None:
    """Write the geometric read-out from unscaled ``(T, M)`` logits.

    ``epsilon_P = 0`` chooses the first maximum. A positive temperature gives the soft
    read-out ``softmax(logits · delta / epsilon_P)``, scaled by :func:`scale_logits_`.
    Output may alias logits; all scratch is caller-owned with shape ``(T, 1)``.
    """
    if not _is_soft(epsilon_P):
        torch.argmax(logits, dim=1, keepdim=True, out=scratch_idx)
        out.zero_()
        out.scatter_(1, scratch_idx, 1.0)
        return
    scale_logits_(out, logits, delta, epsilon_P, scratch_keepdim)
    torch.softmax(out, 1, out=out)


def apply_output_gate_(
    out: torch.Tensor,
    source: torch.Tensor,
    effective_output_weights: torch.Tensor | None,
) -> torch.Tensor:
    """Scale each row of ``source`` by its output weight into ``out``, and return the result.

    Unlabelled rows have zero weight, so the scaling also removes them. ``out`` may be
    ``source`` itself, which makes the call the in-place form. A caller whose weights are
    already included in ``source`` passes ``None`` and gets ``source`` back, with ``out``
    untouched.

    Args:
        out: ``(T, N)`` output, modified unless the weights are ``None``.
        source: ``(T, N)`` rows to gate; read-only unless aliased with ``out``.
        effective_output_weights: ``(T,)`` per-instance output·sample weights (``eow``),
            zero on unlabelled rows, or ``None`` when ``source`` already carries them.

    Returns:
        ``out``, or ``source`` itself when ``effective_output_weights`` is ``None``.
    """
    if effective_output_weights is None:
        return source
    return torch.mul(source, effective_output_weights.unsqueeze(1), out=out)


def update_classification_output_(
    theta_out: torch.Tensor,
    log_theta_out: torch.Tensor,
    counts: torch.Tensor,
    scratch_norm: torch.Tensor,
    norm_dim: int = 0,
    pseudocount: float = 0.0,
) -> None:
    """Set a classification head's ``theta_out`` from its co-occurrence counts and its log.

    Normalises the joint counts ``counts = Piᵀ @ (eow ∘ gamma_source)`` along ``norm_dim``,
    then refreshes ``log_theta_out = log(max(θ_out, tiny))`` (:func:`floored_log_`).
    ``pseudocount`` adds a constant ``a`` to every count first, giving
    ``(counts + a)/(slicesum + N·a)`` with ``N = theta_out.shape[norm_dim]``.

    ``norm_dim=0`` normalises columns, so each column is a distribution over the ``M``
    classes. A cluster with no labelled mass takes the uniform ``1/M``, and a wholly
    unlabelled batch makes the whole matrix uniform. ``norm_dim=1`` normalises rows, so
    each row is a distribution over the ``K_source`` clusters; a class with no mass
    takes the uniform ``1/K_source``.

    Args:
        theta_out: ``(M, K_source)`` output, overwritten.
        log_theta_out: ``(M, K_source)`` output, overwritten with the log of ``θ_out``.
        counts: ``(M, K_source)`` labelled co-occurrence counts with the source clusters.
        scratch_norm: Workspace for the slice sums: ``(K_source,)`` at ``norm_dim=0``,
            ``(M,)`` at ``norm_dim=1``.
        norm_dim: Axis that sums to one, ``0`` (columns) or ``1`` (rows).
        pseudocount: Added to every count before normalising; ``0.0`` is a no-op.
    """
    theta_out.copy_(counts)
    if pseudocount != 0.0:
        theta_out.add_(pseudocount)  # Dirichlet pseudocount: counts + a before normalising
    normalise_(theta_out, norm_dim, scratch_norm)
    floored_log_(log_theta_out, theta_out)  # floor + log into log_theta_out


def update_regression_output_(
    Cy: torch.Tensor,
    numerator: torch.Tensor,
    denom: torch.Tensor,
    Y_mean_weighted: torch.Tensor,
    scratch_K: torch.Tensor,
) -> None:
    """Set a regression head's centroids ``Cy`` from its weighted target sums and masses.

    Writes ``Cy[m, k] = numerator[m, k] / denom[k]`` with ``numerator = Yᵀ @ (eow ∘
    gamma_source)`` and ``denom`` its column sums. A cluster with mass at or below
    ``finfo(dtype).tiny`` takes the weighted labelled mean ``Y_mean_weighted[m]``
    instead. Denominators are protected before division, so a wholly unlabelled batch
    makes every column that mean. The sums are read-only, and ``Cy`` must not share
    storage with them, as it briefly holds the protected divisors.

    Args:
        Cy: ``(M, K_source)`` output, overwritten.
        numerator: ``(M, K_source)`` weighted target sums of the clusters.
        denom: ``(K_source,)`` weighted masses of the clusters.
        Y_mean_weighted: ``(M,)`` ``eow``-weighted labelled mean of ``Y``, the
            zero-mass fallback.
        scratch_K: ``(K_source,)`` workspace.
    """
    tiny = _tiny(Cy.dtype)
    Cy.copy_(denom.unsqueeze(0)).clamp_min_(tiny)
    torch.div(numerator, Cy, out=Cy)
    torch.gt(denom, tiny, out=scratch_K)
    Cy.mul_(scratch_K.unsqueeze(0))
    scratch_K.neg_().add_(1.0)  # now 1 for an empty column, else 0
    Cy.addcmul_(Y_mean_weighted.unsqueeze(1), scratch_K.unsqueeze(0))  # empty cols <- mean


def classification_output_accumulate_into_source_cost_(
    cost_source: torch.Tensor,
    log_theta_out: torch.Tensor,
    Pi: torch.Tensor,
    delta: float,
    effective_output_weights: torch.Tensor | None,
    scratch_TM: torch.Tensor,
) -> None:
    """Add the classification output term to the source block's coupling cost.

    Adds ``-delta·eow[t]·Σ_m Pi[t, m]·log_theta_out[m, k_source]`` to
    ``cost_source[t, k_source]``. The caller must zero ``cost_source`` first, and
    ``scratch_TM`` must not share storage with it.

    Args:
        cost_source: ``(T, K_source)`` cost buffer, accumulated into.
        log_theta_out: ``(M, K_source)`` log of the transition matrix.
        Pi: ``(T, M)`` classification target, one-hot or per-instance distribution; its
            unlabelled rows are zero when ``effective_output_weights`` is ``None``.
        delta: Raw output coupling strength, no ``1/T``.
        effective_output_weights: ``(T,)`` per-instance class·sample weights (``eow``),
            zero on unlabelled rows, or ``None`` when ``Pi`` already carries them.
        scratch_TM: ``(T, M)`` workspace.
    """
    gated = apply_output_gate_(scratch_TM, Pi, effective_output_weights)
    cost_source.addmm_(gated, log_theta_out, alpha=-delta)  # += -δ · (eow∘Pi) @ logθ_out


def regression_output_accumulate_into_source_cost_(
    cost_source: torch.Tensor,
    Cy: torch.Tensor,
    Y: torch.Tensor,
    delta: float,
    effective_output_weights: torch.Tensor | None,
    scratch_K: torch.Tensor,
    scratch_KM: torch.Tensor,
    *,
    Wm: torch.Tensor | None = None,
) -> None:
    """Add the regression output term, less its row constant, to the source block's cost.

    Adds ``delta·eow[t]·(‖Cy_k‖²_Wm - 2·⟨Y_t, Cy_k⟩_Wm)`` to ``cost_source[t, k]``: the
    residual ``delta·eow[t]·‖Y_t - Cy_k‖²_Wm`` without ``delta·eow[t]·‖Y_t‖²_Wm``, which
    shifts every cost in a row equally and so leaves the source assignment unchanged.

    Args:
        cost_source: ``(T, K_source)`` cost buffer, accumulated into.
        Cy: ``(M, K_source)`` regression centroids of the head.
        Y: ``(T, M)`` regression target with each row multiplied by its ``eow``, zero on
            unlabelled rows.
        delta: Raw output coupling strength, no ``1/T``.
        effective_output_weights: ``(T,)`` per-instance output·sample weights (``eow``),
            zero on unlabelled rows, or ``None`` when every row weighs one.
        scratch_K: ``(K_source,)`` workspace.
        scratch_KM: ``(K_source, M)`` workspace.
        Wm: ``(M,)`` output-dimension weights, or ``None`` for all 1.
    """
    Cy_t = Cy.transpose(0, 1)
    # The centroid term: scratch_K[k] = ‖Cy_k‖²_Wm, one value per cluster
    torch.mul(Cy_t, Cy_t, out=scratch_KM)
    if Wm is not None:
        scratch_KM.mul_(Wm)
    torch.sum(scratch_KM, dim=1, out=scratch_K)
    # The cross term: cost[t, k] -= 2·delta·⟨Y_t, Cy_k⟩_Wm, where Y already carries eow[t]
    if Wm is not None:
        torch.mul(Cy_t, Wm, out=scratch_KM)  # (Wm ∘ Cy)ᵀ
    cost_source.addmm_(Y, Cy if Wm is None else scratch_KM.transpose(0, 1), alpha=-2.0 * delta)
    # The centroid term, weighted per row: cost[t, k] += delta·eow[t]·‖Cy_k‖²_Wm
    if effective_output_weights is None:
        cost_source.add_(scratch_K, alpha=delta)  # every eow[t] is one: broadcast over the rows
    else:
        cost_source.addr_(effective_output_weights, scratch_K, alpha=delta)  # outer product


def classification_output_loss_(
    out_scalar: torch.Tensor,
    log_theta_out: torch.Tensor,
    Pi: torch.Tensor,
    gamma_source: torch.Tensor,
    delta: float,
    effective_output_weights: torch.Tensor | None,
    scratch_TM: torch.Tensor,
    scratch_TK_source: torch.Tensor,
    scratch_scalar: torch.Tensor,
) -> None:
    """Add a classification head's loss to ``out_scalar``.

    Adds ``-delta·Σ_{t,m,k} eow[t]·Pi[t,m]·gamma_source[t,k]·log_theta_out[m,k]``.

    Args:
        out_scalar: ``()`` loss accumulator, added into.
        log_theta_out: ``(M, K_source)`` log of the head's transition matrix.
        Pi: ``(T, M)`` classification target, one-hot or per-instance distribution; its
            unlabelled rows are zero when ``effective_output_weights`` is ``None``.
        gamma_source: ``(T, K_source)`` source-block affiliations.
        delta: Raw output coupling strength, no ``1/T``.
        effective_output_weights: ``(T,)`` per-instance class·sample weights (``eow``),
            zero on unlabelled rows, or ``None`` when ``Pi`` already carries them.
        scratch_TM: ``(T, M)`` workspace.
        scratch_TK_source: ``(T, K_source)`` workspace.
        scratch_scalar: ``()`` workspace.
    """
    gated = apply_output_gate_(scratch_TM, Pi, effective_output_weights)
    torch.mm(gated, log_theta_out, out=scratch_TK_source)  # S = (eow∘Pi) @ logθ_out
    scratch_TK_source.mul_(gamma_source)  # S ∘ Γ_source
    torch.sum(
        scratch_TK_source, dim=(0, 1), out=scratch_scalar
    )  # full reduction into the scalar scratch
    out_scalar.add_(scratch_scalar, alpha=-delta)  # += -δ · Σ_{t,m,k}


def regression_output_loss_(
    out_scalar: torch.Tensor,
    Cy: torch.Tensor,
    Y: torch.Tensor,
    target_moment: torch.Tensor,
    gamma_source: torch.Tensor,
    delta: float,
    effective_output_weights: torch.Tensor,
    scratch_TK: torch.Tensor,
    scratch_K: torch.Tensor,
    scratch_KM: torch.Tensor,
    scratch_scalar: torch.Tensor,
    *,
    Wm: torch.Tensor | None = None,
) -> None:
    """Add a regression head's loss to ``out_scalar``.

    Adds ``delta·Σ_{t,k} eow[t]·gamma_source[t,k]·‖Y_t - Cy_k‖²_Wm``. Each row of
    ``gamma_source`` sums to one, so the targets' own term ``Σ_t eow[t]·‖Y_t‖²_Wm`` is
    read from ``target_moment`` once instead of per row and cluster.

    Args:
        out_scalar: ``()`` loss accumulator, added into.
        Cy: ``(M, K_source)`` regression centroids of the head.
        Y: ``(T, M)`` regression target.
        target_moment: ``(M,)`` weighted second moments ``Σ_t eow[t]·Y[t, m]²``.
        gamma_source: ``(T, K_source)`` source-block affiliations.
        delta: Raw output coupling strength, no ``1/T``.
        effective_output_weights: ``(T,)`` per-instance output·sample weights (``eow``),
            zero on unlabelled rows.
        scratch_TK: ``(T, K_source)`` workspace.
        scratch_K: ``(K_source,)`` workspace.
        scratch_KM: ``(K_source, M)`` workspace.
        scratch_scalar: ``()`` workspace.
        Wm: ``(M,)`` output-dimension weights, or ``None`` for all 1.
    """
    # ‖Cy_k‖²_Wm - 2⟨Y_t, Cy_k⟩_Wm: not a distance, so it is not clamped at zero
    weighted_sq_distance_(
        scratch_TK, Y, Cy.transpose(0, 1), None, scratch_K, scratch_KM, feature_weights=Wm
    )
    scratch_TK.mul_(gamma_source)
    apply_output_gate_(scratch_TK, scratch_TK, effective_output_weights)
    torch.sum(scratch_TK, dim=(0, 1), out=scratch_scalar)  # full reduction into the scalar scratch
    out_scalar.add_(scratch_scalar, alpha=delta)  # += δ · Σ_{t,k_s}
    if Wm is None:
        torch.sum(target_moment, dim=0, out=scratch_scalar)
    else:
        torch.dot(Wm, target_moment, out=scratch_scalar)
    out_scalar.add_(scratch_scalar, alpha=delta)  # += δ · Σ_t eow[t]·‖Y_t‖²_Wm


def propagate_classification_output_(
    out_TM: torch.Tensor,
    gamma_source: torch.Tensor,
    theta_out: torch.Tensor,
    scratch_T1: torch.Tensor,
) -> None:
    """Write the predicted class distribution ``out_TM = gamma_source @ θ_outᵀ``.

    Every row is then divided by its sum, so it is a distribution over the ``M``
    classes. The row sums are floored at ``finfo(dtype).tiny``, so an all-zero
    row stays zero instead of becoming ``NaN``.

    ``scratch_T1`` must not share storage with ``out_TM``, which the shapes permit at
    ``M == 1``.

    Args:
        out_TM: ``(T, M)`` output, overwritten.
        gamma_source: ``(T, K_source)`` source-block affiliations.
        theta_out: ``(M, K_source)`` left-stochastic transition matrix of the head.
        scratch_T1: ``(T, 1)`` workspace for the row sums.
    """
    tiny = _tiny(out_TM.dtype)
    torch.mm(gamma_source, theta_out.transpose(0, 1), out=out_TM)  # out ← gamma_source @ θ_outᵀ
    torch.sum(out_TM, dim=1, keepdim=True, out=scratch_T1)  # row sums Σ_m out[t,m]
    scratch_T1.clamp_min_(tiny)
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
    Cy: torch.Tensor,
    numerator: torch.Tensor,
    denom: torch.Tensor,
    target_moment: torch.Tensor,
    delta: float,
    epsilon_M: float,
    Wm_cost: torch.Tensor,
    scratch_MK: torch.Tensor,
    scratch_keepdim: torch.Tensor,
    scratch_idx: torch.Tensor,
) -> None:
    """Set a regression head's output-dimension weights ``Wm`` from its weighted target sums.

    Forms the per-dimension weighted residuals
    ``Wm_cost[m] = delta·Σ_t eow[t]·Σ_k gamma_source[t, k]·(Y[t, m] - Cy[m, k])²`` from the
    sums, as ``delta·(target_moment[m] - 2·Σ_k Cy[m, k]·numerator[m, k] +
    Σ_k denom[k]·Cy[m, k]²)``, which holds because each row of the affiliations sums to
    one. It then assigns them onto the simplex with :func:`assign_simplex_` at temperature
    ``epsilon_M``.

    Args:
        Wm: ``(M,)`` output, overwritten.
        Cy: ``(M, K_source)`` regression centroids of the head.
        numerator: ``(M, K_source)`` weighted target sums ``Yᵀ @ (eow ∘ gamma_source)``.
        denom: ``(K_source,)`` weighted cluster masses ``eowᵀ @ gamma_source``.
        target_moment: ``(M,)`` weighted second moments ``Σ_t eow[t]·Y[t, m]²``.
        delta: Raw output coupling strength, not scaled by ``1/T``.
        epsilon_M: Finite non-negative softmax temperature.
        Wm_cost: ``(M,)`` workspace receiving the per-dimension residuals.
        scratch_MK: ``(M, K_source)`` workspace.
        scratch_keepdim: ``(1,)`` workspace for the softmax reduction.
        scratch_idx: ``(1,)`` int64 workspace for the hard-assignment index.
    """
    torch.mul(Cy, denom, out=scratch_MK)  # denom[k]·Cy
    scratch_MK.sub_(numerator, alpha=2.0)
    scratch_MK.mul_(Cy)  # denom[k]·Cy² - 2·Cy·numerator
    torch.sum(scratch_MK, dim=1, out=Wm_cost)
    Wm_cost.add_(target_moment).mul_(delta)
    assign_simplex_(Wm, Wm_cost, epsilon_M, 0, scratch_keepdim, scratch_idx)
