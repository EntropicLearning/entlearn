# Primitives

State-free tensor operations for affiliations, centroids, coupling terms, dissimilarities, geometry, normalisation, outputs, reductions, statistics and transitions.

## entlearn.primitives

State-free tensor primitives, re-exported flat as the curated public surface.

### accumulate_coupling\_

```
accumulate_coupling_(
    cost,
    log_theta,
    gamma,
    delta_eff,
    sample_weights=None,
    scratch_TK=None,
)
```

Add the coupling cost for one end of a transition to `cost`.

Adds `-delta_eff·w[t]·Σ_k gamma[t, k]·log_theta[k, k_own]` to `cost[t, k_own]`, with `w[t] = sample_weights[t]` when given, else 1. The caller must zero `cost` first.

The same call serves either end of a transition. The caller picks which by how it orients the arguments:

- Target side: pass `log_theta.transpose(0, 1)` and the source's affiliations.
- Source side: pass `log_theta` and the target's affiliations.

Parameters:

| Name             | Type     | Description                                                           | Default                                                 |
| ---------------- | -------- | --------------------------------------------------------------------- | ------------------------------------------------------- |
| `cost`           | `Tensor` | (T, K_own) cost buffer, accumulated into.                             | *required*                                              |
| `log_theta`      | `Tensor` | (K, K_own) log of the transition matrix, oriented as explained above. | *required*                                              |
| `gamma`          | `Tensor` | (T, K) the other block's affiliations.                                | *required*                                              |
| `delta_eff`      | `float`  | Coupling strength, including the 1/T divisor.                         | *required*                                              |
| `sample_weights` | \`Tensor | None\`                                                                | (T,) per-instance weights, or None.                     |
| `scratch_TK`     | \`Tensor | None\`                                                                | (T, K) workspace, required iff sample_weights is given. |

Source code in `src/entlearn/primitives/coupling.py`

```
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
```

### apply_output_gate\_

```
apply_output_gate_(
    out, source, effective_output_weights, labelled
)
```

Scale each row of `source` by its output weight and zero the unlabelled rows.

`out` may be `source` itself. Both operations are element-wise, so the aliased call is the in-place form. A caller whose weights are already included into `source` passes `None` and gets only the mask.

Parameters:

| Name                       | Type     | Description                                             | Default                                                                                  |
| -------------------------- | -------- | ------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `out`                      | `Tensor` | (T, N) output (modified).                               | *required*                                                                               |
| `source`                   | `Tensor` | (T, N) rows to gate; read-only unless aliased with out. | *required*                                                                               |
| `effective_output_weights` | \`Tensor | None\`                                                  | (T,) per-instance output·sample weights (eow), or None when source already carries them. |
| `labelled`                 | `Tensor` | (T,) boolean mask of labelled instances.                | *required*                                                                               |

Source code in `src/entlearn/primitives/output.py`

```
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
```

### argmin_assign\_

```
argmin_assign_(out, cost, dim, scratch_idx)
```

Write a one-hot at the minimum of `cost` along `dim` into `out`.

The `temp -> 0` limit of :func:`softmax_with_temp_`. Ties go to the first index.

Parameters:

| Name          | Type     | Description                                    | Default    |
| ------------- | -------- | ---------------------------------------------- | ---------- |
| `out`         | `Tensor` | (..., N) output, overwritten.                  | *required* |
| `cost`        | `Tensor` | (..., N) cost tensor, read-only.               | *required* |
| `dim`         | `int`    | Axis to take the argmin over.                  | *required* |
| `scratch_idx` | `Tensor` | (..., 1) int64 workspace for the argmin index. | *required* |

Source code in `src/entlearn/primitives/softmax.py`

```
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
```

### assemble_categorical_cost\_

```
assemble_categorical_cost_(
    disc_cost,
    cat_cost,
    xent_scratch,
    X_cat_i,
    logC_cat_i,
    weight_scale,
    Wt,
)
```

Add one categorical feature's cross-entropy error to the two cost caches.

Computes `xent[t, k] = -Σ_m X̃[t, m]·logC[k, m]`, or the gather `-logC[k, code_t]` when `X_cat_i` holds integer codes, then adds::

```
cat_cost[t, k]  += weight_scale · xent[t, k]
disc_cost[t, k] += weight_scale · Wt[t] · xent[t, k]
```

The caller must zero both caches first.

Parameters:

| Name           | Type     | Description                                                               | Default                                                                                                                          |
| -------------- | -------- | ------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `disc_cost`    | \`Tensor | None\`                                                                    | (T, K) Wt-weighted discretisation error, accumulated into. None skips it and leaves only cat_cost.                               |
| `cat_cost`     | `Tensor` | (T, K) unweighted cross-entropy summed across features, accumulated into. | *required*                                                                                                                       |
| `xent_scratch` | `Tensor` | (T, K) workspace for this feature's cross-entropy.                        | *required*                                                                                                                       |
| `X_cat_i`      | `Tensor` | (T,) integer codes or (T, M_d) per-instance distribution.                 | *required*                                                                                                                       |
| `logC_cat_i`   | `Tensor` | (K, M_d) log of this feature's categorical centroids.                     | *required*                                                                                                                       |
| `weight_scale` | \`float  | Tensor\`                                                                  | The scalar δ_cat · s_i · Wd[D_cont + i] for this feature. A zero-dimensional tensor keeps a device-resident scale on the device. |
| `Wt`           | \`Tensor | None\`                                                                    | (T,) per-instance weights. Required only when disc_cost is given.                                                                |

Source code in `src/entlearn/primitives/distance.py`

```
def assemble_categorical_cost_(
    disc_cost: torch.Tensor | None,
    cat_cost: torch.Tensor,
    xent_scratch: torch.Tensor,
    X_cat_i: torch.Tensor,
    logC_cat_i: torch.Tensor,
    weight_scale: float | torch.Tensor,
    Wt: torch.Tensor | None,
) -> None:
    """Add one categorical feature's cross-entropy error to the two cost caches.

    Computes ``xent[t, k] = -Σ_m X̃[t, m]·logC[k, m]``, or the gather
    ``-logC[k, code_t]`` when ``X_cat_i`` holds integer codes, then adds::

        cat_cost[t, k]  += weight_scale · xent[t, k]
        disc_cost[t, k] += weight_scale · Wt[t] · xent[t, k]

    The caller must zero both caches first.

    Args:
        disc_cost: ``(T, K)`` Wt-weighted discretisation error, accumulated into.
            ``None`` skips it and leaves only ``cat_cost``.
        cat_cost: ``(T, K)`` unweighted cross-entropy summed across features,
            accumulated into.
        xent_scratch: ``(T, K)`` workspace for this feature's cross-entropy.
        X_cat_i: ``(T,)`` integer codes or ``(T, M_d)`` per-instance distribution.
        logC_cat_i: ``(K, M_d)`` log of this feature's categorical centroids.
        weight_scale: The scalar ``δ_cat · s_i · Wd[D_cont + i]`` for this feature.
            A zero-dimensional tensor keeps a device-resident scale on the device.
        Wt: ``(T,)`` per-instance weights. Required only when ``disc_cost`` is given.
    """
    if X_cat_i.dim() == 1:  # integer codes: gather column code_t of logC
        torch.index_select(logC_cat_i.transpose(0, 1), 0, X_cat_i, out=xent_scratch)
    else:  # (T, M_d) distribution
        torch.mm(X_cat_i, logC_cat_i.transpose(0, 1), out=xent_scratch)
    xent_scratch.neg_().mul_(weight_scale)  # ws · cross-entropy
    cat_cost.add_(xent_scratch)  # cat_cost += ws · xent
    if disc_cost is not None:
        assert Wt is not None  # required alongside disc_cost
        disc_cost.addcmul_(xent_scratch, Wt.unsqueeze(1))  # += ws·Wt·xent
```

### assemble_euclidean_cost\_

```
assemble_euclidean_cost_(
    cost,
    sqdist_wd,
    X,
    C,
    Wd,
    Wt,
    X_sq_wd_sum,
    scratch_KD,
    scratch_K,
)
```

Refresh the distance cache `sqdist_wd` and add its weighted form to `cost`.

Writes `sqdist_wd[t, k] = Σ_d Wd[d]·(X[t, d] - C[k, d])²`, then adds `Wt[t]·sqdist_wd[t, k]` to `cost[t, k]`. The caller must zero `cost` first.

Parameters:

| Name          | Type     | Description                                    | Default    |
| ------------- | -------- | ---------------------------------------------- | ---------- |
| `cost`        | `Tensor` | (T, K) discretisation error, accumulated into. | *required* |
| `sqdist_wd`   | `Tensor` | (T, K) distance cache, overwritten.            | *required* |
| `X`           | `Tensor` | (T, D) continuous data.                        | *required* |
| `C`           | `Tensor` | (K, D) continuous centroids.                   | *required* |
| `Wd`          | `Tensor` | (D,) continuous feature weights.               | *required* |
| `Wt`          | `Tensor` | (T,) per-instance weights.                     | *required* |
| `X_sq_wd_sum` | `Tensor` | (T,) precomputed Σ_d Wd[d]·X[t, d]².           | *required* |
| `scratch_KD`  | `Tensor` | (K, D) workspace.                              | *required* |
| `scratch_K`   | `Tensor` | (K,) workspace.                                | *required* |

Source code in `src/entlearn/primitives/distance.py`

```
def assemble_euclidean_cost_(
    cost: torch.Tensor,
    sqdist_wd: torch.Tensor,
    X: torch.Tensor,
    C: torch.Tensor,
    Wd: torch.Tensor,
    Wt: torch.Tensor,
    X_sq_wd_sum: torch.Tensor,
    scratch_KD: torch.Tensor,
    scratch_K: torch.Tensor,
) -> None:
    """Refresh the distance cache ``sqdist_wd`` and add its weighted form to ``cost``.

    Writes ``sqdist_wd[t, k] = Σ_d Wd[d]·(X[t, d] - C[k, d])²``, then adds
    ``Wt[t]·sqdist_wd[t, k]`` to ``cost[t, k]``. The caller must zero ``cost`` first.

    Args:
        cost: ``(T, K)`` discretisation error, accumulated into.
        sqdist_wd: ``(T, K)`` distance cache, overwritten.
        X: ``(T, D)`` continuous data.
        C: ``(K, D)`` continuous centroids.
        Wd: ``(D,)`` continuous feature weights.
        Wt: ``(T,)`` per-instance weights.
        X_sq_wd_sum: ``(T,)`` precomputed ``Σ_d Wd[d]·X[t, d]²``.
        scratch_KD: ``(K, D)`` workspace.
        scratch_K: ``(K,)`` workspace.
    """
    weighted_sq_distance_(sqdist_wd, X, C, X_sq_wd_sum, scratch_K, scratch_KD, feature_weights=Wd)
    cost.addcmul_(sqdist_wd, Wt.unsqueeze(1))  # cost += Wt[:, None] · sqdist_wd
```

### assemble_metrised_cost\_

```
assemble_metrised_cost_(
    sqdist,
    X,
    mu,
    T_proj,
    alpha,
    k,
    *,
    RHS,
    combo,
    proj_sq,
    b_all,
    mu_sq,
    X_sq_sum,
)
```

Write the metrised distance `g[t,k]` into `sqdist`.

Computes `g[t,k] = (1+alpha)·‖x_t-mu_k‖² - ‖T_kᵀ(x_t-mu_k)‖²` for the first `k` clusters. One fused matmul `combo = X @ [muᵀ | stacked projectors]` yields both the inner product `⟨x_t, mu_k⟩` and the projected coordinates `T_kᵀ x_t`, and the Euclidean term comes out of `combo` by `‖x-mu‖² = ‖x‖² - 2⟨x,mu⟩ + ‖mu‖²`.

Parameters:

| Name       | Type     | Description                                                                   | Default                               |
| ---------- | -------- | ----------------------------------------------------------------------------- | ------------------------------------- |
| `sqdist`   | `Tensor` | (T, k) output, overwritten with the metrised distances.                       | *required*                            |
| `X`        | `Tensor` | (T, D) continuous data.                                                       | *required*                            |
| `mu`       | `Tensor` | (K_max, D) centroids; only the first k rows are read.                         | *required*                            |
| `T_proj`   | `Tensor` | (K_max, D, d) orthonormal projectors; only the first k read.                  | *required*                            |
| `alpha`    | `float`  | scalar metric anisotropy ≥ 0.                                                 | *required*                            |
| `k`        | `int`    | number of active clusters (≤ K_max); pins the k-compact slicing.              | *required*                            |
| `RHS`      | `Tensor` | (D, K_max + K_max·d) scratch; columns 0:k+k·d receive the k-compact \[muᵀ(:k) | stacked T_proj(:k)\] right-hand side. |
| `combo`    | `Tensor` | (T, K_max + K_max·d) scratch; columns 0:k+k·d receive the fused GEMM output.  | *required*                            |
| `proj_sq`  | `Tensor` | (T, K_max) scratch; columns 0:k receive ‖T_kᵀ(x-mu)‖².                        | *required*                            |
| `b_all`    | `Tensor` | (K_max, d) scratch; rows 0:k receive T_kᵀ mu_k.                               | *required*                            |
| `mu_sq`    | `Tensor` | (K_max,) scratch; entries 0:k receive ‖mu_k‖².                                | *required*                            |
| `X_sq_sum` | `Tensor` | (T,) precomputed Σ_d X[t,d]².                                                 | *required*                            |

Source code in `src/entlearn/primitives/manifold.py`

```
def assemble_metrised_cost_(
    sqdist: torch.Tensor,
    X: torch.Tensor,
    mu: torch.Tensor,
    T_proj: torch.Tensor,
    alpha: float,
    k: int,
    *,
    RHS: torch.Tensor,
    combo: torch.Tensor,
    proj_sq: torch.Tensor,
    b_all: torch.Tensor,
    mu_sq: torch.Tensor,
    X_sq_sum: torch.Tensor,
) -> None:
    """Write the metrised distance ``g[t,k]`` into ``sqdist``.

    Computes ``g[t,k] = (1+alpha)·‖x_t-mu_k‖² - ‖T_kᵀ(x_t-mu_k)‖²`` for the first ``k``
    clusters. One fused matmul ``combo = X @ [muᵀ | stacked projectors]`` yields both
    the inner product ``⟨x_t, mu_k⟩`` and the projected coordinates ``T_kᵀ x_t``, and
    the Euclidean term comes out of ``combo`` by
    ``‖x-mu‖² = ‖x‖² - 2⟨x,mu⟩ + ‖mu‖²``.

    Args:
        sqdist: ``(T, k)`` output, overwritten with the metrised distances.
        X: ``(T, D)`` continuous data.
        mu: ``(K_max, D)`` centroids; only the first ``k`` rows are read.
        T_proj: ``(K_max, D, d)`` orthonormal projectors; only the first ``k`` read.
        alpha: scalar metric anisotropy ``≥ 0``.
        k: number of active clusters (``≤ K_max``); pins the k-compact slicing.
        RHS: ``(D, K_max + K_max·d)`` scratch; columns ``0:k+k·d`` receive the
            k-compact ``[muᵀ(:k) | stacked T_proj(:k)]`` right-hand side.
        combo: ``(T, K_max + K_max·d)`` scratch; columns ``0:k+k·d`` receive the
            fused GEMM output.
        proj_sq: ``(T, K_max)`` scratch; columns ``0:k`` receive ``‖T_kᵀ(x-mu)‖²``.
        b_all: ``(K_max, d)`` scratch; rows ``0:k`` receive ``T_kᵀ mu_k``.
        mu_sq: ``(K_max,)`` scratch; entries ``0:k`` receive ``‖mu_k‖²``.
        X_sq_sum: ``(T,)`` precomputed ``Σ_d X[t,d]²``.
    """
    D = X.shape[1]
    d = T_proj.shape[2]
    kd = k * d

    # k-compact RHS: [muᵀ(:k) | stacked projectors(:k)] in columns 0:k+k·d
    RHS[:, :k].copy_(mu[:k].t())  # muᵀ into the first k columns
    # 3-D destination copy of the permuted projectors.
    RHS[:, k : k + kd].view(D, k, d).copy_(T_proj[:k].permute(1, 0, 2))

    # Fused GEMM: combo[:, :k]=⟨x,mu_k⟩, combo[:, k:k+kd]=T_kᵀx (flattened)
    torch.mm(X, RHS[:, : k + kd], out=combo[:, : k + kd])
    euclid_ip = combo[:, :k]  # (T, k) ⟨x_t, mu_k⟩
    proj_coords = combo[:, k : k + kd].view(X.shape[0], k, d)  # (T, k, d) T_kᵀx_t

    # ‖mu_k‖² via bmm (no (K,D) staging buffer)
    torch.bmm(mu[:k].unsqueeze(1), mu[:k].unsqueeze(2), out=mu_sq[:k].view(k, 1, 1))

    # Euclidean term from combo: (1+alpha)·(‖x‖² - 2⟨x,mu⟩ + ‖mu‖²) into sqdist
    sqdist.copy_(euclid_ip)
    sqdist.mul_(-2.0)
    sqdist.add_(X_sq_sum.unsqueeze(1))  # + ‖x_t‖²  (broadcast over k)
    sqdist.add_(mu_sq[:k].unsqueeze(0))  # + ‖mu_k‖²  (broadcast over t)
    sqdist.mul_(1.0 + alpha)  # (1+alpha)·‖x-mu‖²

    # Projection term: proj_sq[t,k] = Σ_i (T_kᵀx - T_kᵀmu)_i²
    # b_all = T_kᵀ mu_k via one bmm: (k,1,D)·(k,D,d) -> (k,1,d).
    torch.bmm(mu[:k].unsqueeze(1), T_proj[:k], out=b_all[:k].view(k, 1, d))
    proj_coords.sub_(b_all[:k].unsqueeze(0))  # (proj_coords, T_kᵀmu_k), in place
    proj_coords.mul_(proj_coords)  # square in place
    torch.sum(proj_coords, dim=2, out=proj_sq[:, :k])  # Σ_i (...)²

    # Combine in place: sqdist ← (1+alpha)·euclid - proj_sq
    sqdist.sub_(proj_sq[:, :k])
```

### assign_simplex\_

```
assign_simplex_(
    out, cost, temp, dim, scratch_keepdim, scratch_idx
)
```

Assign `cost` onto the simplex along `dim`: soft softmax, or hard one-hot.

Takes :func:`softmax_with_temp_` when the temperature exceeds the dtype machine precision, and :func:`argmin_assign_` when it does not.

Parameters:

| Name              | Type     | Description                                                    | Default    |
| ----------------- | -------- | -------------------------------------------------------------- | ---------- |
| `out`             | `Tensor` | (..., N) output, overwritten.                                  | *required* |
| `cost`            | `Tensor` | (..., N) cost tensor, read-only.                               | *required* |
| `temp`            | `float`  | Softmax temperature; must be finite and > 0 on the soft route. | *required* |
| `dim`             | `int`    | Axis to assign along.                                          | *required* |
| `scratch_keepdim` | `Tensor` | (..., 1) workspace for the soft route.                         | *required* |
| `scratch_idx`     | `Tensor` | (..., 1) int64 workspace for the hard route.                   | *required* |

Source code in `src/entlearn/primitives/softmax.py`

```
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
```

### classification_output_accumulate_into_source_cost\_

```
classification_output_accumulate_into_source_cost_(
    cost_source,
    log_theta_out,
    Pi,
    delta,
    effective_output_weights,
    labelled,
    scratch_TM,
)
```

Add the classification output term to the source block's coupling cost.

Adds `-delta·labelled[t]·eow[t]·Σ_m Pi[t, m]·log_theta_out[m, k_source]` to `cost_source[t, k_source]`. The caller must zero `cost_source` first, and `scratch_TM` must not share storage with it.

Parameters:

| Name                       | Type     | Description                                                         | Default                                                                             |
| -------------------------- | -------- | ------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `cost_source`              | `Tensor` | (T, K_source) cost buffer, accumulated into.                        | *required*                                                                          |
| `log_theta_out`            | `Tensor` | (M, K_source) log of the transition matrix.                         | *required*                                                                          |
| `Pi`                       | `Tensor` | (T, M) classification target, one-hot or per-instance distribution. | *required*                                                                          |
| `delta`                    | `float`  | Raw output coupling strength, no 1/T.                               | *required*                                                                          |
| `effective_output_weights` | \`Tensor | None\`                                                              | (T,) per-instance class·sample weights (eow), or None when Pi already carries them. |
| `labelled`                 | `Tensor` | (T,) boolean mask of labelled instances.                            | *required*                                                                          |
| `scratch_TM`               | `Tensor` | (T, M) workspace.                                                   | *required*                                                                          |

Source code in `src/entlearn/primitives/output.py`

```
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
```

### classification_output_loss\_

```
classification_output_loss_(
    out_scalar,
    log_theta_out,
    Pi,
    gamma_source,
    delta,
    effective_output_weights,
    labelled,
    scratch_TM,
    scratch_TK_source,
    scratch_scalar,
)
```

Add a classification head's loss to `out_scalar`.

Adds `-delta·Σ_{t,m,k} labelled[t]·eow[t]·Pi[t,m]·gamma_source[t,k]· log_theta_out[m,k]`.

Parameters:

| Name                       | Type     | Description                                                         | Default                                                                             |
| -------------------------- | -------- | ------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `out_scalar`               | `Tensor` | () loss accumulator, added into.                                    | *required*                                                                          |
| `log_theta_out`            | `Tensor` | (M, K_source) log of the head's transition matrix.                  | *required*                                                                          |
| `Pi`                       | `Tensor` | (T, M) classification target, one-hot or per-instance distribution. | *required*                                                                          |
| `gamma_source`             | `Tensor` | (T, K_source) source-block affiliations.                            | *required*                                                                          |
| `delta`                    | `float`  | Raw output coupling strength, no 1/T.                               | *required*                                                                          |
| `effective_output_weights` | \`Tensor | None\`                                                              | (T,) per-instance class·sample weights (eow), or None when Pi already carries them. |
| `labelled`                 | `Tensor` | (T,) boolean mask of labelled instances.                            | *required*                                                                          |
| `scratch_TM`               | `Tensor` | (T, M) workspace.                                                   | *required*                                                                          |
| `scratch_TK_source`        | `Tensor` | (T, K_source) workspace.                                            | *required*                                                                          |
| `scratch_scalar`           | `Tensor` | () workspace.                                                       | *required*                                                                          |

Source code in `src/entlearn/primitives/output.py`

```
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
```

### compute_gamma_wt\_

```
compute_gamma_wt_(gamma_wt, denom, gamma, Wt=None)
```

Write `gamma_wt = Γ ∘ Wt` and its column sums `denom[k] = Σ_t gamma_wt[t, k]`.

`Wt = None` means uniform weights, so `gamma_wt = Γ` and `denom` is the plain column sum of `Γ`.

Parameters:

| Name       | Type     | Description                                    | Default                             |
| ---------- | -------- | ---------------------------------------------- | ----------------------------------- |
| `gamma_wt` | `Tensor` | (T, K) output, overwritten.                    | *required*                          |
| `denom`    | `Tensor` | (K,) output, overwritten with the column sums. | *required*                          |
| `gamma`    | `Tensor` | (T, K) row-stochastic cluster affiliations Γ.  | *required*                          |
| `Wt`       | \`Tensor | None\`                                         | (T,) per-instance weights, or None. |

Source code in `src/entlearn/primitives/centroids.py`

```
def compute_gamma_wt_(
    gamma_wt: torch.Tensor,
    denom: torch.Tensor,
    gamma: torch.Tensor,
    Wt: torch.Tensor | None = None,
) -> None:
    """Write ``gamma_wt = Γ ∘ Wt`` and its column sums ``denom[k] = Σ_t gamma_wt[t, k]``.

    ``Wt = None`` means uniform weights, so ``gamma_wt = Γ`` and ``denom`` is the plain
    column sum of ``Γ``.

    Args:
        gamma_wt: ``(T, K)`` output, overwritten.
        denom: ``(K,)`` output, overwritten with the column sums.
        gamma: ``(T, K)`` row-stochastic cluster affiliations ``Γ``.
        Wt: ``(T,)`` per-instance weights, or ``None``.
    """
    if Wt is None:
        gamma_wt.copy_(gamma)
    else:
        torch.mul(gamma, Wt.unsqueeze(1), out=gamma_wt)
    torch.sum(gamma_wt, dim=0, out=denom)  # per-cluster mass
```

### compute_log_partition\_

```
compute_log_partition_(
    out_scalar, b, temp, scratch_T, scratch_scalar
)
```

Write the log-partition `logsumexp_t(-b/temp)` into `out_scalar`.

Computes `m + log Σ_t exp(u[t] - m)` with `u = -b/temp` and `m = max_t u`, over the `(T,)` vector `b`. `out_scalar` is overwritten, not added into.

Parameters:

| Name             | Type     | Description                                  | Default    |
| ---------------- | -------- | -------------------------------------------- | ---------- |
| `out_scalar`     | `Tensor` | () output, overwritten.                      | *required* |
| `b`              | `Tensor` | (T,) cost vector, read-only. T >= 1.         | *required* |
| `temp`           | `float`  | Softmax temperature; must be finite and > 0. | *required* |
| `scratch_T`      | `Tensor` | (T,) workspace for the shifted exponentials. | *required* |
| `scratch_scalar` | `Tensor` | () workspace for the max m.                  | *required* |

Source code in `src/entlearn/primitives/softmax.py`

```
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
```

### compute_wd_cost_categorical\_

```
compute_wd_cost_categorical_(
    Wd_cost_slice_cat,
    X_cat,
    gamma_wt,
    logC_cat_list,
    combined_scales,
    cat_numerator_list,
    scratch_KM,
    *,
    need_cost=True,
)
```

Write each categorical feature's discretisation error into `Wd_cost_slice_cat`.

Per feature `i`, forms the `(M_d, K)` numerator and takes its Frobenius inner product with the log centroids::

```
cat_numerator_i[m, k] = Σ_t X̃_i[m, t]·gamma_wt[t, k]
Wd_cost_slice_cat[i]  = -combined_scales[i]·Σ_{k,m} logC_cat_i[k,m]·cat_numerator_i[m,k]
```

A feature given as `(T,)` integer codes builds its numerator by scatter-add rather than by a matmul against a dense one-hot matrix; a `(T, M_d)` feature uses the matmul. Elements of `Wd_cost_slice_cat` are overwritten.

Also leaves every `cat_numerator_i` populated for the categorical centroid update.

Parameters:

| Name                 | Type               | Description                                                                                               | Default    |
| -------------------- | ------------------ | --------------------------------------------------------------------------------------------------------- | ---------- |
| `Wd_cost_slice_cat`  | `Tensor`           | (D_cat,) output, overwritten.                                                                             | *required* |
| `X_cat`              | `Sequence[Tensor]` | Per feature, a (T, M_d) distribution or a (T,) long tensor of integer codes.                              | *required* |
| `gamma_wt`           | `Tensor`           | (T, K) the Γ ∘ Wt product.                                                                                | *required* |
| `logC_cat_list`      | `Sequence[Tensor]` | Per feature, (K, M_d) log of the categorical centroids.                                                   | *required* |
| `combined_scales`    | `Tensor`           | (D_cat,) per-feature scaling coefficients, independent of Wd.                                             | *required* |
| `cat_numerator_list` | `Sequence[Tensor]` | Per feature, an (M_d, K_max) output whose leading active columns are overwritten with X̃_iᵀ @ gamma_wt.    | *required* |
| `scratch_KM`         | `Tensor`           | (K, max_d M_d) workspace.                                                                                 | *required* |
| `need_cost`          | `bool`             | False computes only cat_numerator_list and leaves Wd_cost_slice_cat untouched. Pass it when Wd is frozen. | `True`     |

Source code in `src/entlearn/primitives/centroids.py`

```
def compute_wd_cost_categorical_(
    Wd_cost_slice_cat: torch.Tensor,
    X_cat: Sequence[torch.Tensor],
    gamma_wt: torch.Tensor,
    logC_cat_list: Sequence[torch.Tensor],
    combined_scales: torch.Tensor,
    cat_numerator_list: Sequence[torch.Tensor],
    scratch_KM: torch.Tensor,
    *,
    need_cost: bool = True,
) -> None:
    """Write each categorical feature's discretisation error into ``Wd_cost_slice_cat``.

    Per feature ``i``, forms the ``(M_d, K)`` numerator and takes its Frobenius inner
    product with the log centroids::

        cat_numerator_i[m, k] = Σ_t X̃_i[m, t]·gamma_wt[t, k]
        Wd_cost_slice_cat[i]  = -combined_scales[i]·Σ_{k,m} logC_cat_i[k,m]·cat_numerator_i[m,k]

    A feature given as ``(T,)`` integer codes builds its numerator by scatter-add
    rather than by a matmul against a dense one-hot matrix; a ``(T, M_d)`` feature uses
    the matmul. Elements of ``Wd_cost_slice_cat`` are overwritten.

    Also leaves every ``cat_numerator_i`` populated for the categorical centroid update.

    Args:
        Wd_cost_slice_cat: ``(D_cat,)`` output, overwritten.
        X_cat: Per feature, a ``(T, M_d)`` distribution or a ``(T,)`` long tensor of
            integer codes.
        gamma_wt: ``(T, K)`` the ``Γ ∘ Wt`` product.
        logC_cat_list: Per feature, ``(K, M_d)`` log of the categorical centroids.
        combined_scales: ``(D_cat,)`` per-feature scaling coefficients, independent of
            ``Wd``.
        cat_numerator_list: Per feature, an ``(M_d, K_max)`` output whose leading
            active columns are overwritten with ``X̃_iᵀ @ gamma_wt``.
        scratch_KM: ``(K, max_d M_d)`` workspace.
        need_cost: ``False`` computes only ``cat_numerator_list`` and leaves
            ``Wd_cost_slice_cat`` untouched. Pass it when ``Wd`` is frozen.
    """
    K = gamma_wt.shape[1]
    for X_i, numerator, logC_i, scale_i, slice_i in zip(
        X_cat, cat_numerator_list, logC_cat_list, combined_scales, Wd_cost_slice_cat, strict=True
    ):
        num_i = numerator[:, :K]
        if X_i.dim() == 1:  # fast path: one-hot scatter-add
            num_i.zero_()
            num_i.index_add_(0, X_i, gamma_wt)  # num_i[code_t] += gamma_wt[t]
        else:  # matrix path
            torch.mm(X_i.transpose(0, 1), gamma_wt, out=num_i)  # X̃ᵀ @ gamma_wt
        if not need_cost:
            continue  # Wd frozen: the numerator above is the only output the caller consumes
        view = scratch_KM[:, : logC_i.shape[1]]  # (K, M_d)
        torch.mul(logC_i, num_i.transpose(0, 1), out=view)  # logC ∘ numᵀ
        torch.sum(view, dim=(0, 1), out=slice_i)  # ⟨logC, numᵀ⟩_F -> output element
        slice_i.mul_(scale_i).neg_()  # times -combined_scales[i]
```

### compute_wd_cost_euclidean\_

```
compute_wd_cost_euclidean_(
    Wd_cost_slice,
    X,
    C,
    gamma_wt,
    denom,
    X_sq,
    centroid_numerator,
    scratch_T,
    scratch_KD,
    scratch_D,
    *,
    need_cost=True,
)
```

Write the per-feature Euclidean cost `b_d` into `Wd_cost_slice`.

Computes `b_d[d] = Σ_t Σ_k gamma_wt[t,k]·(X[t,d] - C[k,d])²`, expanded as::

```
b_d[d] = Σ_t mass[t]·X_sq[t,d]      (mass[t] = Σ_k gamma_wt[t,k])
       - 2·Σ_k C[k,d]·numer[k,d]    (numer = gamma_wtᵀ @ X)
       + Σ_k denom[k]·C[k,d]²
```

Also leaves `centroid_numerator = gamma_wtᵀ @ X` populated, which :func:`update_euclidean_centroids_` then consumes. `centroid_numerator` and `scratch_KD` must be distinct buffers. `C` is not modified.

Parameters:

| Name                 | Type     | Description                                                                                           | Default    |
| -------------------- | -------- | ----------------------------------------------------------------------------------------------------- | ---------- |
| `Wd_cost_slice`      | `Tensor` | (D,) output, overwritten with b_d.                                                                    | *required* |
| `X`                  | `Tensor` | (T, D) continuous data.                                                                               | *required* |
| `C`                  | `Tensor` | (K, D) current continuous centroids, read-only.                                                       | *required* |
| `gamma_wt`           | `Tensor` | (T, K) the Γ ∘ Wt product.                                                                            | *required* |
| `denom`              | `Tensor` | (K,) per-cluster mass Σ_t gamma_wt[t,k].                                                              | *required* |
| `X_sq`               | `Tensor` | (T, D) element-wise X².                                                                               | *required* |
| `centroid_numerator` | `Tensor` | (K, D) output, overwritten with gamma_wtᵀ @ X.                                                        | *required* |
| `scratch_T`          | `Tensor` | (T,) workspace.                                                                                       | *required* |
| `scratch_KD`         | `Tensor` | (K, D) workspace.                                                                                     | *required* |
| `scratch_D`          | `Tensor` | (D,) workspace.                                                                                       | *required* |
| `need_cost`          | `bool`   | False computes only centroid_numerator and leaves Wd_cost_slice untouched. Pass it when Wd is frozen. | `True`     |

Source code in `src/entlearn/primitives/centroids.py`

```
def compute_wd_cost_euclidean_(
    Wd_cost_slice: torch.Tensor,
    X: torch.Tensor,
    C: torch.Tensor,
    gamma_wt: torch.Tensor,
    denom: torch.Tensor,
    X_sq: torch.Tensor,
    centroid_numerator: torch.Tensor,
    scratch_T: torch.Tensor,
    scratch_KD: torch.Tensor,
    scratch_D: torch.Tensor,
    *,
    need_cost: bool = True,
) -> None:
    """Write the per-feature Euclidean cost ``b_d`` into ``Wd_cost_slice``.

    Computes ``b_d[d] = Σ_t Σ_k gamma_wt[t,k]·(X[t,d] - C[k,d])²``, expanded as::

        b_d[d] = Σ_t mass[t]·X_sq[t,d]      (mass[t] = Σ_k gamma_wt[t,k])
               - 2·Σ_k C[k,d]·numer[k,d]    (numer = gamma_wtᵀ @ X)
               + Σ_k denom[k]·C[k,d]²

    Also leaves ``centroid_numerator = gamma_wtᵀ @ X`` populated, which
    :func:`update_euclidean_centroids_` then consumes. ``centroid_numerator`` and
    ``scratch_KD`` must be distinct buffers. ``C`` is not modified.

    Args:
        Wd_cost_slice: ``(D,)`` output, overwritten with ``b_d``.
        X: ``(T, D)`` continuous data.
        C: ``(K, D)`` current continuous centroids, read-only.
        gamma_wt: ``(T, K)`` the ``Γ ∘ Wt`` product.
        denom: ``(K,)`` per-cluster mass ``Σ_t gamma_wt[t,k]``.
        X_sq: ``(T, D)`` element-wise ``X²``.
        centroid_numerator: ``(K, D)`` output, overwritten with ``gamma_wtᵀ @ X``.
        scratch_T: ``(T,)`` workspace.
        scratch_KD: ``(K, D)`` workspace.
        scratch_D: ``(D,)`` workspace.
        need_cost: ``False`` computes only ``centroid_numerator`` and leaves
            ``Wd_cost_slice`` untouched. Pass it when ``Wd`` is frozen.
    """
    torch.mm(gamma_wt.transpose(0, 1), X, out=centroid_numerator)  # numer = gamma_wtᵀ @ X
    if not need_cost:
        return  # Wd frozen: the numerator above is the only output the caller consumes
    # Term 1: Σ_t mass[t]·X_sq[t,d], with mass[t] = Σ_k gamma_wt[t,k]
    torch.sum(gamma_wt, dim=1, out=scratch_T)
    torch.mv(X_sq.transpose(0, 1), scratch_T, out=Wd_cost_slice)
    # Term 2: - 2·Σ_k C[k,d]·numer[k,d]
    torch.mul(C, centroid_numerator, out=scratch_KD)
    torch.sum(scratch_KD, dim=0, out=scratch_D)
    Wd_cost_slice.sub_(scratch_D, alpha=2.0)
    # Term 3: + Σ_k denom[k]·C[k,d]²
    torch.mul(C, C, out=scratch_KD)
    scratch_KD.mul_(denom.unsqueeze(1))
    torch.sum(scratch_KD, dim=0, out=scratch_D)
    Wd_cost_slice.add_(scratch_D)
```

### compute_wt_cost\_

```
compute_wt_cost_(
    Wt_cost, gamma, sqdist_wd, cat_cost, scratch_TK
)
```

Write the per-instance weight cost into `Wt_cost`.

Reduces the two cached costs over the clusters::

```
Wt_cost[t] = Σ_k gamma[t,k]·(sqdist_wd[t,k] + cat_cost[t,k])
```

Either cost may be `None` when that modality is absent, but not both.

Parameters:

| Name         | Type     | Description                             | Default                                                                                |
| ------------ | -------- | --------------------------------------- | -------------------------------------------------------------------------------------- |
| `Wt_cost`    | `Tensor` | (T,) output, overwritten.               | *required*                                                                             |
| `gamma`      | `Tensor` | (T, K) end-of-iteration affiliations Γ. | *required*                                                                             |
| `sqdist_wd`  | \`Tensor | None\`                                  | (T, K) Wd-weighted Euclidean distances, or None when D_cont == 0.                      |
| `cat_cost`   | \`Tensor | None\`                                  | (T, K) summed categorical cost, weighted by Wd but not by Wt, or None when D_cat == 0. |
| `scratch_TK` | `Tensor` | (T, K) workspace.                       | *required*                                                                             |

Source code in `src/entlearn/primitives/reductions.py`

```
def compute_wt_cost_(
    Wt_cost: torch.Tensor,
    gamma: torch.Tensor,
    sqdist_wd: torch.Tensor | None,
    cat_cost: torch.Tensor | None,
    scratch_TK: torch.Tensor,
) -> None:
    """Write the per-instance weight cost into ``Wt_cost``.

    Reduces the two cached costs over the clusters::

        Wt_cost[t] = Σ_k gamma[t,k]·(sqdist_wd[t,k] + cat_cost[t,k])

    Either cost may be ``None`` when that modality is absent, but not both.

    Args:
        Wt_cost: ``(T,)`` output, overwritten.
        gamma: ``(T, K)`` end-of-iteration affiliations Γ.
        sqdist_wd: ``(T, K)`` Wd-weighted Euclidean distances, or ``None`` when
            ``D_cont == 0``.
        cat_cost: ``(T, K)`` summed categorical cost, weighted by ``Wd`` but not by
            ``Wt``, or ``None`` when ``D_cat == 0``.
        scratch_TK: ``(T, K)`` workspace.
    """
    if sqdist_wd is not None and cat_cost is not None:  # continuous and categorical features
        torch.add(sqdist_wd, cat_cost, out=scratch_TK)  # sqdist_wd + cat_cost
        scratch_TK.mul_(gamma)  # gamma ∘ (sqdist_wd + cat_cost)
    elif sqdist_wd is not None:  # continuous only
        torch.mul(gamma, sqdist_wd, out=scratch_TK)  # gamma ∘ sqdist_wd
    else:  # categorical only
        assert cat_cost is not None, "at least one of sqdist_wd / cat_cost is required"
        torch.mul(gamma, cat_cost, out=scratch_TK)  # gamma ∘ cat_cost
    torch.sum(scratch_TK, dim=1, out=Wt_cost)  # Σ_k -> (T,) row reduction
```

### densify_categorical

```
densify_categorical(X_cat, m_cat, dtype)
```

Return every categorical column as a `(T, M_d)` distribution.

A column of int64 codes is one-hot-encoded into fresh rows; a column that is already a distribution is passed through unchanged (not copied). The inputs are not mutated.

Parameters:

| Name    | Type           | Description                                                                                             | Default    |
| ------- | -------------- | ------------------------------------------------------------------------------------------------------- | ---------- |
| `X_cat` | `list[Tensor]` | Per feature, either a (T,) int64 tensor of codes or an already-dense (T, M_d) distribution (read-only). | *required* |
| `m_cat` | `list[int]`    | The per-feature categorical cardinalities M_d.                                                          | *required* |
| `dtype` | `dtype`        | Floating dtype for a freshly one-hot-encoded column.                                                    | *required* |

Returns:

| Type           | Description                                                 |
| -------------- | ----------------------------------------------------------- |
| `list[Tensor]` | list\[torch.Tensor\]: Per feature, a (T, M_d) distribution. |

Source code in `src/entlearn/primitives/encoding.py`

```
def densify_categorical(
    X_cat: list[torch.Tensor], m_cat: list[int], dtype: torch.dtype
) -> list[torch.Tensor]:
    """Return every categorical column as a ``(T, M_d)`` distribution.

    A column of int64 codes is one-hot-encoded into fresh rows; a column that is
    already a distribution is passed through unchanged (not copied). The inputs are
    not mutated.

    Args:
        X_cat: Per feature, either a ``(T,)`` int64 tensor of codes or an
            already-dense ``(T, M_d)`` distribution (read-only).
        m_cat: The per-feature categorical cardinalities ``M_d``.
        dtype: Floating dtype for a freshly one-hot-encoded column.

    Returns:
        list[torch.Tensor]: Per feature, a ``(T, M_d)`` distribution.
    """
    dense: list[torch.Tensor] = []
    for i, m_d in enumerate(m_cat):
        col = X_cat[i]
        if col.dim() == 1:  # int64 codes -> fresh one-hot rows
            col = torch.nn.functional.one_hot(col, m_d).to(dtype)
        dense.append(col)
    return dense
```

### effective_dimension

```
effective_dimension(p, dim=-1, *, normalise=True)
```

Return the exponential of entropy along `dim`.

Parameters:

| Name        | Type     | Description                                 | Default    |
| ----------- | -------- | ------------------------------------------- | ---------- |
| `p`         | `Tensor` | A probability tensor.                       | *required* |
| `dim`       | `int`    | Axis that contains the probability vectors. | `-1`       |
| `normalise` | `bool`   | Divide by the axis length when true.        | `True`     |

Returns:

| Type     | Description                               |
| -------- | ----------------------------------------- |
| `Tensor` | The effective dimension with dim removed. |

Source code in `src/entlearn/primitives/statistics.py`

```
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
```

### entropy

```
entropy(p, dim=-1)
```

Return Shannon entropy along `dim`.

Source code in `src/entlearn/primitives/statistics.py`

```
def entropy(p: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Return Shannon entropy along ``dim``."""
    log_p = p.clamp_min(_eps(p.dtype)).log()
    return -(p * log_p).sum(dim=dim)
```

### entropy_penalty\_

```
entropy_penalty_(
    out_scalar,
    p,
    coefficient,
    log_buffer,
    scratch_scalar,
    *,
    row_weights=None,
)
```

Subtract a Shannon-entropy reward from the loss in `out_scalar`.

With `H = -Σ p·log p` over every element of `p`, applies `out_scalar -= coefficient·H`. When `row_weights` is given, each first-axis slice contributes its weight. The log is floored at the dtype machine precision, so zero entries do not contribute. `log_buffer` must match `p`. The caller must zero `out_scalar` first.

Parameters:

| Name             | Type     | Description                                 | Default                                                   |
| ---------------- | -------- | ------------------------------------------- | --------------------------------------------------------- |
| `out_scalar`     | `Tensor` | () loss accumulator, decreased in place.    | *required*                                                |
| `p`              | `Tensor` | Probability tensor of any shape, read-only. | *required*                                                |
| `coefficient`    | `float`  | Non-negative finite weight.                 | *required*                                                |
| `log_buffer`     | `Tensor` | Workspace shaped like p.                    | *required*                                                |
| `scratch_scalar` | `Tensor` | () workspace.                               | *required*                                                |
| `row_weights`    | \`Tensor | None\`                                      | Optional (p.shape[0],) first-axis weights for a matrix p. |

Source code in `src/entlearn/primitives/statistics.py`

```
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
```

### floored_log\_

```
floored_log_(out, source)
```

Write `log(max(source, eps))` into `out`.

`eps` is the machine precision of `out`'s dtype, so zero entries give a large negative number rather than `-inf`.

Parameters:

| Name     | Type     | Description                     | Default    |
| -------- | -------- | ------------------------------- | ---------- |
| `out`    | `Tensor` | Output buffer, overwritten.     | *required* |
| `source` | `Tensor` | Non-negative tensor, read-only. | *required* |

Source code in `src/entlearn/primitives/normalise.py`

```
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
```

### grouped_means

```
grouped_means(index, weights, dense, cat_rows, k)
```

Return mass-weighted row means for each group.

A group whose mass is at or below the machine precision of `weights`'s dtype receives the global dense mean and uniform categorical rows.

Parameters:

| Name       | Type           | Description                                              | Default    |
| ---------- | -------------- | -------------------------------------------------------- | ---------- |
| `index`    | `Tensor`       | (N,) Integer group index for each row.                   | *required* |
| `weights`  | `Tensor`       | (N,) Mass for each row.                                  | *required* |
| `dense`    | `Tensor`       | (N, D) Continuous rows.                                  | *required* |
| `cat_rows` | `list[Tensor]` | (N, M_d) Categorical distribution rows for each feature. | *required* |
| `k`        | `int`          | Number of groups.                                        | *required* |

Returns:

| Type           | Description                                                             |
| -------------- | ----------------------------------------------------------------------- |
| `Tensor`       | The (k, D) dense means, the per-feature (k, M_d) categorical means, and |
| `list[Tensor]` | the (k,) group masses.                                                  |

Source code in `src/entlearn/primitives/geometry.py`

```
def grouped_means(
    index: torch.Tensor,
    weights: torch.Tensor,
    dense: torch.Tensor,
    cat_rows: list[torch.Tensor],
    k: int,
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
    """Return mass-weighted row means for each group.

    A group whose mass is at or below the machine precision of ``weights``'s dtype
    receives the global dense mean and uniform categorical rows.

    Args:
        index: ``(N,)``  Integer group index for each row.
        weights: ``(N,)``  Mass for each row.
        dense: ``(N, D)`` Continuous rows.
        cat_rows: ``(N, M_d)`` Categorical distribution rows for each feature.
        k: Number of groups.

    Returns:
        The ``(k, D)`` dense means, the per-feature ``(k, M_d)`` categorical means, and
        the ``(k,)`` group masses.
    """
    masses = torch.zeros(k, dtype=weights.dtype, device=weights.device)
    masses.index_add_(0, index, weights)
    dense_means = torch.zeros(k, dense.shape[1], dtype=dense.dtype, device=dense.device)
    dense_means.index_add_(0, index, weights.unsqueeze(1) * dense)
    categorical_means: list[torch.Tensor] = []
    for rows in cat_rows:
        means = torch.zeros(k, rows.shape[1], dtype=rows.dtype, device=rows.device)
        means.index_add_(0, index, weights.unsqueeze(1) * rows)
        categorical_means.append(means)
    _finalise_group_means_(dense_means, categorical_means, masses, weights, dense)
    return dense_means, categorical_means, masses
```

### inlier_scores\_

```
inlier_scores_(out, Wt_train, Wt_test)
```

Evaluate the right-continuous empirical CDF of `Wt_train` at `Wt_test`.

Source code in `src/entlearn/primitives/statistics.py`

```
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
```

### mark_non_empty_clusters\_

```
mark_non_empty_clusters_(not_empty, cluster_mass, gamma)
```

Flag the clusters whose unweighted mass exceeds dtype machine precision, and count them.

Writes the unweighted mass and the survivor mask::

```
cluster_mass[k] = Σ_t gamma[t, k]
not_empty[k]    = cluster_mass[k] > finfo(gamma.dtype).eps
```

Parameters:

| Name           | Type     | Description                                             | Default    |
| -------------- | -------- | ------------------------------------------------------- | ---------- |
| `not_empty`    | `Tensor` | (K,) bool output, overwritten.                          | *required* |
| `cluster_mass` | `Tensor` | (K,) output, overwritten with the column sums of gamma. | *required* |
| `gamma`        | `Tensor` | (T, K) affiliations, read-only.                         | *required* |

Returns:

| Name  | Type  | Description                                          |
| ----- | ----- | ---------------------------------------------------- |
| `int` | `int` | How many clusters are above dtype machine precision. |

Source code in `src/entlearn/primitives/geometry.py`

```
def mark_non_empty_clusters_(
    not_empty: torch.Tensor,
    cluster_mass: torch.Tensor,
    gamma: torch.Tensor,
) -> int:
    """Flag the clusters whose unweighted mass exceeds dtype machine precision, and count them.

    Writes the unweighted mass and the survivor mask::

        cluster_mass[k] = Σ_t gamma[t, k]
        not_empty[k]    = cluster_mass[k] > finfo(gamma.dtype).eps

    Args:
        not_empty: ``(K,)`` bool output, overwritten.
        cluster_mass: ``(K,)`` output, overwritten with the column sums of ``gamma``.
        gamma: ``(T, K)`` affiliations, read-only.

    Returns:
        int: How many clusters are above dtype machine precision.
    """
    torch.sum(gamma, dim=0, out=cluster_mass)  # cluster_mass[k] = Σ_t gamma[t,k]
    torch.gt(cluster_mass, _eps(gamma.dtype), out=not_empty)
    return int(not_empty.sum().item())  # count survivors
```

### normalise\_

```
normalise_(theta, dim, scratch_K)
```

Project `theta` onto a stochastic simplex along `dim`, in-place.

Parameters:

| Name        | Type     | Description                                                           | Default    |
| ----------- | -------- | --------------------------------------------------------------------- | ---------- |
| `theta`     | `Tensor` | (K_target, K_source), normalised in place; must be non-negative.      | *required* |
| `dim`       | `int`    | Axis summed over: 0 normalises the columns, 1 the rows.               | *required* |
| `scratch_K` | `Tensor` | Workspace for the sums: (K_source,) for dim=0, (K_target,) for dim=1. | *required* |

Source code in `src/entlearn/primitives/normalise.py`

```
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
```

### project

```
project(Y, mu, T_proj, gamma)
```

EOMC reconstruction `Y^proj[n] = Σ_k gamma[n,k]·(mu_k + T_k T_kᵀ (Y_n - mu_k))`.

`Y` `(N, D)`, `mu` `(K, D)`, `T_proj` `(K, D, d)`, `gamma` `(N, K)`; returns `(N, D)`. Splits the mixture into a data term and a centroid term, `Σ_k gamma[n,k]·T_k T_kᵀ Y_n + Σ_k gamma[n,k]·(mu_k - T_k T_kᵀ mu_k)`, so three matrix products provide the whole reconstruction and the `(N, K, D)` broadcast tensor is never built. The largest intermediate is the `(N, K·d)` block of projected coordinates.

Source code in `src/entlearn/primitives/manifold.py`

```
def project(
    Y: torch.Tensor, mu: torch.Tensor, T_proj: torch.Tensor, gamma: torch.Tensor
) -> torch.Tensor:
    """EOMC reconstruction ``Y^proj[n] = Σ_k gamma[n,k]·(mu_k + T_k T_kᵀ (Y_n - mu_k))``.

    ``Y`` ``(N, D)``, ``mu`` ``(K, D)``, ``T_proj`` ``(K, D, d)``, ``gamma`` ``(N, K)``;
    returns ``(N, D)``. Splits the mixture into a data term and a centroid term,
    ``Σ_k gamma[n,k]·T_k T_kᵀ Y_n + Σ_k gamma[n,k]·(mu_k - T_k T_kᵀ mu_k)``, so three
    matrix products provide the whole reconstruction and the ``(N, K, D)`` broadcast
    tensor is never built. The largest intermediate is the ``(N, K·d)`` block of
    projected coordinates.
    """
    N, D = Y.shape
    K, _, d = T_proj.shape
    # [T_1 | … | T_K] as (D, K·d): the stacked right-hand side assemble_metrised_cost_
    # builds, so one GEMM yields every cluster's projected coordinates at once.
    T_flat = T_proj.permute(1, 0, 2).reshape(D, K * d)
    coords = Y @ T_flat  # (N, K·d) = T_kᵀ Y_n, k-major
    coords.view(N, K, d).mul_(gamma.unsqueeze(2))  # weight cluster k's block by gamma[:,k]
    out = coords @ T_flat.t()  # (N, D) = Σ_k gamma[n,k]·T_k T_kᵀ Y_n
    # The centroid term, one cluster-sized bmm pair then a single gamma-weighted mixture.
    b = torch.bmm(T_proj.transpose(1, 2), mu.unsqueeze(2))  # (K, d, 1) = T_kᵀ mu_k
    p = torch.bmm(T_proj, b).squeeze(2)  # (K, D) = T_k T_kᵀ mu_k
    out.addmm_(gamma, mu - p)  # += Σ_k gamma[n,k]·(mu_k - T_k T_kᵀ mu_k)
    return out
```

### propagate_classification_output\_

```
propagate_classification_output_(
    out_TM, gamma_source, theta_out, scratch_T1
)
```

Write the predicted class distribution `out_TM = gamma_source @ θ_outᵀ`.

Every row is then divided by its sum, so it is a distribution over the `M` classes. The row sums are floored at the dtype machine precision, so an all-zero row stays zero instead of becoming `NaN`.

`scratch_T1` must not share storage with `out_TM`, which the shapes permit at `M == 1`.

Parameters:

| Name           | Type     | Description                                                  | Default    |
| -------------- | -------- | ------------------------------------------------------------ | ---------- |
| `out_TM`       | `Tensor` | (T, M) output, overwritten.                                  | *required* |
| `gamma_source` | `Tensor` | (T, K_source) source-block affiliations.                     | *required* |
| `theta_out`    | `Tensor` | (M, K_source) left-stochastic transition matrix of the head. | *required* |
| `scratch_T1`   | `Tensor` | (T, 1) workspace for the row sums.                           | *required* |

Source code in `src/entlearn/primitives/output.py`

```
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
```

### propagate_regression_output\_

```
propagate_regression_output_(out_TM, gamma_source, Cy)
```

Write the predicted targets `out_TM = gamma_source @ Cyᵀ`.

Parameters:

| Name           | Type     | Description                                     | Default    |
| -------------- | -------- | ----------------------------------------------- | ---------- |
| `out_TM`       | `Tensor` | (T, M) output, overwritten.                     | *required* |
| `gamma_source` | `Tensor` | (T, K_source) source-block affiliations.        | *required* |
| `Cy`           | `Tensor` | (M, K_source) regression centroids of the head. | *required* |

Source code in `src/entlearn/primitives/output.py`

```
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
```

### reduce_input_cost\_

```
reduce_input_cost_(
    out_scalar, disc_cost, gamma, scratch_TK, scratch_scalar
)
```

Add the input block's discretisation error to the loss in `out_scalar`.

Adds `Σ_{t,k} gamma[t,k]·disc_cost[t,k]`. The caller must zero `out_scalar` first.

Parameters:

| Name             | Type     | Description                      | Default    |
| ---------------- | -------- | -------------------------------- | ---------- |
| `out_scalar`     | `Tensor` | () loss accumulator, added into. | *required* |
| `disc_cost`      | `Tensor` | (T, K) per-instance input costs. | *required* |
| `gamma`          | `Tensor` | (T, K) input-block affiliations. | *required* |
| `scratch_TK`     | `Tensor` | (T, K) workspace.                | *required* |
| `scratch_scalar` | `Tensor` | () workspace.                    | *required* |

Source code in `src/entlearn/primitives/reductions.py`

```
def reduce_input_cost_(
    out_scalar: torch.Tensor,
    disc_cost: torch.Tensor,
    gamma: torch.Tensor,
    scratch_TK: torch.Tensor,
    scratch_scalar: torch.Tensor,
) -> None:
    """Add the input block's discretisation error to the loss in ``out_scalar``.

    Adds ``Σ_{t,k} gamma[t,k]·disc_cost[t,k]``. The caller must zero ``out_scalar``
    first.

    Args:
        out_scalar: ``()`` loss accumulator, added into.
        disc_cost: ``(T, K)`` per-instance input costs.
        gamma: ``(T, K)`` input-block affiliations.
        scratch_TK: ``(T, K)`` workspace.
        scratch_scalar: ``()`` workspace.
    """
    torch.mul(gamma, disc_cost, out=scratch_TK)  # gamma ∘ disc_cost
    torch.sum(scratch_TK, dim=(0, 1), out=scratch_scalar)  # full reduction into the scalar scratch
    out_scalar.add_(scratch_scalar)  # += Σ_{t,k} gamma·disc_cost
```

### refresh_weighted_norm\_

```
refresh_weighted_norm_(X_sq_wd_sum, X_sq, Wd)
```

Write the Wd-weighted squared norm `X_sq_wd_sum ← X_sq @ Wd`.

Computes `X_sq_wd_sum[t] = Σ_d Wd[d]·X_sq[t, d]`, the `‖x_t‖²_w` term of the distance expansion. Rerun it after every `Wd` update.

Parameters:

| Name          | Type     | Description                                | Default    |
| ------------- | -------- | ------------------------------------------ | ---------- |
| `X_sq_wd_sum` | `Tensor` | (T,) output, overwritten.                  | *required* |
| `X_sq`        | `Tensor` | (T, D) element-wise X², independent of Wd. | *required* |
| `Wd`          | `Tensor` | (D,) continuous feature weights.           | *required* |

Source code in `src/entlearn/primitives/distance.py`

```
def refresh_weighted_norm_(
    X_sq_wd_sum: torch.Tensor,
    X_sq: torch.Tensor,
    Wd: torch.Tensor,
) -> None:
    """Write the Wd-weighted squared norm ``X_sq_wd_sum ← X_sq @ Wd``.

    Computes ``X_sq_wd_sum[t] = Σ_d Wd[d]·X_sq[t, d]``, the ``‖x_t‖²_w`` term of the
    distance expansion. Rerun it after every ``Wd`` update.

    Args:
        X_sq_wd_sum: ``(T,)`` output, overwritten.
        X_sq: ``(T, D)`` element-wise ``X²``, independent of ``Wd``.
        Wd: ``(D,)`` continuous feature weights.
    """
    torch.mv(X_sq, Wd, out=X_sq_wd_sum)
```

### regression_output_accumulate_into_source_cost\_

```
regression_output_accumulate_into_source_cost_(
    cost_source,
    Cy,
    Y,
    target_sq,
    delta,
    effective_output_weights,
    labelled,
    scratch_TK,
    scratch_K,
    scratch_KD,
    *,
    Wm=None,
)
```

Add the regression output term to the source block's coupling cost.

Adds `delta·labelled[t]·eow[t]·‖Y_t - Cy_k‖²_Wm` to `cost_source[t, k]`. Both `scratch_TK` and `scratch_KD` must be disjoint from `cost_source` so already accumulated coupling costs remain intact.

Parameters:

| Name                       | Type     | Description                                                            | Default                                           |
| -------------------------- | -------- | ---------------------------------------------------------------------- | ------------------------------------------------- |
| `cost_source`              | `Tensor` | (T, K_source) cost buffer, accumulated into.                           | *required*                                        |
| `Cy`                       | `Tensor` | (M, K_source) regression centroids of the head.                        | *required*                                        |
| `Y`                        | `Tensor` | (T, M) regression target.                                              | *required*                                        |
| `target_sq`                | `Tensor` | (T,) precomputed Σ_m Y[t, m]², or Σ_m Wm[m]·Y[t, m]² when Wm is given. | *required*                                        |
| `delta`                    | `float`  | Raw output coupling strength, no 1/T.                                  | *required*                                        |
| `effective_output_weights` | `Tensor` | (T,) per-instance output·sample weights (eow).                         | *required*                                        |
| `labelled`                 | `Tensor` | (T,) boolean mask of labelled instances.                               | *required*                                        |
| `scratch_TK`               | `Tensor` | (T, K_source) workspace.                                               | *required*                                        |
| `scratch_K`                | `Tensor` | (K_source,) workspace.                                                 | *required*                                        |
| `scratch_KD`               | `Tensor` | (K_source, M) workspace.                                               | *required*                                        |
| `Wm`                       | \`Tensor | None\`                                                                 | (M,) output-dimension weights, or None for all 1. |

Source code in `src/entlearn/primitives/output.py`

```
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
```

### regression_output_loss\_

```
regression_output_loss_(
    out_scalar,
    Cy,
    Y,
    target_sq,
    gamma_source,
    delta,
    effective_output_weights,
    labelled,
    scratch_TK,
    scratch_K,
    scratch_KD,
    scratch_scalar,
    *,
    Wm=None,
)
```

Add a regression head's loss to `out_scalar`.

Adds `delta·Σ_{t,k} labelled[t]·eow[t]·gamma_source[t,k]·‖Y_t - Cy_k‖²_Wm`.

Parameters:

| Name                       | Type     | Description                                                            | Default                                           |
| -------------------------- | -------- | ---------------------------------------------------------------------- | ------------------------------------------------- |
| `out_scalar`               | `Tensor` | () loss accumulator, added into.                                       | *required*                                        |
| `Cy`                       | `Tensor` | (M, K_source) regression centroids of the head.                        | *required*                                        |
| `Y`                        | `Tensor` | (T, M) regression target.                                              | *required*                                        |
| `target_sq`                | `Tensor` | (T,) precomputed Σ_m Y[t, m]², or Σ_m Wm[m]·Y[t, m]² when Wm is given. | *required*                                        |
| `gamma_source`             | `Tensor` | (T, K_source) source-block affiliations.                               | *required*                                        |
| `delta`                    | `float`  | Raw output coupling strength, no 1/T.                                  | *required*                                        |
| `effective_output_weights` | `Tensor` | (T,) per-instance output·sample weights (eow).                         | *required*                                        |
| `labelled`                 | `Tensor` | (T,) boolean mask of labelled instances.                               | *required*                                        |
| `scratch_TK`               | `Tensor` | (T, K_source) workspace.                                               | *required*                                        |
| `scratch_K`                | `Tensor` | (K_source,) workspace.                                                 | *required*                                        |
| `scratch_KD`               | `Tensor` | (K_source, M) workspace.                                               | *required*                                        |
| `scratch_scalar`           | `Tensor` | () workspace.                                                          | *required*                                        |
| `Wm`                       | \`Tensor | None\`                                                                 | (M,) output-dimension weights, or None for all 1. |

Source code in `src/entlearn/primitives/output.py`

```
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
```

### seed_dissimilarity

```
seed_dissimilarity(
    X, X_cat, logX_cat, Wd_cont, Wd_cat, seed
)
```

Return each row's dissimilarity to a selected data row.

Source code in `src/entlearn/primitives/dissimilarity.py`

```
def seed_dissimilarity(
    X: torch.Tensor,
    X_cat: list[torch.Tensor],
    logX_cat: list[torch.Tensor],
    Wd_cont: torch.Tensor,
    Wd_cat: torch.Tensor,
    seed: int,
) -> torch.Tensor:
    """Return each row's dissimilarity to a selected data row."""
    return feature_d2_to_point(
        X,
        X_cat,
        [categorical_log[seed] for categorical_log in logX_cat],
        Wd_cont,
        Wd_cat,
        X[seed],
    )
```

### softmax_with_temp\_

```
softmax_with_temp_(out, cost, temp, dim, scratch_keepdim)
```

Write `softmax(-cost/temp, dim)` into `out`.

Every slice along `dim` sums to 1. A slice whose exponentials are all non-finite becomes the uniform `1/N`, `N = cost.shape[dim]`.

Parameters:

| Name              | Type     | Description                                              | Default    |
| ----------------- | -------- | -------------------------------------------------------- | ---------- |
| `out`             | `Tensor` | (..., N) output, overwritten.                            | *required* |
| `cost`            | `Tensor` | (..., N) cost tensor, read-only.                         | *required* |
| `temp`            | `float`  | Softmax temperature; must be finite and > 0.             | *required* |
| `dim`             | `int`    | Axis to softmax over.                                    | *required* |
| `scratch_keepdim` | `Tensor` | (..., 1) workspace for the slice max and the normaliser. | *required* |

Source code in `src/entlearn/primitives/softmax.py`

```
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
```

### split_wd

```
split_wd(wd, d_cont)
```

Split the feature weights `wd` into their continuous and categorical parts.

The first `d_cont` entries weight the continuous features; the rest weight the categorical ones. Both returned tensors are views of `wd`.

Parameters:

| Name     | Type     | Description                                         | Default    |
| -------- | -------- | --------------------------------------------------- | ---------- |
| `wd`     | `Tensor` | (D_tot,) feature-weight vector.                     | *required* |
| `d_cont` | `int`    | number of continuous features (the boundary index). | *required* |

Returns:

| Type                    | Description                                                                |
| ----------------------- | -------------------------------------------------------------------------- |
| `tuple[Tensor, Tensor]` | tuple\[torch.Tensor, torch.Tensor\]: the continuous and categorical views. |

Source code in `src/entlearn/primitives/encoding.py`

```
def split_wd(wd: torch.Tensor, d_cont: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Split the feature weights ``wd`` into their continuous and categorical parts.

    The first ``d_cont`` entries weight the continuous features; the rest weight the
    categorical ones. Both returned tensors are views of ``wd``.

    Args:
        wd: ``(D_tot,)`` feature-weight vector.
        d_cont: number of continuous features (the boundary index).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: the continuous and categorical views.
    """
    return wd[:d_cont], wd[d_cont:]
```

### transition_partial_loss\_

```
transition_partial_loss_(
    out_scalar,
    log_theta,
    gamma_target,
    gamma_source,
    delta_eff,
    scratch_TK_source,
    scratch_scalar,
    sample_weights=None,
)
```

Add a connection's coupling loss to `out_scalar`.

Adds `-delta_eff·Σ_{t,k_target,k_source} w[t]·gamma_target[t,k_target]· gamma_source[t,k_source]·log_theta[k_target,k_source]`, with `w[t] = sample_weights[t]` when given, else 1. The caller must zero `out_scalar` first.

Parameters:

| Name                | Type     | Description                                        | Default                             |
| ------------------- | -------- | -------------------------------------------------- | ----------------------------------- |
| `out_scalar`        | `Tensor` | () loss accumulator, added into.                   | *required*                          |
| `log_theta`         | `Tensor` | (K_target, K_source) log of the transition matrix. | *required*                          |
| `gamma_target`      | `Tensor` | (T, K_target) target-block affiliations.           | *required*                          |
| `gamma_source`      | `Tensor` | (T, K_source) source-block affiliations.           | *required*                          |
| `delta_eff`         | `float`  | Coupling strength, including the 1/T divisor.      | *required*                          |
| `scratch_TK_source` | `Tensor` | (T, K_source) workspace.                           | *required*                          |
| `scratch_scalar`    | `Tensor` | () workspace.                                      | *required*                          |
| `sample_weights`    | \`Tensor | None\`                                             | (T,) per-instance weights, or None. |

Source code in `src/entlearn/primitives/coupling.py`

```
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
```

### update_categorical_centroids\_

```
update_categorical_centroids_(
    C_cat_i, logC_cat_i, cat_numerator_i, denom, empty
)
```

Set one categorical feature's centroid distribution and refresh its log cache.

Writes `C_cat_i[k, m] = cat_numerator_i[m, k] / denom[k]`, so every row is a distribution over the feature's `M_d` categories, then refreshes `logC_cat_i = log(max(C_cat_i, eps))`. A cluster whose mass `denom[k]` is at or below `finfo(dtype).eps` counts as empty and takes the uniform `1 / M_d`.

`C_cat_i` must not overlap the numerator or denominator, as it briefly holds the protected divisors. The shared denominator remains unchanged.

Parameters:

| Name              | Type     | Description                                                   | Default    |
| ----------------- | -------- | ------------------------------------------------------------- | ---------- |
| `C_cat_i`         | `Tensor` | (K, M_d) output, overwritten.                                 | *required* |
| `logC_cat_i`      | `Tensor` | (K, M_d) output, overwritten with log(C_cat_i).               | *required* |
| `cat_numerator_i` | `Tensor` | (M_d, K) weighted counts X̃_iᵀ @ (Γ ∘ Wt).                     | *required* |
| `denom`           | `Tensor` | (K,) per-cluster mass Σ_t gamma_wt[t, k].                     | *required* |
| `empty`           | `Tensor` | (K,) bool workspace, overwritten with the empty-cluster mask. | *required* |

Source code in `src/entlearn/primitives/centroids.py`

```
def update_categorical_centroids_(
    C_cat_i: torch.Tensor,
    logC_cat_i: torch.Tensor,
    cat_numerator_i: torch.Tensor,
    denom: torch.Tensor,
    empty: torch.Tensor,
) -> None:
    """Set one categorical feature's centroid distribution and refresh its log cache.

    Writes ``C_cat_i[k, m] = cat_numerator_i[m, k] / denom[k]``, so every row is a
    distribution over the feature's ``M_d`` categories, then refreshes
    ``logC_cat_i = log(max(C_cat_i, eps))``. A cluster whose mass ``denom[k]`` is at or
    below ``finfo(dtype).eps`` counts as empty and takes the uniform ``1 / M_d``.

    ``C_cat_i`` must not overlap the numerator or denominator, as it briefly holds the
    protected divisors. The shared denominator remains unchanged.

    Args:
        C_cat_i: ``(K, M_d)`` output, overwritten.
        logC_cat_i: ``(K, M_d)`` output, overwritten with ``log(C_cat_i)``.
        cat_numerator_i: ``(M_d, K)`` weighted counts ``X̃_iᵀ @ (Γ ∘ Wt)``.
        denom: ``(K,)`` per-cluster mass ``Σ_t gamma_wt[t, k]``.
        empty: ``(K,)`` bool workspace, overwritten with the empty-cluster mask.
    """
    eps = _eps(C_cat_i.dtype)
    uniform = 1.0 / C_cat_i.shape[1]  # 1 / M_d
    torch.le(denom, eps, out=empty)  # empty cluster
    C_cat_i.copy_(denom.unsqueeze(1))
    C_cat_i.masked_fill_(empty.unsqueeze(1), 1.0)
    torch.div(cat_numerator_i.transpose(0, 1), C_cat_i, out=C_cat_i)
    C_cat_i.masked_fill_(empty.unsqueeze(1), uniform)
    floored_log_(logC_cat_i, C_cat_i)  # cache log(C_cat_i)
```

### update_classification_output\_

```
update_classification_output_(
    theta_out,
    log_theta_out,
    Pi,
    gamma_source,
    effective_output_weights,
    labelled,
    scratch_norm,
    scratch_TK_source,
    norm_dim=0,
)
```

Set a classification head's `theta_out` and refresh its log cache.

Normalises the joint counts `Piᵀ @ (labelled·eow ∘ gamma_source)` along `norm_dim`, then refreshes `log_theta_out = log(max(θ_out, eps))`.

`norm_dim=0` normalises columns, so each column is a distribution over the `M` classes. A cluster with no labelled mass takes the uniform `1/M`, and a wholly unlabelled batch makes the whole matrix uniform. `norm_dim=1` normalises rows, so each row is a distribution over the `K_source` clusters; a class with no mass takes the uniform `1/K_source`.

Parameters:

| Name                       | Type     | Description                                                                  | Default                                                                             |
| -------------------------- | -------- | ---------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `theta_out`                | `Tensor` | (M, K_source) output, overwritten.                                           | *required*                                                                          |
| `log_theta_out`            | `Tensor` | (M, K_source) output, overwritten with the floored log.                      | *required*                                                                          |
| `Pi`                       | `Tensor` | (T, M) classification target, one-hot or per-instance distribution.          | *required*                                                                          |
| `gamma_source`             | `Tensor` | (T, K_source) source-block affiliations.                                     | *required*                                                                          |
| `effective_output_weights` | \`Tensor | None\`                                                                       | (T,) per-instance class·sample weights (eow), or None when Pi already carries them. |
| `labelled`                 | `Tensor` | (T,) boolean mask of labelled instances.                                     | *required*                                                                          |
| `scratch_norm`             | `Tensor` | Workspace for the slice sums: (K_source,) at norm_dim=0, (M,) at norm_dim=1. | *required*                                                                          |
| `scratch_TK_source`        | `Tensor` | (T, K_source) workspace.                                                     | *required*                                                                          |
| `norm_dim`                 | `int`    | Axis that sums to one, 0 (columns) or 1 (rows).                              | `0`                                                                                 |

Source code in `src/entlearn/primitives/output.py`

```
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
```

### update_euclidean_centroids\_

```
update_euclidean_centroids_(
    C_out, centroid_numerator, denom, empty
)
```

Set the continuous centroids to `C ← centroid_numerator / denom`.

`C_out[k, :]` becomes the weighted mean of cluster `k`'s instances. A cluster whose mass `denom[k]` is at or below `finfo(dtype).eps` counts as empty and is zeroed instead.

`C_out` must not overlap the numerator or denominator: it briefly holds the protected divisors. The shared denominator remains unchanged.

Parameters:

| Name                 | Type     | Description                                                                      | Default    |
| -------------------- | -------- | -------------------------------------------------------------------------------- | ---------- |
| `C_out`              | `Tensor` | (K, D) output, overwritten.                                                      | *required* |
| `centroid_numerator` | `Tensor` | (K, D) weighted sum gamma_wtᵀ @ X, as left by :func:compute_wd_cost_euclidean\_. | *required* |
| `denom`              | `Tensor` | (K,) per-cluster mass Σ_t gamma_wt[t, k].                                        | *required* |
| `empty`              | `Tensor` | (K,) bool workspace, overwritten with the empty-cluster mask.                    | *required* |

Source code in `src/entlearn/primitives/centroids.py`

```
def update_euclidean_centroids_(
    C_out: torch.Tensor,
    centroid_numerator: torch.Tensor,
    denom: torch.Tensor,
    empty: torch.Tensor,
) -> None:
    """Set the continuous centroids to ``C ← centroid_numerator / denom``.

    ``C_out[k, :]`` becomes the weighted mean of cluster ``k``'s instances. A cluster
    whose mass ``denom[k]`` is at or below ``finfo(dtype).eps`` counts as empty and is
    zeroed instead.

    ``C_out`` must not overlap the numerator or denominator: it briefly holds the
    protected divisors. The shared denominator remains unchanged.

    Args:
        C_out: ``(K, D)`` output, overwritten.
        centroid_numerator: ``(K, D)`` weighted sum ``gamma_wtᵀ @ X``, as left by
            :func:`compute_wd_cost_euclidean_`.
        denom: ``(K,)`` per-cluster mass ``Σ_t gamma_wt[t, k]``.
        empty: ``(K,)`` bool workspace, overwritten with the empty-cluster mask.
    """
    eps = _eps(C_out.dtype)
    torch.le(denom, eps, out=empty)  # empty clusters
    # Reuse the destination for safe divisors; the shared masses remain read-only.
    C_out.copy_(denom.unsqueeze(1))
    C_out.masked_fill_(empty.unsqueeze(1), 1.0)
    torch.div(centroid_numerator, C_out, out=C_out)
    C_out.masked_fill_(empty.unsqueeze(1), 0.0)  # zero empty clusters
```

### update_output_weights\_

```
update_output_weights_(
    Wm,
    Y,
    Y_sq,
    Cy,
    gamma_source,
    effective_output_weights,
    labelled,
    delta,
    epsilon_M,
    Wm_cost,
    masked_TK,
    scratch_MK,
    scratch_K,
    scratch_T,
    scratch_keepdim,
    scratch_idx,
    scratch_KM,
    scratch_M,
)
```

Set a regression head's output-dimension weights `Wm`.

Forms the per-dimension weighted residuals `Wm_cost[m] = delta·Σ_t labelled[t]·eow[t]·Σ_k gamma_source[t, k]· (Y[t, m] - Cy[m, k])²`, then assigns them onto the simplex with :func:`assign_simplex_` at temperature `epsilon_M`.

`masked_TK`, `scratch_MK` and `scratch_KM` must be distinct buffers.

Parameters:

| Name                       | Type     | Description                                               | Default    |
| -------------------------- | -------- | --------------------------------------------------------- | ---------- |
| `Wm`                       | `Tensor` | (M,) output, overwritten.                                 | *required* |
| `Y`                        | `Tensor` | (T, M) regression target.                                 | *required* |
| `Y_sq`                     | `Tensor` | (T, M) element-wise Y² cache.                             | *required* |
| `Cy`                       | `Tensor` | (M, K_source) regression centroids of the head.           | *required* |
| `gamma_source`             | `Tensor` | (T, K_source) source-block affiliations.                  | *required* |
| `effective_output_weights` | `Tensor` | (T,) per-instance output·sample weights (eow).            | *required* |
| `labelled`                 | `Tensor` | (T,) boolean mask of labelled instances.                  | *required* |
| `delta`                    | `float`  | Raw output coupling strength, not scaled by 1/T.          | *required* |
| `epsilon_M`                | `float`  | Finite non-negative softmax temperature.                  | *required* |
| `Wm_cost`                  | `Tensor` | (M,) workspace receiving the per-dimension residuals.     | *required* |
| `masked_TK`                | `Tensor` | (T, K_source) workspace.                                  | *required* |
| `scratch_MK`               | `Tensor` | (M, K_source) workspace.                                  | *required* |
| `scratch_K`                | `Tensor` | (K_source,) workspace.                                    | *required* |
| `scratch_T`                | `Tensor` | (T,) workspace.                                           | *required* |
| `scratch_keepdim`          | `Tensor` | (1,) workspace for the softmax reduction.                 | *required* |
| `scratch_idx`              | `Tensor` | (1,) int64 workspace for the hard-assignment index.       | *required* |
| `scratch_KM`               | `Tensor` | (K_source, M) workspace for the Euclidean cost expansion. | *required* |
| `scratch_M`                | `Tensor` | (M,) workspace for the per-dimension reductions.          | *required* |

Source code in `src/entlearn/primitives/output.py`

```
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
```

### update_regression_output\_

```
update_regression_output_(
    Cy,
    Y,
    gamma_source,
    effective_output_weights,
    labelled,
    Y_mean_weighted,
    denom_buf,
    scratch_MK,
    scratch_TK_source,
)
```

Set a regression head's centroids `Cy`.

Writes `Cy[m, k] = numer[m, k] / denom[k]` with `numer = Yᵀ @ (labelled·eow ∘ gamma_source)` and `denom` its column sums. A cluster with mass at or below the dtype machine precision takes the weighted labelled mean `Y_mean_weighted[m]` instead. Denominators are protected before division, a wholly unlabelled batch therefore makes every column that mean.

`scratch_TK_source`, `scratch_MK` and `Cy` must be distinct buffers. `Cy` briefly holds the protected divisors.

Parameters:

| Name                       | Type     | Description                                                   | Default    |
| -------------------------- | -------- | ------------------------------------------------------------- | ---------- |
| `Cy`                       | `Tensor` | (M, K_source) output, overwritten.                            | *required* |
| `Y`                        | `Tensor` | (T, M) regression target.                                     | *required* |
| `gamma_source`             | `Tensor` | (T, K_source) source-block affiliations.                      | *required* |
| `effective_output_weights` | `Tensor` | (T,) per-instance output·sample weights (eow).                | *required* |
| `labelled`                 | `Tensor` | (T,) boolean mask of labelled instances.                      | *required* |
| `Y_mean_weighted`          | `Tensor` | (M,) eow-weighted labelled mean of Y, the zero-mass fallback. | *required* |
| `denom_buf`                | `Tensor` | (K_source,) workspace.                                        | *required* |
| `scratch_MK`               | `Tensor` | (M, K_source) workspace.                                      | *required* |
| `scratch_TK_source`        | `Tensor` | (T, K_source) workspace.                                      | *required* |

Source code in `src/entlearn/primitives/output.py`

```
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
```

### update_theta\_

```
update_theta_(
    theta,
    log_theta,
    gamma_target,
    gamma_source,
    scratch_K_norm,
    sample_weights=None,
    scratch_TK_source=None,
    pseudocount=0.0,
    norm_dim=0,
)
```

Set `theta` to the normalised co-occurrence of target and source affiliations.

Computes `θ ← normalise(Γ_targetᵀ @ Γ_source, norm_dim)` in place and refreshes `log_theta = log(max(θ, eps))`. `sample_weights`, if given, weights the source affiliations first; `pseudocount` adds a constant `a` to every count, giving `(counts + a)/(slicesum + N·a)` with `N = theta.shape[norm_dim]`.

`norm_dim` picks the axis that sums to one: `0` normalises columns, `1` normalises rows. A slice with no mass becomes uniform.

Parameters:

| Name                | Type     | Description                                                                         | Default                                                        |
| ------------------- | -------- | ----------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| `theta`             | `Tensor` | (K_target, K_source), overwritten.                                                  | *required*                                                     |
| `log_theta`         | `Tensor` | (K_target, K_source), overwritten.                                                  | *required*                                                     |
| `gamma_target`      | `Tensor` | (T, K_target) target-block affiliations.                                            | *required*                                                     |
| `gamma_source`      | `Tensor` | (T, K_source) source-block affiliations.                                            | *required*                                                     |
| `scratch_K_norm`    | `Tensor` | Workspace for the slice sums: (K_source,) at norm_dim=0, (K_target,) at norm_dim=1. | *required*                                                     |
| `sample_weights`    | \`Tensor | None\`                                                                              | (T,) per-instance weights, or None.                            |
| `scratch_TK_source` | \`Tensor | None\`                                                                              | (T, K_source) workspace, required iff sample_weights is given. |
| `pseudocount`       | `float`  | Added to every count before normalising; 0.0 is a no-op.                            | `0.0`                                                          |
| `norm_dim`          | `int`    | Axis that sums to one after normalising, 0 (columns) or 1 (rows).                   | `0`                                                            |

Source code in `src/entlearn/primitives/transitions.py`

```
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
```

### weighted_assign_simplex\_

```
weighted_assign_simplex_(
    out,
    cost,
    temp,
    row_weights,
    scratch_keepdim,
    scratch_idx,
)
```

Assign rows onto the simplex when row `t`'s entropy carries weight `row_weights[t]`.

Row `t` receives `softmax(-cost[t] / (temp · row_weights[t]))`. A temperature at or below the dtype machine precision assigns every row by hard minimum cost.

Parameters:

| Name              | Type     | Description                                                            | Default    |
| ----------------- | -------- | ---------------------------------------------------------------------- | ---------- |
| `out`             | `Tensor` | (T, K) row-stochastic output (modified).                               | *required* |
| `cost`            | `Tensor` | (T, K) weighted assignment cost, read-only.                            | *required* |
| `temp`            | `float`  | Raw affiliation temperature. Must be finite and > 0 on the soft route. | *required* |
| `row_weights`     | `Tensor` | (T,) positive row weights.                                             | *required* |
| `scratch_keepdim` | `Tensor` | (T, 1) workspace (modified).                                           | *required* |
| `scratch_idx`     | `Tensor` | (T, 1) int64 workspace (modified).                                     | *required* |

Source code in `src/entlearn/primitives/softmax.py`

```
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
```

### weighted_sq_distance\_

```
weighted_sq_distance_(
    out,
    A,
    B,
    A_sq_sum,
    scratch_K,
    scratch_KD,
    feature_weights=None,
)
```

Write the feature-weighted squared distances into `out`.

Computes `out[t, k] = Σ_d w[d]·(A[t, d] - B[k, d])²`, expanded as `‖a‖²_w - 2·⟨a, b⟩_w + ‖b‖²_w`. `feature_weights=None` means every weight is 1.

Operands should be appropriately scaled; continuous features are expected in or near `[0, 1]`. Large offsets and nearly coincident points can cause cancellation and loss of relative accuracy in this expansion.

Parameters:

| Name              | Type     | Description                                                               | Default                            |
| ----------------- | -------- | ------------------------------------------------------------------------- | ---------------------------------- |
| `out`             | `Tensor` | (T, K) output, overwritten.                                               | *required*                         |
| `A`               | `Tensor` | (T, D) left operand, e.g. data X or targets Y.                            | *required*                         |
| `B`               | `Tensor` | (K, D) right operand, e.g. centroids C or Cyᵀ.                            | *required*                         |
| `A_sq_sum`        | `Tensor` | (T,) precomputed Σ_d w[d]·A[t, d]², weighted by the same feature_weights. | *required*                         |
| `scratch_K`       | `Tensor` | (K,) workspace for ‖b_k‖²_w.                                              | *required*                         |
| `scratch_KD`      | `Tensor` | (K, D) workspace.                                                         | *required*                         |
| `feature_weights` | \`Tensor | None\`                                                                    | (D,) per-feature weights, or None. |

Source code in `src/entlearn/primitives/distance.py`

```
def weighted_sq_distance_(
    out: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    A_sq_sum: torch.Tensor,
    scratch_K: torch.Tensor,
    scratch_KD: torch.Tensor,
    feature_weights: torch.Tensor | None = None,
) -> None:
    """Write the feature-weighted squared distances into ``out``.

    Computes ``out[t, k] = Σ_d w[d]·(A[t, d] - B[k, d])²``, expanded as
    ``‖a‖²_w - 2·⟨a, b⟩_w + ‖b‖²_w``. ``feature_weights=None`` means every weight is 1.

    Operands should be appropriately scaled; continuous features are expected in
    or near ``[0, 1]``. Large offsets and nearly coincident points can cause
    cancellation and loss of relative accuracy in this expansion.

    Args:
        out: ``(T, K)`` output, overwritten.
        A: ``(T, D)`` left operand, e.g. data ``X`` or targets ``Y``.
        B: ``(K, D)`` right operand, e.g. centroids ``C`` or ``Cyᵀ``.
        A_sq_sum: ``(T,)`` precomputed ``Σ_d w[d]·A[t, d]²``, weighted by the same
            ``feature_weights``.
        scratch_K: ``(K,)`` workspace for ``‖b_k‖²_w``.
        scratch_KD: ``(K, D)`` workspace.
        feature_weights: ``(D,)`` per-feature weights, or ``None``.
    """
    if feature_weights is not None:
        torch.mul(B, feature_weights, out=scratch_KD)  # B ∘ w
        torch.mm(A, scratch_KD.transpose(0, 1), out=out)  # ⟨A, B⟩_w
        scratch_KD.mul_(B)  # w · B²
    else:
        torch.mm(A, B.transpose(0, 1), out=out)  # ⟨A, B⟩
        torch.mul(B, B, out=scratch_KD)  # B²
    torch.sum(scratch_KD, dim=1, out=scratch_K)  # ‖B‖²_w  (Σ_d w·B²)
    out.mul_(-2.0)
    out.add_(A_sq_sum.unsqueeze(1))  # + ‖a‖²_w  (broadcast over K)
    out.add_(scratch_K.unsqueeze(0))  # + ‖b‖²_w  (broadcast over T)
```

### weighted_sq_norm\_

```
weighted_sq_norm_(out, X, Wd)
```

Write `out[t] = Σ_d Wd[d]·X[t, d]²` without materialising `X²`.

The same quantity as :func:`refresh_weighted_norm_`, reduced straight from `X` a row block at a time. Use this when the `X²` cache is not already available.

Parameters:

| Name  | Type     | Description                      | Default    |
| ----- | -------- | -------------------------------- | ---------- |
| `out` | `Tensor` | (T,) output, overwritten.        | *required* |
| `X`   | `Tensor` | (T, D) continuous data.          | *required* |
| `Wd`  | `Tensor` | (D,) continuous feature weights. | *required* |

Source code in `src/entlearn/primitives/distance.py`

```
def weighted_sq_norm_(out: torch.Tensor, X: torch.Tensor, Wd: torch.Tensor) -> None:
    """Write ``out[t] = Σ_d Wd[d]·X[t, d]²`` without materialising ``X²``.

    The same quantity as :func:`refresh_weighted_norm_`, reduced straight from ``X`` a
    row block at a time. Use this when the ``X²`` cache is not already available.

    Args:
        out: ``(T,)`` output, overwritten.
        X: ``(T, D)`` continuous data.
        Wd: ``(D,)`` continuous feature weights.
    """
    n_rows, width = X.shape
    if n_rows == 0 or width == 0:
        out.zero_()
        return
    step = max(1, _NORM_BLOCK_BYTES // (width * X.element_size()))
    for start in range(0, n_rows, step):
        stop = min(start + step, n_rows)
        block = X[start:stop]
        torch.mv(block * block, Wd, out=out[start:stop])
```

### weighted_subspace\_

```
weighted_subspace_(
    X,
    mu,
    w,
    d,
    *,
    out,
    sqrtw_buf,
    centred_buf,
    wsum_buf=None,
    cov_buf=None,
    evals_buf=None,
    evecs_buf=None,
    revidx=None,
    svd_U=None,
    svd_S=None,
    svd_Vh=None,
)
```

Top-`d` eigenvectors of the `w`-weighted covariance of `X` about `mu`.

`Cov = Σ_t w_t (x_t-mu)(x_t-mu)ᵀ / Σ_t w_t`; writes its `d` dominant eigenvectors `(D, d)` into `out` and returns it. When `D <= N` it takes the eigendecomposition of the `D x D` covariance. When `D > N` it takes the SVD of the `N x D` scaled-centred matrix instead. The covariance is rescaled without shifting its spectrum, and the selected eigenvectors are copied directly into `out`.

Always pass `out`, `sqrtw_buf` and `centred_buf`, plus the set for whichever branch runs: `wsum_buf`, `cov_buf`, `evals_buf`, `evecs_buf` and `revidx` for `D <= N`, or `svd_U`, `svd_S` and `svd_Vh` for `D > N`.

Parameters:

| Name          | Type     | Description                                                 | Default                                                                                                          |
| ------------- | -------- | ----------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `X`           | `Tensor` | (N, D) data rows.                                           | *required*                                                                                                       |
| `mu`          | `Tensor` | (D,) centre the covariance is taken about.                  | *required*                                                                                                       |
| `w`           | `Tensor` | (N,) non-negative instance weights.                         | *required*                                                                                                       |
| `d`           | `int`    | number of dominant eigenvectors to return.                  | *required*                                                                                                       |
| `out`         | `Tensor` | (D, d) output, overwritten with the top-d eigenvectors.     | *required*                                                                                                       |
| `sqrtw_buf`   | `Tensor` | (N,) scratch; receives sqrt(w).                             | *required*                                                                                                       |
| `centred_buf` | `Tensor` | (N, D) scratch; receives the scaled-centred sqrt(w)·(X-mu). | *required*                                                                                                       |
| `wsum_buf`    | \`Tensor | None\`                                                      | (1,) scratch (D≤N); receives weight mass and covariance scale.                                                   |
| `cov_buf`     | \`Tensor | None\`                                                      | (D, D) scratch (D≤N); receives the rescaled covariance.                                                          |
| `evals_buf`   | \`Tensor | None\`                                                      | (D,) write-only eigenvalue scratch (D≤N).                                                                        |
| `evecs_buf`   | \`Tensor | None\`                                                      | (D, D) scratch (D≤N); receives the eigenvectors.                                                                 |
| `revidx`      | \`Tensor | None\`                                                      | (d,) int64 [D-1, …, D-d] (D≤N); selects the top-d descending.                                                    |
| `svd_U`       | \`Tensor | None\`                                                      | (N, m) write-only left-singular scratch (D>N), m = min(N, D).                                                    |
| `svd_S`       | \`Tensor | None\`                                                      | (m,) write-only singular-value scratch (D>N).                                                                    |
| `svd_Vh`      | \`Tensor | None\`                                                      | (m, D) scratch (D>N), or (D, D) when d > N needs an orthonormal completion; receives the right singular vectors. |

Returns:

| Type     | Description                                           |
| -------- | ----------------------------------------------------- |
| `Tensor` | The out tensor holding the top-d eigenvectors (D, d). |

Raises:

| Type         | Description                           |
| ------------ | ------------------------------------- |
| `ValueError` | If a required buffer is not supplied. |

Source code in `src/entlearn/primitives/manifold.py`

```
def weighted_subspace_(
    X: torch.Tensor,
    mu: torch.Tensor,
    w: torch.Tensor,
    d: int,
    *,
    out: torch.Tensor,
    sqrtw_buf: torch.Tensor,
    centred_buf: torch.Tensor,
    wsum_buf: torch.Tensor | None = None,
    cov_buf: torch.Tensor | None = None,
    evals_buf: torch.Tensor | None = None,
    evecs_buf: torch.Tensor | None = None,
    revidx: torch.Tensor | None = None,
    svd_U: torch.Tensor | None = None,
    svd_S: torch.Tensor | None = None,
    svd_Vh: torch.Tensor | None = None,
) -> torch.Tensor:
    """Top-``d`` eigenvectors of the ``w``-weighted covariance of ``X`` about ``mu``.

    ``Cov = Σ_t w_t (x_t-mu)(x_t-mu)ᵀ / Σ_t w_t``; writes its ``d`` dominant
    eigenvectors ``(D, d)`` into ``out`` and returns it. When ``D <= N`` it takes the
    eigendecomposition of the ``D x D`` covariance. When ``D > N`` it takes the SVD of
    the ``N x D`` scaled-centred matrix instead. The covariance is rescaled without
    shifting its spectrum, and the selected eigenvectors are copied directly into ``out``.

    Always pass ``out``, ``sqrtw_buf`` and ``centred_buf``, plus the set for whichever
    branch runs: ``wsum_buf``, ``cov_buf``, ``evals_buf``, ``evecs_buf`` and ``revidx``
    for ``D <= N``, or ``svd_U``, ``svd_S`` and ``svd_Vh`` for ``D > N``.

    Args:
        X: ``(N, D)`` data rows.
        mu: ``(D,)`` centre the covariance is taken about.
        w: ``(N,)`` non-negative instance weights.
        d: number of dominant eigenvectors to return.
        out: ``(D, d)`` output, overwritten with the top-``d`` eigenvectors.
        sqrtw_buf: ``(N,)`` scratch; receives ``sqrt(w)``.
        centred_buf: ``(N, D)`` scratch; receives the scaled-centred ``sqrt(w)·(X-mu)``.
        wsum_buf: ``(1,)`` scratch (D≤N); receives weight mass and covariance scale.
        cov_buf: ``(D, D)`` scratch (D≤N); receives the rescaled covariance.
        evals_buf: ``(D,)`` write-only eigenvalue scratch (D≤N).
        evecs_buf: ``(D, D)`` scratch (D≤N); receives the eigenvectors.
        revidx: ``(d,)`` int64 ``[D-1, …, D-d]`` (D≤N); selects the top-``d`` descending.
        svd_U: ``(N, m)`` write-only left-singular scratch (D>N), ``m = min(N, D)``.
        svd_S: ``(m,)`` write-only singular-value scratch (D>N).
        svd_Vh: ``(m, D)`` scratch (D>N), or ``(D, D)`` when ``d > N`` needs
            an orthonormal completion; receives the right singular vectors.

    Returns:
        The ``out`` tensor holding the top-``d`` eigenvectors ``(D, d)``.

    Raises:
        ValueError: If a required buffer is not supplied.
    """
    N, D = X.shape
    # M = sqrt(w)·(X - mu): the scaled-centred matrix is orientation-independent, so
    # build it once (into centred_buf) before the eigh/SVD dispatch below.
    sw = torch.sqrt(w, out=sqrtw_buf)  # (N,)
    M = torch.sub(X, mu, out=centred_buf)  # (N, D) centred
    M.mul_(sw.unsqueeze(1))  # scale rows by sqrt(w)
    if D <= N:  # eigh the DxD covariance
        # A raise is used here instead of an assert, since `python -O` strips asserts, and `torch.sum`/`torch.mm` accept `out=None` by allocating.
        if wsum_buf is None or cov_buf is None or evals_buf is None or evecs_buf is None:
            raise ValueError("the D <= N branch needs wsum_buf, cov_buf, evals_buf and evecs_buf")
        if revidx is None:
            raise ValueError("the D <= N branch needs revidx")
        # wsum = clamp_min(Σ_t w_t)
        wsum = torch.sum(w, dim=0, keepdim=True, out=wsum_buf)
        wsum.clamp_min_(torch.finfo(X.dtype).tiny)
        # cov = MᵀM / wsum into cov_buf.
        cov = torch.mm(M.t(), M, out=cov_buf)
        cov.div_(wsum)
        # Scalar rescaling imposes no absolute noise floor. An absolute diagonal
        # shift can instead erase the signal of a small but nonzero cloud.
        torch.amax(cov.diagonal(), dim=0, keepdim=True, out=wsum_buf)
        wsum_buf.clamp_min_(torch.finfo(X.dtype).tiny)
        cov.div_(wsum_buf)
        # eigh (ascending); top-d descending via the reversed-index select.
        _evals, evecs = torch.linalg.eigh(cov, out=(evals_buf, evecs_buf))
        # revidx = [D-1, …, D-d] selects the top-d columns in descending order
        return torch.index_select(evecs, 1, revidx, out=out)
    # SVD the thin NxD matrix
    if svd_U is None or svd_S is None or svd_Vh is None:
        raise ValueError("the D > N branch needs svd_U, svd_S and svd_Vh")
    full = d > N
    if svd_Vh.shape != (D if full else N, D):
        raise ValueError("svd_Vh must hold the requested right singular basis")
    # gesvd on CUDA: the default Jacobi driver loses weak directions (see seed_projectors).
    _U, _S, Vh = torch.linalg.svd(
        M, full_matrices=full, driver="gesvd" if M.is_cuda else None, out=(svd_U, svd_S, svd_Vh)
    )
    out.copy_(Vh[:d].t())  # right singular vectors = eigenvectors
    return out
```
