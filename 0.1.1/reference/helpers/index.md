# Optional helpers

These helpers keep no state. `feature_weights` chooses starting feature weights from statistics of the labelled rows, `delta_cat_guess` computes the default `delta_cat` from the training data, and `reporting` summarises a fitted `Network` through its public methods. The statistical helpers load SciPy or scikit-learn only when called; the `sklearn` extra installs both.

## Initialisation

When the features far outnumber the labelled rows, pass `null_permutations` (three is usually enough with hundreds of features) and `random_state`. The helper then subtracts a permutation-null quantile from each score, so the many unrelated features no longer take most of the weight. It is off by default.

## entlearn.helpers.feature_weights

```
feature_weights(
    X_cont,
    categorical_codes,
    target,
    *,
    task,
    method,
    labelled_mask=None,
    output_weights=None,
    null_permutations=0,
    null_quantile=0.99,
    random_state=None,
    dtype=float64,
    device=None,
)
```

Return feature weights scored from each feature's relationship with the target.

Continuous features precede categorical features in the returned tensor. `"correlation"` scores a feature by correlation ratio, Cramér's V or absolute Pearson correlation according to its modality and the task. `"mutual_info"` scores it with scikit-learn's mutual-information estimator. The weights are the scores divided by their sum. When fewer than two rows are labelled or no score is positive, the weights are uniform.

Both statistics are biased, so an unrelated feature can still have non-zero score. A positive `null_permutations` corrects for this. The labelled rows' targets are shuffled that many times and every feature is scored against each shuffle. The `null_quantile` of that pool is subtracted from the modality's scores.

Use the threshold when the features outnumber the labelled rows. With few features relative to the labelled rows, the bias is small and the threshold can remove weakly informative features, so it is off by default.

Parameters:

| Name                | Type                                      | Description                                                                                                                 | Default                                                                                                                                                                                        |
| ------------------- | ----------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `X_cont`            | `object`                                  | Two-dimensional continuous input accepted by NumPy.                                                                         | *required*                                                                                                                                                                                     |
| `categorical_codes` | `Sequence[object]`                        | One one-dimensional code array per categorical feature.                                                                     | *required*                                                                                                                                                                                     |
| `target`            | `object`                                  | One-dimensional classification codes or one- or two-dimensional regression targets.                                         | *required*                                                                                                                                                                                     |
| `task`              | `Literal['classification', 'regression']` | Statistical relationship to measure.                                                                                        | *required*                                                                                                                                                                                     |
| `method`            | `Literal['correlation', 'mutual_info']`   | Statistic that scores each feature, "correlation" or "mutual_info".                                                         | *required*                                                                                                                                                                                     |
| `labelled_mask`     | \`object                                  | None\`                                                                                                                      | Optional boolean array selecting labelled rows.                                                                                                                                                |
| `output_weights`    | \`object                                  | None\`                                                                                                                      | Optional regression-output weights used when averaging scores.                                                                                                                                 |
| `null_permutations` | `int`                                     | Number of target permutations that estimate the scores of features unrelated to the target. 0 leaves the scores unadjusted. | `0`                                                                                                                                                                                            |
| `null_quantile`     | `float`                                   | Quantile, in [0, 1], of the pooled null scores subtracted from each score when null_permutations is positive.               | `0.99`                                                                                                                                                                                         |
| `random_state`      | \`int                                     | None\`                                                                                                                      | Seed for the random numbers the call draws: the target permutations and scikit-learn's mutual-information estimator. It must be None for "correlation" without permutations, which draws none. |
| `dtype`             | `dtype`                                   | Floating dtype of the returned tensor. Statistical calculations use float64 regardless of this output dtype.                | `float64`                                                                                                                                                                                      |
| `device`            | \`device                                  | str                                                                                                                         | None\`                                                                                                                                                                                         |

Returns:

| Type     | Description                                                  |
| -------- | ------------------------------------------------------------ |
| `Tensor` | Detached, non-negative feature weights with unit total mass. |

Raises:

| Type         | Description                                                                                                                                                                                                                                |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `ValueError` | If method is unknown, null_permutations is not a non-negative integer, null_quantile lies outside [0, 1], random_state accompanies "correlation" without permutations, the arrays are incompatible or the tensor placement is unsupported. |

Source code in `src/entlearn/helpers/_feature_weights.py`

```
def feature_weights(
    X_cont: object,
    categorical_codes: Sequence[object],
    target: object,
    *,
    task: Literal["classification", "regression"],
    method: Literal["correlation", "mutual_info"],
    labelled_mask: object | None = None,
    output_weights: object | None = None,
    null_permutations: int = 0,
    null_quantile: float = 0.99,
    random_state: int | None = None,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return feature weights scored from each feature's relationship with the target.

    Continuous features precede categorical features in the returned tensor.
    ``"correlation"`` scores a feature by correlation ratio, Cramér's V or absolute
    Pearson correlation according to its modality and the task. ``"mutual_info"``
    scores it with scikit-learn's mutual-information estimator. The weights are the
    scores divided by their sum. When fewer than two rows are labelled or no score is
    positive, the weights are uniform.

    Both statistics are biased, so an unrelated feature can still have non-zero
    score. A positive ``null_permutations`` corrects for this. The labelled rows'
    targets are shuffled that many times and every feature is scored against each
    shuffle.  The ``null_quantile`` of that pool is subtracted from the modality's
    scores.

    Use the threshold when the features outnumber the labelled rows.
    With few features relative to the labelled rows, the bias is small and
    the threshold can remove weakly informative features, so it is off by default.

    Args:
        X_cont: Two-dimensional continuous input accepted by NumPy.
        categorical_codes: One one-dimensional code array per categorical feature.
        target: One-dimensional classification codes or one- or two-dimensional
            regression targets.
        task: Statistical relationship to measure.
        method: Statistic that scores each feature, ``"correlation"`` or
            ``"mutual_info"``.
        labelled_mask: Optional boolean array selecting labelled rows.
        output_weights: Optional regression-output weights used when averaging scores.
        null_permutations: Number of target permutations that estimate the scores of
            features unrelated to the target. ``0`` leaves the scores unadjusted.
        null_quantile: Quantile, in ``[0, 1]``, of the pooled null scores subtracted
            from each score when ``null_permutations`` is positive.
        random_state: Seed for the random numbers the call draws: the target
            permutations and scikit-learn's mutual-information estimator. It must be
            ``None`` for ``"correlation"`` without permutations, which draws none.
        dtype: Floating dtype of the returned tensor. Statistical calculations use
            float64 regardless of this output dtype.
        device: Device of the returned tensor.

    Returns:
        Detached, non-negative feature weights with unit total mass.

    Raises:
        ValueError: If ``method`` is unknown, ``null_permutations`` is not a
            non-negative integer, ``null_quantile`` lies outside ``[0, 1]``,
            ``random_state`` accompanies ``"correlation"`` without permutations, the
            arrays are incompatible or the tensor placement is unsupported.
    """
    if method not in ("correlation", "mutual_info"):
        raise ValueError("method must be 'correlation' or 'mutual_info'")
    if (
        isinstance(null_permutations, bool)
        or not isinstance(null_permutations, Integral)
        or null_permutations < 0
    ):
        raise ValueError("null_permutations must be a non-negative integer")
    if (
        isinstance(null_quantile, bool)
        or not isinstance(null_quantile, Real)
        or not 0.0 <= null_quantile <= 1.0
    ):
        raise ValueError("null_quantile must be a number in [0, 1]")
    if method == "correlation" and null_permutations == 0 and random_state is not None:
        raise ValueError("random_state must be None for correlation without null permutations")
    scores = _feature_weight_scores(
        X_cont,
        categorical_codes,
        target,
        task,
        method,
        labelled_mask,
        output_weights,
        random_state,
        int(null_permutations),
        float(null_quantile),
    )
    return _to_tensor(scores, dtype=dtype, device=device)
```

## Categorical scale

## entlearn.helpers.delta_cat_guess

```
delta_cat_guess(
    X_cont, categorical_codes, *, sample_weights=None
)
```

Return a starting value for `delta_cat`.

Errors in a continuous feature have magnitude similar to its variance. Categorical features, on the other hand, have potentially unbounded cross-entropy errors, depending on their entropy and number of levels.

This helper returns a scaling factor that brings the latter to a comparable range, using the median variance of the continuous features and the median cross-entropy of the categorical ones, both weighted by the sample weights.

Every column is treated as uninformative, so the returned value is a heuristic that can be used as a starting point. Fitting uses it when `Input.delta_cat` is `None`.

Parameters:

| Name                | Type               | Description                                                                                                                                                    | Default                                 |
| ------------------- | ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------- |
| `X_cont`            | `object`           | Two-dimensional continuous input accepted by NumPy, scaled as for the fit. The calculation keeps a float32 or float64 input's dtype; other inputs use float64. | *required*                              |
| `categorical_codes` | `Sequence[object]` | One one-dimensional code array per categorical feature. A feature's levels are the distinct codes it holds.                                                    | *required*                              |
| `sample_weights`    | \`object           | None\`                                                                                                                                                         | Optional positive weights for the rows. |

Returns:

| Type    | Description                  |
| ------- | ---------------------------- |
| `float` | The positive starting value. |

Raises:

| Type         | Description                                                                                                                                                                                |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `ValueError` | If no continuous feature varies or no categorical feature has at least two levels, if X_cont holds a non-finite value or a code is NaN or None, or if the arrays or weights are malformed. |

Source code in `src/entlearn/helpers/_delta_cat.py`

```
def delta_cat_guess(
    X_cont: object,
    categorical_codes: Sequence[object],
    *,
    sample_weights: object | None = None,
) -> float:
    """Return a starting value for ``delta_cat``.

    Errors in a continuous feature have magnitude similar to its variance. Categorical
    features, on the other hand, have potentially unbounded cross-entropy errors, depending
    on their entropy and number of levels.

    This helper returns a scaling factor that brings the latter to a comparable range, using
    the median variance of the continuous features and the median cross-entropy of the
    categorical ones, both weighted by the sample weights.

    Every column is treated as uninformative, so the returned value is a heuristic that can
    be used as a starting point. Fitting uses it when ``Input.delta_cat`` is ``None``.

    Args:
        X_cont: Two-dimensional continuous input accepted by NumPy, scaled as for the fit.
            The calculation keeps a float32 or float64 input's dtype; other inputs use float64.
        categorical_codes: One one-dimensional code array per categorical feature. A
            feature's levels are the distinct codes it holds.
        sample_weights: Optional positive weights for the rows.

    Returns:
        The positive starting value.

    Raises:
        ValueError: If no continuous feature varies or no categorical feature has at least
            two levels, if ``X_cont`` holds a non-finite value or a code is NaN or ``None``,
            or if the arrays or weights are malformed.
    """
    array = np.asarray(X_cont)
    if array.dtype not in (np.float32, np.float64):
        array = array.astype(np.float64)
    # The checks Network.fit applies to the same inputs.
    X = _validate_continuous(torch.from_numpy(array), None)
    rows = X.shape[0]
    weights = None
    if sample_weights is not None:
        weights = _stage_weights(
            torch.from_numpy(np.asarray(sample_weights, dtype=np.float64)),
            name="sample_weights",
            length=rows,
            dtype=X.dtype,
            device=X.device,
            positive=True,
        )

    codes = []
    for position, column in enumerate(categorical_codes):
        array = np.asarray(column)
        if array.shape != (rows,):
            raise ValueError(f"categorical_codes[{position}] must have one entry per X_cont row")
        if _has_missing(array):
            raise ValueError(f"categorical_codes[{position}] contains NaN or None")
        codes.append(torch.from_numpy(np.unique(array, return_inverse=True)[1]))
    guess = _guess_delta_cat(X, codes, weights)
    if guess is None:
        raise ValueError(
            "delta_cat_guess needs a varying continuous feature and a categorical feature "
            "with at least two levels"
        )
    return guess
```

## Fitted reporting

## entlearn.helpers.reporting

State-free tensor reporting through the public fitted Network interface.

### active_features

```
active_features(model, *, tol=1.0)
```

Return sorted int64 indices of features whose weight exceeds `tol / D`.

Indices use tensor feature order and remain on the fitted device. This reports standard-input weights, not manifold tangent participation. A threshold is relative to the uniform share, not an absolute numerical floor. The default selects weights strictly above that uniform share.

Raises:

| Type         | Description                                                                                                   |
| ------------ | ------------------------------------------------------------------------------------------------------------- |
| `ValueError` | If the model is not a fitted Network, has no feature weights, has uniform weights, or tol is outside \[0, D). |

Source code in `src/entlearn/helpers/reporting.py`

```
@torch.inference_mode()
def active_features(model: Network, *, tol: float = 1.0) -> torch.Tensor:
    """Return sorted int64 indices of features whose weight exceeds ``tol / D``.

    Indices use tensor feature order and remain on the fitted device.
    This reports standard-input weights, not manifold tangent participation.
    A threshold is relative to
    the uniform share, not an absolute numerical floor. The default selects
    weights strictly above that uniform share.

    Raises:
        ValueError: If the model is not a fitted Network, has no feature weights,
            has uniform weights, or ``tol`` is outside ``[0, D)``.
    """
    _require_network(model)
    return _active_mask(model, tol).nonzero(as_tuple=True)[0]
```

### count_parameters

```
count_parameters(
    model,
    *,
    include_affiliations=False,
    raw=False,
    active_tol=None,
)
```

Count parameter entries or geometric degrees of freedom of a Network.

Fixed feature, instance and output weights are excluded from the count. Training affiliations and instance weights (if learned) count only if `include_affiliations=True`.

By default (`raw=False`):

- each vector simplex has D-1 free parameters.
- A manifold basis represents a subspace, contributing `d * (D - d)` freedoms per active cluster instead of `D * d`. With `alpha=0`, the centroid also loses its `d` redundant tangent coordinates.

`active_tol` retains only features whose weight exceeds `active_tol / D`. Information needed for choosing the subset itself is not considered.

Raises:

| Type         | Description                                                                                                                 |
| ------------ | --------------------------------------------------------------------------------------------------------------------------- |
| `ValueError` | If the model or active threshold is invalid, or a requested fitted value is unavailable, including omitted row-bound state. |

Source code in `src/entlearn/helpers/reporting.py`

```
@torch.inference_mode()
def count_parameters(  # noqa: C901
    model: Network,
    *,
    include_affiliations: bool = False,
    raw: bool = False,
    active_tol: float | None = None,
) -> int:
    """Count parameter entries or geometric degrees of freedom of a Network.

    Fixed feature, instance and output weights are excluded from the count. Training
    affiliations and instance weights (if learned) count only if ``include_affiliations=True``.

    By default (``raw=False``):
    - each vector simplex has D-1 free parameters.
    - A manifold basis represents a subspace, contributing ``d * (D - d)`` freedoms per
    active cluster instead of ``D * d``. With ``alpha=0``, the centroid also
    loses its ``d`` redundant tangent coordinates.

    ``active_tol`` retains only features whose weight exceeds ``active_tol / D``.
    Information needed for choosing the subset itself is not considered.

    Raises:
        ValueError: If the model or active threshold is invalid, or a requested
            fitted value is unavailable, including omitted row-bound state.
    """
    _require_network(model)
    mask = None if active_tol is None else _active_mask(model, active_tol)
    first, head = _reporting_owners(model)
    continuous = model.inspect("continuous_centroids")[first.name]
    total = (
        continuous.numel()
        if mask is None
        else (continuous.shape[0] * int(mask[: model.schema.D_cont].sum()))
    )
    match first:
        case Input():
            if model.schema.M_cat:
                categorical = model.inspect("categorical_centroids")[first.name]
                for index, centroid in enumerate(categorical):
                    if mask is None or bool(mask[model.schema.D_cont + index]):
                        total += centroid.numel() - (0 if raw else centroid.shape[0])
            if math.isfinite(first.epsilon_D):
                weights = model.inspect("feature_weights")[first.name]
                total += weights.numel() - (0 if raw else 1) if mask is None else int(mask.sum())
        case ManifoldInput():
            projectors = model.inspect("manifold_projectors")[first.name]
            K, D, d = projectors.shape
            total += projectors.numel() if raw else K * d * (D - d)
            if not raw and first.alpha == 0:
                total -= K * d
        case _:
            assert_never(first)
    if any(isinstance(block, Hidden) for block in model.recipe.blocks):
        transitions = model.inspect("transition_matrices")
        for connection in model.recipe.connections:
            if connection.name in transitions:
                theta = transitions[connection.name]
                total += theta.numel() - (
                    0 if raw else theta.shape[1 if connection.coupling is Coupling.M else 0]
                )
    parameters = model.inspect("head_parameters")[head.name]
    match head:
        case ClassificationHead():
            theta = parameters["theta"]
            total += theta.numel() - (
                0 if raw else theta.shape[1 if head.coupling is Coupling.M else 0]
            )
        case RegressionHead():
            total += parameters["C_y"].numel()
            if math.isfinite(head.epsilon_M):
                total += parameters["W_M"].numel() - (0 if raw else 1)
        case _:
            assert_never(head)
    if include_affiliations:
        affiliations = model.inspect("training_affiliations")
        for gamma in affiliations.values():
            total += gamma.numel() - (0 if raw else gamma.shape[0])
        if math.isfinite(first.epsilon_T):
            weights = model.inspect("training_instance_weights")[first.name]
            total += weights.numel() - (0 if raw else 1)
    return total
```

### effective_dimensions

```
effective_dimensions(model, *, normalise=True)
```

Report realised spreads of fitted distributions.

Each clustering block reports the mean row-wise `affiliations` spread. The input also reports `instance_weights` and, for standard input, `feature_weights`. A regression head reports `output_weights` when explicitly represented, including fixed weights. `normalise=True` divides each effective dimension by the number of entries in its distribution.

Raises:

| Type         | Description                                                        |
| ------------ | ------------------------------------------------------------------ |
| `ValueError` | If the model is not fitted or its training values are unavailable. |

Source code in `src/entlearn/helpers/reporting.py`

```
@torch.inference_mode()
def effective_dimensions(
    model: Network, *, normalise: bool = True
) -> dict[str, dict[str, torch.Tensor]]:
    """Report realised spreads of fitted distributions.

    Each clustering block reports the mean row-wise ``affiliations`` spread.
    The input also reports ``instance_weights`` and, for standard input,
    ``feature_weights``. A regression head reports ``output_weights`` when
    explicitly represented, including fixed weights. ``normalise=True`` divides
    each effective dimension by the number of entries in its distribution.

    Raises:
        ValueError: If the model is not fitted or its training values are unavailable.
    """
    _require_network(model)
    first, _ = _reporting_owners(model)
    affiliations = model.inspect("training_affiliations")
    report = {
        name: {"affiliations": effective_dimension(gamma, normalise=normalise).mean()}
        for name, gamma in affiliations.items()
    }
    for request, field in (
        ("training_instance_weights", "instance_weights"),
        ("feature_weights", "feature_weights"),
    ):
        if field == "feature_weights" and not isinstance(first, Input):
            continue
        for name, weights in model.inspect(request).items():
            report[name][field] = effective_dimension(weights, normalise=normalise)
    for name, parameters in model.inspect("head_parameters").items():
        if "W_M" in parameters:
            report[name] = {
                "output_weights": effective_dimension(parameters["W_M"], normalise=normalise)
            }
    return report
```

### feature_importances

```
feature_importances(model)
```

Return fitted feature weights or weighted manifold tangent participation.

Standard input returns its learned or fixed feature simplex. Manifold input averages the diagonal tangent-projector leverage using training affiliations weighted by fitted training instance weights, then divides by subspace dimension. Those instance weights are learned, supplied, or uniform according to the fit.

For orthonormal tangent bases Tₖ ∈ ℝᴰˣᵈ, training affiliations Γₜₖ and fitted instance weights wₜ, the weighted occupancy and feature participation are:

```
ρₖ = (∑ₜ wₜ Γₜₖ) / (∑ₜ wₜ)
ℓₖⱼ = (TₖTₖᵀ)ⱼⱼ = ∑ᵣ₌₁ᵈ (Tₖ)ⱼᵣ²
importanceⱼ = (∑ₖ ρₖ ℓₖⱼ) / d
```

Since ∑ₖ ρₖ = 1 and tr(TₖTₖᵀ) = d, ∑ⱼ importanceⱼ = 1. Rotating a tangent basis leaves its projector and participation unchanged.

Manifold participation depends on feature coordinates and scaling; it is neither a learned feature-weight parameter nor a measure of predictive importance. It requires retained training affiliations and instance weights. Raises when row-bound values are unavailable.

Source code in `src/entlearn/helpers/reporting.py`

```
@torch.inference_mode()
def feature_importances(model: Network) -> torch.Tensor:
    """Return fitted feature weights or weighted manifold tangent participation.

    Standard input returns its learned or fixed feature simplex. Manifold input
    averages the diagonal tangent-projector leverage using training affiliations
    weighted by fitted training instance weights, then divides by subspace dimension.
    Those instance weights are learned, supplied, or uniform according to the fit.

    For orthonormal tangent bases Tₖ ∈ ℝᴰˣᵈ, training affiliations Γₜₖ and fitted
    instance weights wₜ, the weighted occupancy and feature participation are:

        ρₖ = (∑ₜ wₜ Γₜₖ) / (∑ₜ wₜ)
        ℓₖⱼ = (TₖTₖᵀ)ⱼⱼ = ∑ᵣ₌₁ᵈ (Tₖ)ⱼᵣ²
        importanceⱼ = (∑ₖ ρₖ ℓₖⱼ) / d

    Since ∑ₖ ρₖ = 1 and tr(TₖTₖᵀ) = d, ∑ⱼ importanceⱼ = 1.
    Rotating a tangent basis leaves its projector and participation unchanged.

    Manifold participation depends on feature coordinates and scaling; it is neither
    a learned feature-weight parameter nor a measure of predictive importance.
    It requires retained training affiliations and instance weights.
    Raises when row-bound values are unavailable.
    """
    _require_network(model)
    first, _ = _reporting_owners(model)
    match first:
        case Input():
            return model.inspect("feature_weights")[first.name]
        case ManifoldInput():
            basis = model.inspect("manifold_projectors")[first.name]
            gamma = model.inspect("training_affiliations")[first.name]
            weights = model.inspect("training_instance_weights")[first.name]
            occupancy = (weights / weights.sum()) @ gamma
            return occupancy @ basis.square().sum(dim=2) / basis.shape[2]
        case _:
            assert_never(first)
```

### target_weights

```
target_weights(model)
```

Return detached regression target weights in fitted output order.

Finite `epsilon_M` learns a relative preference for lower training residual costs. The costs average squared residuals against cluster output centroids under training affiliations and supervision weights. They depend on target scale, coupling strength and temperature. Infinite `epsilon_M` keeps supplied weights fixed, or uses an implicit uniform simplex when none were supplied.

Source code in `src/entlearn/helpers/reporting.py`

```
@torch.inference_mode()
def target_weights(model: Network) -> torch.Tensor:
    """Return detached regression target weights in fitted output order.

    Finite ``epsilon_M`` learns a relative preference for lower training residual
    costs. The costs average squared residuals
    against cluster output centroids under training affiliations and supervision
    weights. They depend on target scale, coupling strength and temperature.
    Infinite ``epsilon_M`` keeps supplied weights fixed, or uses an implicit
    uniform simplex when none were supplied.
    """
    _require_network(model)
    _, head = _reporting_owners(model)
    if not isinstance(head, RegressionHead):
        raise ValueError("target_weights requires a regression Network")
    parameters = model.inspect("head_parameters")[head.name]
    if "W_M" in parameters:
        return parameters["W_M"]
    return parameters["C_y"].new_full((model.schema.M,), 1.0 / model.schema.M)
```
