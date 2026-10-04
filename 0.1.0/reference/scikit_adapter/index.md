# Scikit-learn adapter

The optional tabular interface. See the [estimator guide](https://entropiclearning.github.io/entlearn/0.1.0/guide/estimators/index.md) for shared staging and named parameters, and the [regression and reconstruction guide](https://entropiclearning.github.io/entlearn/0.1.0/guide/estimators/#regression-and-reconstruction) for target shapes, missing-target rules and original-space reconstruction.

[Refit and reuse](https://entropiclearning.github.io/entlearn/0.1.0/guide/estimators/#refit-and-reuse) describes the `warm_start` modes that continue a fit, and how `clone` copies an estimator's parameters without its fitted network. [Save the entire estimator](https://entropiclearning.github.io/entlearn/0.1.0/guide/estimators/#saving-an-estimator) shows how pickle and joblib keep a fitted estimator, including its ability to resume.

## entlearn.scikit_adapter.EONClassifier

```
EONClassifier(
    recipe,
    *,
    categorical_features=None,
    class_weights=None,
    initial_state=None,
    warm_start=False,
    n_inits=1,
    n_jobs=None,
    parallel_backend="threads",
    retain="winner",
    scoring=None,
    cv=None,
    return_train_score=False,
    init_rows=None,
    predict_config=None,
    max_iter=100,
    tol=0.0001,
    random_state=None,
    device=None,
    dtype=None,
    verbose=0,
)
```

Bases: `ClassifierMixin`, `_BaseEON`

A hard-label classifier delegating its numerical lifecycle to Network.

Distributions can be supplied as labels/categorical features by using the a `Network` directly.

A `recipe` is required. Constructor values are stored verbatim and validated at fit. Numeric `-1` labels mark unlabelled rows.

`categorical_features` accepts integer column positions, `"from_dtype"` for DataFrame category/object/string columns, or `None` for continuous features. Class weights may be a label-keyed mapping or a vector in sorted class order. `dtype=None` stages directly to float64 and `device=None` selects CPU.

`random_state` is an integral seed in `[0, 2**63)` (not a bool), a NumPy RandomState or `None`. An integer is passed unchanged as the Network root seed. `initial_state` optionally supplies reusable tensor geometry and requires one initialisation. Supplying a state bypasses this estimator's selection `cv` and `scoring`, an enclosing hyperparameter search still owns its ordinary folds and scorer.

Several `n_inits` are selected in-sample when `cv=None` or out-of-sample using the provided splitter otherwise. `scoring=None` uses accuracy for selection. Strings and callables use scikit-learn scorer semantics. `groups` passed to `fit` are resolved by the splitter. Network owns candidate fitting, calibration, retention, ranking and finalisation. `return_train_score=True` also scores each `cv` training partition: exact validation-score ties, then prefer the smaller training-validation gap, then candidate order. Each outcome in `network_.diagnostics` records the scores negated as losses, and `network_.diagnostics.selected_outcome` is the winner's.

`predict_config` is the complete fit-time prediction policy. By default prediction uses the fitted policy, including its calibrated temperature, but can be replaced by supplying a call-time configuration.

`warm_start=False` starts a fresh fit. `"resume"` continues the fitted trajectory on its training rows; `max_iter` is a total iteration ceiling. `"fine_tune"` adapts fitted parameters to compatible data and starts a new loss history, using the current Recipe. To replace it, call `set_params(recipe=new_recipe)` or edit nested Recipe parameters before `fit`. Network permits shape-preserving objective changes, not topology or cluster-count changes. Both continuation modes require `n_inits=1` when fitted. Either string on an unfitted estimator emits a `UserWarning` and fits fresh, including unfitted clones in SearchCV. Boolean `True` is invalid, incompatible continuation raises without falling back or replacing the previous fitted model.

Continuation reuses fitted label/category vocabularies, allowing absent known levels but not new ones. Network owns tensor and Recipe compatibility. Initial-state, random-seed, selection and retention controls apply only to fresh fits; continuation preserves existing members and their original winner. `predict_config=None` retains fitted policy during continuation. Any other request is compared with the fit-time request after both are resolved, so an explicit geometric read-out equals the derived default, a different policy is rejected. Every failure leaves the previous fitted model usable.

Store constructor parameters verbatim for scikit-learn cloning.

Source code in `src/entlearn/scikit_adapter/classifier.py`

```
def __init__(
    self,
    recipe: Recipe,
    *,
    categorical_features: Sequence[int] | Literal["from_dtype"] | None = None,
    class_weights: object = None,
    initial_state: InitialState | None = None,
    warm_start: Literal[False, "resume", "fine_tune"] = False,
    n_inits: int = 1,
    n_jobs: int | None = None,
    parallel_backend: Literal["threads", "processes"] = "threads",
    retain: Literal["winner", "states", "members"] = "winner",
    scoring: str | Callable[..., float] | None = None,
    cv: int | BaseCrossValidator | Iterable | None = None,
    return_train_score: bool = False,
    init_rows: Sequence[int] | Callable[..., Sequence[int]] | None = None,
    predict_config: PredictConfig | None = None,
    max_iter: int = 100,
    tol: float = 1e-4,
    random_state: int | np.random.RandomState | None = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
    verbose: int = 0,
) -> None:
    """Store constructor parameters verbatim for scikit-learn cloning."""
    self.recipe = recipe
    self.categorical_features = categorical_features
    self.class_weights = class_weights
    self.initial_state = initial_state
    self.warm_start = warm_start
    self.n_inits = n_inits
    self.n_jobs = n_jobs
    self.parallel_backend = parallel_backend
    self.retain = retain
    self.scoring = scoring
    self.cv = cv
    self.return_train_score = return_train_score
    self.init_rows = init_rows
    self.predict_config = predict_config
    self.max_iter = max_iter
    self.tol = tol
    self.random_state = random_state
    self.device = device
    self.dtype = dtype
    self.verbose = verbose
```

### predict

```
predict(X, *, predict_config=None)
```

Decode the most probable class through the fitted label vocabulary.

Source code in `src/entlearn/scikit_adapter/classifier.py`

```
def predict(self, X: object, *, predict_config: PredictConfig | None = None) -> np.ndarray:
    """Decode the most probable class through the fitted label vocabulary."""
    probabilities = self.predict_proba(X, predict_config=predict_config)
    return self.classes_[probabilities.argmax(axis=1)]
```

### predict_proba

```
predict_proba(X, *, predict_config=None)
```

Return probabilities in input row order. Column j is for `classes_[j]`.

Source code in `src/entlearn/scikit_adapter/classifier.py`

```
def predict_proba(
    self, X: object, *, predict_config: PredictConfig | None = None
) -> np.ndarray:
    """Return probabilities in input row order. Column j is for ``classes_[j]``."""
    X_cont, X_cat = self._prepare_prediction_features(X)
    return (
        self.network_.predict(X_cont, X_cat=X_cat or None, predict_config=predict_config)
        .cpu()
        .numpy()
    )
```

## entlearn.scikit_adapter.EONRegressor

```
EONRegressor(
    recipe,
    *,
    categorical_features=None,
    regression_weighting=None,
    initial_state=None,
    warm_start=False,
    n_inits=1,
    n_jobs=None,
    parallel_backend="threads",
    retain="winner",
    scoring=None,
    cv=None,
    return_train_score=False,
    init_rows=None,
    predict_config=None,
    max_iter=100,
    tol=0.0001,
    random_state=None,
    device=None,
    dtype=None,
    verbose=0,
)
```

Bases: `RegressorMixin`, `_BaseEON`

A single- or multi-output regressor delegating its numerical lifecycle to Network.

A `recipe` is required. Constructor parameters are stored verbatim for cloning. Integer and floating tabular values stage directly to `dtype` (float64 by default); `device=None` selects CPU.

`regression_weighting` passes unchanged to Network as `task_weights`. Network alone invokes it on labelled tensor targets and validates its result. Fully missing target rows can be marked as unlabelled using `NaN`. Partially missing multi-output rows are rejected; use direct Network fitting instead, which warns and masks the whole row.

Prediction preserves the fit target's dimensionality and computation precision. By default queries use the fitted prediction policy, dtype and device, even after constructor changes. A call-time `PredictConfig` replaces the policy fully.

Several `n_inits` are selected in-sample when `cv=None` or out-of-sample using the provided splitter otherwise. `scoring=None` uses R² for selection. Strings and callables use scikit-learn scorer semantics. `groups` passed to `fit` are resolved by the splitter. Network owns candidate fitting, retention, ranking and finalisation. `return_train_score=True` also scores each `cv` training partition: exact validation-score ties, then prefer the smaller training-validation gap, then candidate order. Each outcome in `network_.diagnostics` records the scores negated as losses, and `network_.diagnostics.selected_outcome` is the winner's.

`warm_start=False` starts a fresh fit. `"resume"` continues the fitted trajectory on its training rows, with `max_iter` as a total iteration ceiling. `"fine_tune"` adapts to compatible data using the current Recipe and starts new operation diagnostics. To supply a replacement, call `set_params(recipe=new_recipe)` or edit nested Recipe parameters before `fit`. Network permits shape-preserving objective changes, not topology or cluster-count changes. Both continuation modes require `n_inits=1` when fitted. Either string on an unfitted estimator emits a `UserWarning` and fits fresh, including SearchCV clones. Boolean `True` is invalid, incompatible continuation raises without falling back or replacing the previous fitted model.

Continuation preserves fitted category vocabularies, retained members and their original winner. Initial-state, random-seed, selection and retention controls apply only to fresh fits. `predict_config=None` retains fitted policy during continuation. Any other request is compared with the fit-time request after both are resolved, a different policy is rejected. Every failure leaves the previous fitted model usable.

Store constructor parameters verbatim for scikit-learn cloning.

Source code in `src/entlearn/scikit_adapter/regressor.py`

```
def __init__(
    self,
    recipe: Recipe,
    *,
    categorical_features: Sequence[int] | Literal["from_dtype"] | None = None,
    regression_weighting: Callable[[torch.Tensor], torch.Tensor] | None = None,
    initial_state: InitialState | None = None,
    warm_start: Literal[False, "resume", "fine_tune"] = False,
    n_inits: int = 1,
    n_jobs: int | None = None,
    parallel_backend: Literal["threads", "processes"] = "threads",
    retain: Literal["winner", "states", "members"] = "winner",
    scoring: str | Callable[..., float] | None = None,
    cv: int | BaseCrossValidator | Iterable | None = None,
    return_train_score: bool = False,
    init_rows: Sequence[int] | Callable[..., Sequence[int]] | None = None,
    predict_config: PredictConfig | None = None,
    max_iter: int = 100,
    tol: float = 1e-4,
    random_state: int | np.random.RandomState | None = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
    verbose: int = 0,
) -> None:
    """Store constructor parameters verbatim for scikit-learn cloning."""
    self.recipe = recipe
    self.categorical_features = categorical_features
    self.regression_weighting = regression_weighting
    self.initial_state = initial_state
    self.warm_start = warm_start
    self.n_inits = n_inits
    self.n_jobs = n_jobs
    self.parallel_backend = parallel_backend
    self.retain = retain
    self.scoring = scoring
    self.cv = cv
    self.return_train_score = return_train_score
    self.init_rows = init_rows
    self.predict_config = predict_config
    self.max_iter = max_iter
    self.tol = tol
    self.random_state = random_state
    self.device = device
    self.dtype = dtype
    self.verbose = verbose
```

### n_outputs\_

```
n_outputs_
```

Return the fitted number of regression targets without duplicating Network state.

### __sklearn_tags__

```
__sklearn_tags__()
```

Declare support for vector and multi-output targets.

Source code in `src/entlearn/scikit_adapter/regressor.py`

```
def __sklearn_tags__(self) -> Tags:
    """Declare support for vector and multi-output targets."""
    tags = super().__sklearn_tags__()
    tags.target_tags.multi_output = True
    return tags
```

### predict

```
predict(X, *, predict_config=None)
```

Return predictions with the response rank established by the initial fit.

Source code in `src/entlearn/scikit_adapter/regressor.py`

```
def predict(self, X: object, *, predict_config: PredictConfig | None = None) -> np.ndarray:
    """Return predictions with the response rank established by the initial fit."""
    X_cont, X_cat = self._prepare_prediction_features(X)
    result = (
        self.network_.predict(X_cont, X_cat=X_cat or None, predict_config=predict_config)
        .cpu()
        .numpy()
    )
    return result[:, 0] if self._y_was_1d else result
```

Both estimators share one `fit` method:

## EONClassifier.fit and EONRegressor.fit

```
fit(X, y, sample_weight=None, *, groups=None)
```

Stage features and targets, then publish results if Network.fit succeeds.

A supplied `initial_state` is fitted directly, without initialisation selection: the estimator's `cv` splitter is not traversed and its `scoring` is not used, although an invalid `scoring` still raises. An enclosing search keeps its own folds and scorer.

Parameters:

| Name            | Type     | Description                                                                                                                                                               | Default    |
| --------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `X`             | `object` | Dense feature matrix or eager DataFrame, with instances in rows.                                                                                                          | *required* |
| `y`             | `object` | For EONClassifier, hard class labels, with numeric -1 marking unlabelled rows. For EONRegressor, real targets shaped (T,) or (T, M), where fully NaN rows are unlabelled. | *required* |
| `sample_weight` | `object` | Optional positive row weights.                                                                                                                                            | `None`     |
| `groups`        | `object` | Optional group labels passed to candidate-selection splitting.                                                                                                            | `None`     |

Returns:

| Type   | Description            |
| ------ | ---------------------- |
| `Self` | This fitted estimator. |

Raises:

| Type         | Description                                       |
| ------------ | ------------------------------------------------- |
| `ValueError` | If the Recipe, data or controls are incompatible. |

Source code in `src/entlearn/scikit_adapter/base.py`

```
def fit(
    self, X: object, y: object, sample_weight: object = None, *, groups: object = None
) -> Self:
    """Stage features and targets, then publish results if Network.fit succeeds.

    A supplied ``initial_state`` is fitted directly, without initialisation selection:
    the estimator's ``cv`` splitter is not traversed and its ``scoring`` is not used,
    although an invalid ``scoring`` still raises. An enclosing search keeps its own
    folds and scorer.

    Args:
        X: Dense feature matrix or eager DataFrame, with instances in rows.
        y: For ``EONClassifier``, hard class labels, with numeric -1 marking
            unlabelled rows. For ``EONRegressor``, real targets shaped ``(T,)`` or
            ``(T, M)``, where fully NaN rows are unlabelled.
        sample_weight: Optional positive row weights.
        groups: Optional group labels passed to candidate-selection splitting.

    Returns:
        This fitted estimator.

    Raises:
        ValueError: If the Recipe, data or controls are incompatible.
    """
    if type(self.recipe) is not Recipe:
        raise ValueError("recipe must be a valid Recipe")
    classifier = is_classifier(self)
    head_type = ClassificationHead if classifier else RegressionHead
    if not any(isinstance(block, head_type) for block in self.recipe.blocks):
        task = "classification" if classifier else "regression"
        raise ValueError(f"{type(self).__name__} requires a {task} Recipe")
    mode = self._continuation_mode()
    device, dtype = _resolve_device_dtype(self.device, self.dtype)
    X_checked, X_cont, X_cat, metadata = self._stage_fit_features(
        X, continuing=mode is not None, dtype=dtype, device=device
    )
    staged = self._stage_target(y, X, continuing=mode is not None, dtype=dtype, device=device)
    sample_weights = _prepare_sample_weights(X, sample_weight, dtype=dtype, device=device)
    if mode is not None:
        network = self._continue_network(
            mode,
            X_cont,
            staged.tensor,
            X_cat=X_cat,
            sample_weights=sample_weights,
            **staged.weights,
        )
        self.network_ = network
        return self
    seed = _random_seed(self.random_state)
    pairs = _materialise_splits(
        self.cv if self.initial_state is None else None,
        X_checked,
        staged.target,
        groups,
        classifier=classifier,
        device=device,
    )
    init_rows = _resolve_init_rows(self.init_rows, pairs, device=device)
    network = Network.fit(
        self.recipe,
        X_cont,
        staged.tensor,
        X_cat=X_cat or None,
        sample_weights=sample_weights,
        **staged.weights,
        initial_state=self.initial_state,
        n_inits=self.n_inits,
        n_jobs=self.n_jobs,
        parallel_backend=self.parallel_backend,
        retain=self.retain,
        selection_loss=staged.selection_loss(
            X_checked, self.scoring, n_inits=self.n_inits, cv=self.cv
        ),
        validation_pairs=pairs,
        return_train_score=self.return_train_score,
        init_rows=init_rows,
        computation_dtype=dtype,
        predict_config=self.predict_config,
        max_iter=self.max_iter,
        tol=self.tol,
        seed=seed,
        verbose=self.verbose,
    )
    # Old optional feature names must disappear when a fresh fit uses an unnamed matrix.
    self.__dict__.pop("feature_names_in_", None)
    self.__dict__.update(
        metadata, **staged.published, network_=network, _fit_predict_config=self.predict_config
    )
    return self
```

Both estimators record their fitted column layout as `feature_layout_`, listed with the other [fitted attributes](https://entropiclearning.github.io/entlearn/0.1.0/guide/estimators/#labels-scores-and-fitted-attributes):

## entlearn.scikit_adapter.FeatureLayout

```
FeatureLayout(
    continuous_indices,
    categorical_indices,
    dropped_indices,
    categories,
    dropped_categories,
)
```

The fitted column split and categorical vocabularies.

Every original column appears in exactly one position tuple, each sorted in original feature order: `continuous_indices`, kept `categorical_indices` with their sorted level vocabularies in `categories`, or `dropped_indices` for single-level categorical columns with their one-level vocabularies in `dropped_categories`. `tensor_positions` gives the original position of each Network feature: continuous positions followed by kept categorical positions. Dropped columns never reach the Network: `n_columns` still counts them, so reports can restore the original feature order.

### n_columns

```
n_columns
```

Return the original column count, dropped columns included.

### tensor_positions

```
tensor_positions
```

Return the original column position of each Network feature, in tensor order.

### as_tabular

```
as_tabular(result, feature_names)
```

Convert values and attach fitted tabular meanings.

Source code in `src/entlearn/scikit_adapter/features.py`

```
def as_tabular(
    self, result: ReconstructionResult, feature_names: tuple[str, ...] | None
) -> TabularReconstruction:
    """Convert values and attach fitted tabular meanings."""
    continuous = result.continuous.cpu().numpy()
    columns = {
        position: (probability.cpu().numpy(), levels.copy())
        for position, probability, levels in zip(
            self.categorical_indices, result.categorical, self.categories, strict=True
        )
    }
    for position, levels in zip(self.dropped_indices, self.dropped_categories, strict=True):
        columns[position] = (
            np.ones((continuous.shape[0], 1), dtype=continuous.dtype),
            levels.copy(),
        )
    indices = tuple(sorted(columns))
    return TabularReconstruction(
        continuous=continuous,
        categorical=tuple(columns[index][0] for index in indices),
        continuous_indices=self.continuous_indices,
        categorical_indices=indices,
        categories=tuple(columns[index][1] for index in indices),
        dropped_indices=self.dropped_indices,
        feature_names=feature_names,
    )
```

[Select with validation folds](https://entropiclearning.github.io/entlearn/0.1.0/guide/model_selection/#select-with-validation-folds) shows how `common_train_rows` keeps validation rows out of the initialisation.

## entlearn.scikit_adapter.common_train_rows

```
common_train_rows(folds)
```

Return the sorted intersection of every training partition.

An ordinary KFold partition has an empty intersection; expanding-window time series folds often share an initial training core. The helper returns an empty array when there is no common row. Fitting rejects that selection before sampling. This only addresses geometry leakage, not external preprocessing.

Parameters:

| Name    | Type                                | Description                                                    | Default    |
| ------- | ----------------------------------- | -------------------------------------------------------------- | ---------- |
| `folds` | `Sequence[tuple[ndarray, ndarray]]` | Non-empty sequence of training/validation integer-index pairs. | *required* |

Returns:

| Type      | Description                                                 |
| --------- | ----------------------------------------------------------- |
| `ndarray` | One-dimensional int64 array of common training row indices. |

Raises:

| Type         | Description                                                 |
| ------------ | ----------------------------------------------------------- |
| `ValueError` | If no folds are supplied or training indices are malformed. |

Source code in `src/entlearn/scikit_adapter/selection.py`

```
def common_train_rows(folds: Sequence[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Return the sorted intersection of every training partition.

    An ordinary KFold partition has an empty intersection; expanding-window time
    series folds often share an initial training core. The helper returns an empty
    array when there is no common row. Fitting rejects that selection before sampling.
    This only addresses geometry leakage, not external preprocessing.

    Args:
        folds: Non-empty sequence of training/validation integer-index pairs.

    Returns:
        One-dimensional int64 array of common training row indices.

    Raises:
        ValueError: If no folds are supplied or training indices are malformed.
    """
    if not folds:
        raise ValueError("common_train_rows needs at least one fold")
    core = None
    for training, _ in folds:
        indices = np.asarray(training)
        if indices.ndim != 1 or indices.dtype.kind not in "iu":
            raise ValueError("training indices must be one-dimensional integer arrays")
        core = np.unique(indices) if core is None else np.intersect1d(core, indices)
    return np.asarray(core, dtype=np.int64)
```

Both estimators inherit the same fitted queries:

## entlearn.scikit_adapter.EONClassifier.reconstruct

```
reconstruct(X, *, predict_config=None)
```

Return Network reconstruction with original tabular feature meanings.

Standard input blends fitted centroids, manifold input blends projections onto the clusters' planes. Both use final input affiliations under the fitted prediction policy, or a complete call-time replacement. Supply rows in the same units and column order used at fit. Every layout returns a `TabularReconstruction` of fresh arrays. A continuous-only layout has empty categorical fields. Numerical values retain fitted computation precision. Categorical mixtures retain their vocabularies and are never argmax-decoded.

Source code in `src/entlearn/scikit_adapter/base.py`

```
def reconstruct(
    self, X: object, *, predict_config: PredictConfig | None = None
) -> TabularReconstruction:
    """Return Network reconstruction with original tabular feature meanings.

    Standard input blends fitted centroids, manifold input blends projections onto
    the clusters' planes. Both use final input affiliations under the fitted
    prediction policy, or a complete call-time replacement.
    Supply rows in the same units and column order used at fit. Every layout
    returns a ``TabularReconstruction`` of fresh arrays. A continuous-only layout
    has empty categorical fields. Numerical values retain fitted computation
    precision. Categorical mixtures retain their vocabularies and are never
    argmax-decoded.
    """
    continuous, categorical = self._prepare_prediction_features(X)
    result = self.network_.reconstruct(
        continuous, X_cat=categorical, predict_config=predict_config
    )
    names = getattr(self, "feature_names_in_", None)
    return self.feature_layout_.as_tabular(result, None if names is None else tuple(names))
```

## entlearn.scikit_adapter.TabularReconstruction

```
TabularReconstruction(
    continuous,
    categorical,
    continuous_indices,
    categorical_indices,
    categories,
    dropped_indices,
    feature_names,
)
```

Reconstructed values with original feature positions and categorical meanings.

`continuous` columns correspond to `continuous_indices`. Each matrix in `categorical` contains probabilities over its matching `categories` vector, and occupies the matching original `categorical_indices` position. Both position sequences are sorted in original feature order.

A continuous-only layout has empty `categorical`, `categorical_indices` and `categories`. Dropped single-level columns have one-column distributions of ones over their fitted singleton vocabulary. `dropped_indices` distinguishes these constants from model reconstructions. Query values in dropped columns are ignored. `feature_names` contains fitted names when available, otherwise `None`.

## entlearn.scikit_adapter.EONClassifier.recover_instance_weights

```
recover_instance_weights(X, *, predict_config=None)
```

Return query-time recovered instance weights.

Values are relative to the retained training normaliser, not normalised across this query batch or automatically comparable across refits. Omission uses fitted policy, a complete configuration replaces it. A fit without instance-weight recovery raises.

Source code in `src/entlearn/scikit_adapter/base.py`

```
def recover_instance_weights(
    self, X: object, *, predict_config: PredictConfig | None = None
) -> np.ndarray:
    """Return query-time recovered instance weights.

    Values are relative to the retained training normaliser, not normalised
    across this query batch or automatically comparable across refits.
    Omission uses fitted policy, a complete configuration replaces it.
    A fit without instance-weight recovery raises.
    """
    continuous, categorical = self._prepare_prediction_features(X)
    result = self.network_.predict_with_details(
        continuous,
        X_cat=categorical,
        predict_config=predict_config,
        details=("instance_weights",),
    )
    assert result.instance_weights is not None
    return result.instance_weights.cpu().numpy()
```

## entlearn.scikit_adapter.EONClassifier.score_samples

```
score_samples(X, *, predict_config=None)
```

Rank recovered query weights against the fitted training reference instance weights.

Higher values are more typical. A complete override replaces query policy but not the fitted reference. Cross-policy scores therefore compare different recovery modes. A fit without instance-weight recovery raises through Network.

Source code in `src/entlearn/scikit_adapter/base.py`

```
def score_samples(
    self, X: object, *, predict_config: PredictConfig | None = None
) -> np.ndarray:
    """Rank recovered query weights against the fitted training reference instance weights.

    Higher values are more typical.
    A complete override replaces query policy but not the fitted reference.
    Cross-policy scores therefore compare different recovery modes.
    A fit without instance-weight recovery raises through Network.
    """
    continuous, categorical = self._prepare_prediction_features(X)
    return (
        self.network_.score_samples(
            continuous, X_cat=categorical, predict_config=predict_config
        )
        .cpu()
        .numpy()
    )
```

## entlearn.scikit_adapter.EONClassifier.effective_dimensions

```
effective_dimensions(*, normalise=True)
```

Return fitted distribution spreads under stable block and field names.

Training-dependent reports require retained row-bound state. Values are computed by the state-free Network reporting helper, never cached here.

Source code in `src/entlearn/scikit_adapter/base.py`

```
def effective_dimensions(self, *, normalise: bool = True) -> dict[str, dict[str, float]]:
    """Return fitted distribution spreads under stable block and field names.

    Training-dependent reports require retained row-bound state. Values are
    computed by the state-free Network reporting helper, never cached here.
    """
    check_is_fitted(self, "network_")
    return {
        name: {field: float(value) for field, value in values.items()}
        for name, values in effective_dimensions(self.network_, normalise=normalise).items()
    }
```

## entlearn.scikit_adapter.EONClassifier.feature_importances\_

```
feature_importances_
```

Return a feature importance vector in original column order, with zeros for dropped columns.

Standard input reports fitted feature weights. Manifold input reports instance-weighted tangent participation, and requires retained training affiliations and instance weights.

## entlearn.scikit_adapter.EONClassifier.active_features

```
active_features(*, tol=1.0)
```

Return sorted int64 active-feature indices in original column order.

Dropped columns are excluded. Activity uses standard-input feature weights strictly above `tol / D`, where `D` is the retained feature count. Manifold inputs and uniform feature weights have no active-feature report.

Source code in `src/entlearn/scikit_adapter/base.py`

```
def active_features(self, *, tol: float = 1.0) -> np.ndarray:
    """Return sorted int64 active-feature indices in original column order.

    Dropped columns are excluded. Activity uses standard-input feature weights
    strictly above ``tol / D``, where ``D`` is the retained feature count.
    Manifold inputs and uniform feature weights have no active-feature report.
    """
    check_is_fitted(self, "network_")
    indices = active_features(self.network_, tol=tol).cpu().numpy()
    positions = np.asarray(self.feature_layout_.tensor_positions, dtype=np.int64)
    return np.sort(positions[indices])
```

## entlearn.scikit_adapter.EONClassifier.count_parameters

```
count_parameters(
    *,
    include_affiliations=False,
    raw=False,
    active_tol=None,
)
```

Count fitted Network parameters using the tensor reporting rules.

`raw` includes simplex-constrained entries; `include_affiliations` includes row-bound learned values and requires them to be retained. `active_tol` restricts counting to active standard-input features.

Source code in `src/entlearn/scikit_adapter/base.py`

```
def count_parameters(
    self,
    *,
    include_affiliations: bool = False,
    raw: bool = False,
    active_tol: float | None = None,
) -> int:
    """Count fitted Network parameters using the tensor reporting rules.

    ``raw`` includes simplex-constrained entries; ``include_affiliations``
    includes row-bound learned values and requires them to be retained.
    ``active_tol`` restricts counting to active standard-input features.
    """
    check_is_fitted(self, "network_")
    return count_parameters(
        self.network_,
        include_affiliations=include_affiliations,
        raw=raw,
        active_tol=active_tol,
    )
```

These methods are inherited by `EONRegressor`. Estimator methods own tabular conversion and original-column mapping; the [state-free reporting helpers](https://entropiclearning.github.io/entlearn/0.1.0/reference/helpers/index.md) accept only a fitted Network.
