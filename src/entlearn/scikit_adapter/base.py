"""The shared estimator fit lifecycle, tabular staging and fitted-state access."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import partial
from numbers import Integral
from types import SimpleNamespace
from typing import Any, Literal, Self

import numpy as np
import torch
from sklearn.base import BaseEstimator, is_classifier
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils import check_random_state
from sklearn.utils.validation import (
    _check_feature_names,
    _check_n_features,
    _check_sample_weight,
    check_is_fitted,
)

from entlearn import (
    ClassificationHead,
    Head,
    InitialState,
    Network,
    PredictConfig,
    Recipe,
    RegressionHead,
)
from entlearn._seeds import _SEED_UPPER_BOUND
from entlearn._warnings import _warn
from entlearn.helpers.reporting import (
    active_features,
    count_parameters,
    effective_dimensions,
    feature_importances,
)
from entlearn.network.config import _resolve_predict_config
from entlearn.network.validation import _stage_sample_weights
from entlearn.scikit_adapter.features import (
    FeatureLayout,
    TabularReconstruction,
    _apply_feature_layout,
    _check_tabular_input,
    _feature_tensors,
    _fit_feature_layout,
    _resolve_categorical_indices,
    _resolve_device_dtype,
    _staging_dtype,
)
from entlearn.scikit_adapter.parameters import _recipe_parameters, _replace_recipe_parameters
from entlearn.scikit_adapter.selection import (
    _materialise_splits,
    _resolve_init_rows,
    _ScorerSelectionLoss,
)

_RANDOM_STATE_DRAW_HIGH = (1 << 31) - 1


def _prepare_sample_weights(
    X: object, sample_weight: object, *, dtype: torch.dtype, device: torch.device
) -> torch.Tensor | None:
    """Validate optional tabular row weights and stage them in the computation precision."""
    if sample_weight is None:
        return None
    weights = _check_sample_weight(
        sample_weight, X, dtype=_staging_dtype(dtype), allow_all_zero_weights=True
    )
    staged, _ = _stage_sample_weights(
        torch.as_tensor(weights, dtype=dtype, device=device),
        length=len(weights),
        dtype=dtype,
        device=device,
    )
    return staged


def _random_seed(random_state: object) -> int:
    """Translate scikit-learn random-state to one Network root seed."""
    if isinstance(random_state, (bool, np.bool_)):
        raise ValueError("random_state must not be a boolean")
    if isinstance(random_state, Integral):
        seed = int(random_state)
        if not 0 <= seed < _SEED_UPPER_BOUND:
            raise ValueError("random_state must be an integer in [0, 2**63)")
        return seed
    return int(check_random_state(random_state).randint(0, _RANDOM_STATE_DRAW_HIGH))


@dataclass(frozen=True)
class _StagedTarget:
    """An estimator task's staged target and the values its fit forwards.

    ``target`` is the array splitters and scorers see, and ``tensor`` its Network
    form. ``weights`` holds the task's class or task weights, passed unchanged to the
    Network fit or continuation. ``selection_loss`` builds the scorer callback
    from the checked features, and ``published`` holds the task attributes a fresh fit
    publishes.
    """

    target: np.ndarray
    tensor: torch.Tensor
    weights: dict[str, Any]
    selection_loss: Callable[..., _ScorerSelectionLoss | None]
    published: dict[str, Any]


class _BaseEON(BaseEstimator):
    """Abstraction to share the fit lifecycle, Recipe parameters and fitted queries between estimator tasks.

    Each task supplies ``_stage_target``.
    """

    recipe: Recipe
    network_: Network
    feature_layout_: FeatureLayout
    n_features_in_: int
    feature_names_in_: np.ndarray

    categorical_features: Sequence[int] | Literal["from_dtype"] | None
    initial_state: InitialState | None
    warm_start: Literal[False, "resume", "fine_tune"]
    n_inits: int
    n_jobs: int | None
    parallel_backend: Literal["threads", "processes"]
    retain: Literal["winner", "states", "members"]
    scoring: str | Callable[..., float] | None
    cv: int | BaseCrossValidator | Iterable | None
    return_train_score: bool
    init_rows: Sequence[int] | Callable[..., Sequence[int]] | None
    predict_config: PredictConfig | None
    max_iter: int
    tol: float
    random_state: int | np.random.RandomState | None
    device: str | torch.device | None
    dtype: torch.dtype | None
    verbose: int
    _fit_predict_config: PredictConfig | None

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return constructor parameters and, when requested, every named Recipe field."""
        params = super().get_params(deep=deep)
        if deep and type(self.recipe) is Recipe:
            params.update(_recipe_parameters(self.recipe))
        return params

    def set_params(self, **params: Any) -> Self:
        """Apply whole-Recipe replacement first, then all nested edits atomically.

        Names and endpoints are ordinary fields. A rename therefore supplies the
        corresponding endpoint changes in the same call. Only the final graph is
        validated.

        Raises:
            ValueError: If a path is missing, incompatible or makes the Recipe invalid.
        """
        valid = self.get_params(deep=False)
        plain = {key: value for key, value in params.items() if "__" not in key}
        nested = {key: value for key, value in params.items() if "__" in key}
        unknown = plain.keys() - valid.keys()
        if unknown:
            raise ValueError(f"Invalid parameter(s) {sorted(unknown)} for {type(self).__name__}")
        recipe = plain.get("recipe", self.recipe)
        if nested:
            if type(recipe) is not Recipe:
                raise ValueError("nested parameters require a non-null Recipe")
            plain["recipe"] = _replace_recipe_parameters(recipe, nested)
        self.__dict__.update(plain)
        return self

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

    @abstractmethod
    def _stage_target(
        self, y: object, X: object, *, continuing: bool, dtype: torch.dtype, device: torch.device
    ) -> _StagedTarget:
        """Validate and stage the task's target."""

    def _continuation_mode(self) -> Literal["resume", "fine_tune"] | None:
        """Resolve the explicit lifecycle."""
        if self.warm_start is False:
            return None
        if not isinstance(self.warm_start, str) or self.warm_start not in ("resume", "fine_tune"):
            raise ValueError(
                f"warm_start={self.warm_start!r} is not a supported mode. Use 'resume' "
                "to continue the existing fit on its training rows, 'fine_tune' to adapt "
                "fitted parameters to compatible data, or False to start a fresh fit."
            )
        if not hasattr(self, "network_"):
            _warn(
                f"warm_start={self.warm_start!r} was requested, but this estimator has no "
                "fitted model. Performing a fresh fit; subsequent calls will use "
                f"{self.warm_start!r}. Use warm_start=False to request fresh fitting explicitly.",
                UserWarning,
            )
            return None
        if (
            isinstance(self.n_inits, (bool, np.bool_))
            or not isinstance(self.n_inits, Integral)
            or self.n_inits != 1
        ):
            raise ValueError(
                "Continuing a fitted estimator requires n_inits=1. Set n_inits=1 to "
                "continue its retained trajectories, or warm_start=False for a fresh fit with initialisation selection."
            )
        if self.warm_start == "resume" and self.recipe != self.network_.recipe:
            raise ValueError(
                "The estimator Recipe has changed since fitting. Use warm_start='fine_tune' "
                "for compatible objective changes, or warm_start=False for a fresh fit."
            )
        if self.predict_config is not None:
            # Compare requests, not fitted policies, so epsilon_P provenance survives.
            head = next(block for block in self.network_.recipe.blocks if isinstance(block, Head))
            requested = _resolve_predict_config(self.predict_config, head)
            if requested != _resolve_predict_config(self._fit_predict_config, head):
                raise ValueError(
                    "predict_config cannot change the fitted prediction policy or its "
                    "supplied/derived temperature choice during "
                    "continuation. Use None to retain it, pass an override to a prediction "
                    "call, or use warm_start=False for a fresh fit."
                )
        return self.warm_start

    def _continue_network(
        self,
        mode: Literal["resume", "fine_tune"],
        X_cont: torch.Tensor,
        y: torch.Tensor,
        *,
        X_cat: tuple[torch.Tensor, ...],
        sample_weights: torch.Tensor | None,
        class_weights: torch.Tensor | None = None,
        task_weights: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> Network:
        """Request one complete continuation."""
        operation = (
            self.network_.resume
            if mode == "resume"
            else partial(self.network_.fine_tune, recipe=self.recipe)
        )
        return operation(
            X_cont,
            y,
            X_cat=X_cat or None,
            sample_weights=sample_weights,
            class_weights=class_weights,
            task_weights=task_weights,
            computation_dtype=X_cont.dtype,
            max_iter=self.max_iter,
            tol=self.tol,
            verbose=self.verbose,
        )

    def _stage_fit_features(
        self, X: object, *, continuing: bool, dtype: torch.dtype, device: torch.device
    ) -> tuple[object, torch.Tensor, tuple[torch.Tensor, ...], dict[str, object]]:
        """Learn a fresh layout or replay fitted feature meanings."""
        X = _check_tabular_input(X)
        if continuing:
            _check_feature_names(self, X, reset=False)
            _check_n_features(self, X, reset=False)
            layout = self.feature_layout_
            expected = [i for i in range(layout.n_columns) if i not in layout.continuous_indices]
            if _resolve_categorical_indices(X, self.categorical_features) != expected:
                raise ValueError(
                    "categorical_features must keep the fitted column meanings during "
                    "continuation. Restore the fitted layout or use warm_start=False for a "
                    "fresh fit."
                )
            X_cont, codes = _apply_feature_layout(
                X, layout, _staging_dtype(dtype), check_dropped=True
            )
            metadata = {}
        else:
            fitted = SimpleNamespace()
            _check_n_features(fitted, X, reset=True)
            _check_feature_names(fitted, X, reset=True)
            layout, X_cont, codes = _fit_feature_layout(
                X, self.categorical_features, _staging_dtype(dtype)
            )
            if not layout.continuous_indices and not layout.categorical_indices:
                raise ValueError(
                    "X has no usable features after dropping single-level categorical columns"
                )
            metadata = {**vars(fitted), "feature_layout_": layout}
        continuous, categorical = _feature_tensors(X_cont, codes, device=device, dtype=dtype)
        return X, continuous, categorical, metadata

    def _prepare_prediction_features(
        self, X: object
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        check_is_fitted(self, "network_")
        X = _check_tabular_input(X)
        _check_feature_names(self, X, reset=False)
        _check_n_features(self, X, reset=False)
        dtype = self.network_.schema.computation_dtype
        X_cont, codes = _apply_feature_layout(X, self.feature_layout_, _staging_dtype(dtype))
        return _feature_tensors(X_cont, codes, device=self.network_.device, dtype=dtype)

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

    @property
    def feature_importances_(self) -> np.ndarray:
        """Return a feature importance vector in original column order, with zeros for dropped columns.

        Standard input reports fitted feature weights. Manifold input reports
        instance-weighted tangent participation, and
        requires retained training affiliations and instance weights.
        """
        check_is_fitted(self, "network_")
        values = feature_importances(self.network_).cpu().numpy()
        result = np.zeros(self.n_features_in_, dtype=values.dtype)
        result[list(self.feature_layout_.tensor_positions)] = values
        return result

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

    @property
    def loss_curve_(self) -> tuple[float, ...]:
        """Return the fitted Network's loss history."""
        check_is_fitted(self, "network_")
        return self.network_.diagnostics.loss_history

    @property
    def n_iter_(self) -> int:
        """Return the fitted Network's iteration count."""
        check_is_fitted(self, "network_")
        return self.network_.diagnostics.n_iter
