"""Hard-label classification through the public tensor lifecycle."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import partial
from numbers import Integral
from typing import Literal

import numpy as np
import torch
from sklearn.base import ClassifierMixin
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_consistent_length, column_or_1d

from entlearn import ClassificationHead, InitialState, PredictConfig, Recipe
from entlearn.scikit_adapter.base import _BaseEON, _StagedTarget
from entlearn.scikit_adapter.selection import _classification_selection_loss


def _prepare_classification_y(
    y: object, *, classes: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Learn or replay labelled classes and int64 codes, keeping numeric -1 unlabelled."""
    if y is None:
        raise ValueError("requires y to be passed, but the target y is None")
    # Freeze native label values before sklearn's dataframe validation: nullable
    # integer columns otherwise pass through float64 and lose integers above 2**53.
    arr = column_or_1d(np.asarray(y), warn=True)
    # Older dataframe versions expose nullable integers as an object array. Infer
    # an exact integer dtype from its values, never via a floating intermediary.
    if arr.dtype == object and arr.size and all(isinstance(value, Integral) for value in arr):
        lower, upper = min(arr), max(arr)
        dtype = np.int64 if lower < 0 or upper <= np.iinfo(np.int64).max else np.uint64
        bounds = np.iinfo(dtype)
        if lower < bounds.min or upper > bounds.max:
            raise ValueError("integer labels must fit in int64 or uint64")
        arr = np.asarray(arr, dtype=dtype)
    check_classification_targets(arr)
    is_numeric = np.issubdtype(arr.dtype, np.number)
    labelled_mask = arr != -1 if is_numeric else np.ones(arr.shape[0], dtype=bool)
    if not labelled_mask.any():
        raise ValueError("y has no labelled instances; at least one labelled instance is required")
    observed, codes = np.unique(arr[labelled_mask], return_inverse=True)
    if classes is None:
        classes = observed
        if classes.shape[0] < 2:
            raise ValueError(
                f"y contains only {classes.shape[0]} class(es); at least 2 labelled classes are required"
            )
    else:
        # Compare labels directly: casting to the fitted dtype could truncate strings
        # or round large integers into a different, apparently known label.
        mapping = {label: index for index, label in enumerate(classes)}
        unknown = [label for label in observed if label not in mapping]
        if unknown:
            raise ValueError(
                f"y contains labels unseen at fit time: {unknown}. Continuation must use "
                "the fitted classes; use warm_start=False to learn a new label vocabulary."
            )
        codes = np.asarray([mapping[label] for label in observed], dtype=np.int64)[codes]
    y_codes = np.full(arr.shape[0], -1, dtype=np.int64)
    y_codes[labelled_mask] = codes.astype(np.int64, copy=False)
    return classes, y_codes


def _prepare_class_weights(
    weights: object, classes: np.ndarray, *, dtype: torch.dtype, device: torch.device
) -> torch.Tensor | None:
    """Order label-keyed weights and stage a tensor; Network owns weight mathematics."""
    if weights is None:
        return None
    if isinstance(weights, Mapping):
        labels = list(classes)
        missing = [label for label in labels if label not in weights]
        unknown = [label for label in weights if label not in labels]
        if missing:
            raise ValueError(f"class_weights is missing a weight for class(es) {missing}")
        if unknown:
            raise ValueError(f"class_weights has weight(s) for unknown class(es) {unknown}")
        weights = [weights[label] for label in labels]
    return torch.as_tensor(weights, dtype=dtype, device=device)


class EONClassifier(ClassifierMixin, _BaseEON):
    """A hard-label classifier delegating its numerical lifecycle to Network.

    Distributions can be supplied as labels/categorical features by using the a ``Network`` directly.

    A ``recipe`` is required. Constructor values are stored verbatim and validated at
    fit. Numeric ``-1`` labels mark unlabelled rows.

    ``categorical_features`` accepts integer column positions, ``"from_dtype"`` for
    DataFrame category/object/string columns, or ``None`` for continuous features.
    Class weights may be a label-keyed mapping or a vector in sorted class order.
    ``dtype=None`` stages directly to float64 and ``device=None`` selects CPU.

    ``random_state`` is an integral seed in ``[0, 2**63)`` (not a bool), a NumPy
    RandomState or ``None``. An integer is passed unchanged as the Network root seed.
    ``initial_state`` optionally supplies reusable tensor geometry and requires one initialisation.
    Supplying a state bypasses this estimator's selection ``cv`` and ``scoring``,
    an enclosing hyperparameter search still owns its ordinary folds and scorer.

    Several ``n_inits`` are selected in-sample when ``cv=None`` or out-of-sample using the
    provided splitter otherwise. ``scoring=None`` uses accuracy for selection.
    Strings and callables use scikit-learn scorer semantics. ``groups`` passed to
    ``fit`` are resolved by the splitter. Network owns candidate fitting, calibration,
    retention, ranking and finalisation. ``return_train_score=True`` also scores each
    ``cv`` training partition: exact validation-score ties, then
    prefer the smaller training-validation gap, then candidate order. Each outcome in
    ``network_.diagnostics`` records the scores negated as losses, and
    ``network_.diagnostics.selected_outcome`` is the winner's.

    ``predict_config`` is the complete fit-time prediction policy. By default prediction
    uses the fitted policy, including its calibrated temperature, but can be replaced by supplying a call-time configuration.

    ``warm_start=False`` starts a fresh fit. ``"resume"`` continues the fitted
    trajectory on its training rows; ``max_iter`` is a total iteration ceiling.
    ``"fine_tune"`` adapts fitted parameters to compatible data and starts a new
    loss history, using the current Recipe. To replace it, call
    ``set_params(recipe=new_recipe)`` or edit nested Recipe parameters before ``fit``.
    Network permits shape-preserving objective changes, not topology or cluster-count
    changes. Both continuation modes require ``n_inits=1`` when fitted.
    Either string on an unfitted estimator emits a ``UserWarning`` and fits fresh,
    including unfitted clones in SearchCV. Boolean ``True`` is invalid, incompatible continuation
    raises without falling back or replacing the previous fitted model.

    Continuation reuses fitted label/category vocabularies, allowing absent known
    levels but not new ones. Network owns tensor and Recipe compatibility.
    Initial-state, random-seed, selection and retention controls apply only to
    fresh fits; continuation preserves existing members and their original winner.
    ``predict_config=None`` retains fitted policy during continuation. Any other
    request is compared with the fit-time request after both are resolved, so an
    explicit geometric read-out equals the derived default, a different policy is
    rejected. Every failure leaves the previous fitted model usable.
    """

    classes_: np.ndarray

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

    def _stage_target(
        self, y: object, X: object, *, continuing: bool, dtype: torch.dtype, device: torch.device
    ) -> _StagedTarget:
        """Learn or replay the label vocabulary, and order class weights by it."""
        classes, codes = _prepare_classification_y(y, classes=self.classes_ if continuing else None)
        check_consistent_length(X, codes)
        heads = [block for block in self.recipe.blocks if isinstance(block, ClassificationHead)]
        if any(head.n_classes not in (None, len(classes)) for head in heads):
            raise ValueError("the declared n_classes must match the fitted label vocabulary")
        class_weights = _prepare_class_weights(
            self.class_weights, classes, dtype=dtype, device=device
        )
        return _StagedTarget(
            target=codes,
            tensor=torch.as_tensor(codes, dtype=torch.int64, device=device),
            weights={"class_weights": class_weights},
            selection_loss=partial(_classification_selection_loss, codes, classes),
            published={"classes_": classes},
        )

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

    def predict(self, X: object, *, predict_config: PredictConfig | None = None) -> np.ndarray:
        """Decode the most probable class through the fitted label vocabulary."""
        probabilities = self.predict_proba(X, predict_config=predict_config)
        return self.classes_[probabilities.argmax(axis=1)]
