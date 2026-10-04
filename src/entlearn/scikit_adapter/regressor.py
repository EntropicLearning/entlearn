"""Real-valued regression through the public tensor lifecycle."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from functools import partial
from typing import Literal

import numpy as np
import torch
from sklearn.base import RegressorMixin
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils import Tags
from sklearn.utils.validation import check_array, check_consistent_length, check_is_fitted

from entlearn import InitialState, PredictConfig, Recipe
from entlearn.scikit_adapter.base import _BaseEON, _StagedTarget
from entlearn.scikit_adapter.features import _staging_dtype
from entlearn.scikit_adapter.selection import _regression_selection_loss


def _prepare_regression_y(y: object, dtype: torch.dtype) -> tuple[np.ndarray, bool]:
    """Stage real targets directly to computation precision, retaining their dimensionality.

    Fully missing rows mark unlabelled data. Partially missing multi-output rows
    raise here rather than discarding their observed values. Direct tensor fitting
    instead accepts such rows with a warning and masks the whole row.
    """
    if y is None:
        raise ValueError("requires y to be passed, but the target y is None")
    arr = check_array(
        y,
        dtype=_staging_dtype(dtype),
        ensure_2d=False,
        ensure_all_finite="allow-nan",
        order="C",
        force_writeable=True,
    )
    was_1d = arr.ndim == 1
    nan_mask = np.isnan(arr)
    if was_1d:
        labelled = ~nan_mask
    else:
        row_any_nan = nan_mask.any(axis=1)
        row_all_nan = nan_mask.all(axis=1)
        if (row_any_nan & ~row_all_nan).any():
            raise ValueError(
                "y has partially-NaN row(s): a regression target row must be either fully "
                "observed or fully NaN"
            )
        labelled = ~row_all_nan
    if not labelled.any():
        raise ValueError("y has no labelled instances; at least one labelled instance is required")
    return arr, was_1d


class EONRegressor(RegressorMixin, _BaseEON):
    """A single- or multi-output regressor delegating its numerical lifecycle to Network.

    A ``recipe`` is required. Constructor parameters are stored verbatim for cloning.
    Integer and floating tabular values stage directly to ``dtype`` (float64 by
    default); ``device=None`` selects CPU.

    ``regression_weighting`` passes unchanged to Network as ``task_weights``.
    Network alone invokes it on labelled tensor targets and validates its result.
    Fully missing target rows can be marked as unlabelled using ``NaN``.
    Partially missing multi-output rows
    are rejected; use direct Network fitting instead, which warns and masks the whole row.

    Prediction preserves the fit target's dimensionality and computation precision.
    By default queries use the fitted prediction policy, dtype and device, even after
    constructor changes. A call-time ``PredictConfig`` replaces the policy fully.

    Several ``n_inits`` are selected in-sample when ``cv=None`` or out-of-sample using the
    provided splitter otherwise. ``scoring=None`` uses R² for selection. Strings
    and callables use scikit-learn scorer semantics. ``groups`` passed to
        ``fit`` are resolved by the splitter. Network owns candidate fitting, retention, ranking
    and finalisation. ``return_train_score=True`` also scores each ``cv`` training
    partition: exact validation-score ties, then prefer the smaller training-validation
    gap, then candidate order. Each outcome in ``network_.diagnostics`` records the
    scores negated as losses, and ``network_.diagnostics.selected_outcome`` is the
    winner's.

    ``warm_start=False`` starts a fresh fit. ``"resume"`` continues the fitted
    trajectory on its training rows, with ``max_iter`` as a total iteration ceiling.
    ``"fine_tune"`` adapts to compatible data using the current Recipe and starts
    new operation diagnostics. To supply a replacement, call
    ``set_params(recipe=new_recipe)`` or edit nested Recipe parameters before ``fit``.
    Network permits shape-preserving objective changes, not topology or cluster-count
    changes. Both continuation modes require ``n_inits=1`` when fitted.
    Either string on an unfitted estimator emits a ``UserWarning`` and fits fresh,
    including SearchCV clones. Boolean ``True`` is invalid, incompatible continuation
    raises without falling back or replacing the previous fitted model.

    Continuation preserves fitted category vocabularies, retained members and their
    original winner. Initial-state, random-seed, selection and retention controls
    apply only to fresh fits. ``predict_config=None`` retains fitted policy during continuation. Any other
    request is compared with the fit-time request after both are resolved, a different policy is
    rejected. Every failure leaves the previous fitted model usable.
    """

    _y_was_1d: bool

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

    def _stage_target(
        self, y: object, X: object, *, continuing: bool, dtype: torch.dtype, device: torch.device
    ) -> _StagedTarget:
        """Stage real targets in computation precision, retaining their dimensionality."""
        del continuing
        target, was_1d = _prepare_regression_y(y, dtype)
        check_consistent_length(X, target)
        return _StagedTarget(
            target=target,
            tensor=torch.as_tensor(target, dtype=dtype, device=device),
            weights={"task_weights": self.regression_weighting},
            selection_loss=partial(_regression_selection_loss, target, was_1d),
            published={"_y_was_1d": was_1d},
        )

    def predict(self, X: object, *, predict_config: PredictConfig | None = None) -> np.ndarray:
        """Return predictions with the response rank established by the initial fit."""
        X_cont, X_cat = self._prepare_prediction_features(X)
        result = (
            self.network_.predict(X_cont, X_cat=X_cat or None, predict_config=predict_config)
            .cpu()
            .numpy()
        )
        return result[:, 0] if self._y_was_1d else result

    @property
    def n_outputs_(self) -> int:
        """Return the fitted number of regression targets without duplicating Network state."""
        check_is_fitted(self, "network_")
        return self.network_.schema.M

    def __sklearn_tags__(self) -> Tags:
        """Declare support for vector and multi-output targets."""
        tags = super().__sklearn_tags__()
        tags.target_tags.multi_output = True
        return tags
