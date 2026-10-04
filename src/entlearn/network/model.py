"""The public Network lifecycle."""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Callable, Sequence
from functools import partial
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any, Literal, Never, overload

import torch
from torch import Tensor

from entlearn.network.build import _build_recipe
from entlearn.network.candidate import _fit_candidate, _fit_logger
from entlearn.network.config import (
    _DEFAULT_MAX_ITER,
    _DEFAULT_TOL,
    PredictConfig,
    _resolve_predict_config,
)
from entlearn.network.continuation import continue_fit
from entlearn.network.data import _stage_data
from entlearn.network.folds import _row_indices, _validation_folds
from entlearn.network.initialisation.capacity import validate_profile_capacity
from entlearn.network.initialisation.capture import capture_current_state
from entlearn.network.initialisation.geometry import (
    _prepare_initialisation,
    _seed_initial_state,
)
from entlearn.network.initialisation.resolve_state import staging_hints
from entlearn.network.initialisation.seeds import (
    _plan_seeds,
    _validated_seed,
)
from entlearn.network.persistence import _reduce_network
from entlearn.network.predict import predict
from entlearn.network.queries import InspectionName, TensorInspection, inspect, score_samples
from entlearn.network.selection import _select_candidate, _SelectionLoss
from entlearn.network.session import _CompiledGraph
from entlearn.network.state import (
    DataSchema,
    FitDiagnostics,
    InitialState,
    PredictionResult,
    ReconstructionResult,
    _FittedNetwork,
    _NetworkRecord,
    _SelectionHistory,
)
from entlearn.network.state_validation import resume_error
from entlearn.recipe import Recipe

if TYPE_CHECKING:
    from plotly.graph_objects import Figure
    from rich.console import Console, ConsoleOptions, RenderResult

TaskWeights = torch.Tensor | Callable[[torch.Tensor], torch.Tensor]


def _validate_fit_controls(recipe: object, max_iter: object, tol: object) -> float:
    """Return the resolved tolerance after checking the permanent fit controls.

    Raises:
        ValueError: If ``recipe``, ``max_iter`` or ``tol`` is outside its domain.
    """
    if type(recipe) is not Recipe:
        raise ValueError("recipe must be a Recipe")
    if isinstance(max_iter, bool) or not isinstance(max_iter, Integral) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if isinstance(tol, bool) or not isinstance(tol, Real):
        raise ValueError("tol must be a finite non-negative real number")
    resolved = float(tol)
    if not math.isfinite(resolved) or resolved < 0.0:
        raise ValueError("tol must be a finite non-negative real number")
    return resolved


def _validate_selection_controls(
    n_inits: object,
    seed: object,
    n_jobs: object,
    parallel_backend: object,
    retain: object,
    return_train_score: object,
) -> tuple[int, int]:
    """Return the candidate count and root seed after checking the selection controls."""
    if isinstance(n_inits, bool) or not isinstance(n_inits, Integral) or n_inits < 1:
        raise ValueError("n_inits must be a positive integer")
    root = _validated_seed(seed)
    if n_jobs is not None and (
        isinstance(n_jobs, bool) or not isinstance(n_jobs, Integral) or n_jobs == 0
    ):
        raise ValueError("n_jobs must be None or a non-zero integer")
    if parallel_backend not in ("threads", "processes"):
        raise ValueError("parallel_backend must be 'threads' or 'processes'")
    if not isinstance(retain, str) or retain not in ("winner", "states", "members"):
        raise ValueError("retain must be 'winner', 'states' or 'members'")
    if type(return_train_score) is not bool:
        raise ValueError("return_train_score must be a boolean")
    return int(n_inits), root


class Network:
    """The tensor-native model lifecycle."""

    _fitted: _FittedNetwork
    _selection: _SelectionHistory
    _diagnostics: FitDiagnostics
    _initial_states: tuple[InitialState, ...] | None
    _members: tuple[Network, ...] | None

    def __new__(cls) -> Never:
        """Reject direct construction."""
        raise TypeError("Network cannot be constructed directly")

    def __reduce__(self) -> tuple[Callable[..., Network], tuple[object, ...]]:
        """Transport canonical state, preserving owned member/state relationships."""
        return _reduce_network(self)

    def __deepcopy__(self, memo: dict[int, object]) -> Network:
        """Copy through the decoder once, without a preliminary tensor deepcopy."""
        restore, arguments = self.__reduce__()
        return restore(*arguments)

    def __repr__(self) -> str:
        """Summarise the fit in a simple representation."""
        return (
            f"Network(task={self.schema.task!r}, "
            f"K_active={dict(self.schema.K_active)!r}, "
            f"converged={self.diagnostics.converged!r}, "
            f"n_iter={self.diagnostics.n_iter})"
        )

    def save(self, path: str | os.PathLike[str], *, resumable: bool = False) -> None:
        """Save fitted state and structured metadata in one safetensors file.

        The path must end in ``.safetensors``.
        By default, training affiliations and instance weights are omitted.
        ``resumable=True`` requires and saves valid row-bound state for every
        retained member, preserving stopping controls and trajectory diagnostics.
        Either form leaves the source and its continuation capability unchanged.
        """
        from entlearn.network.persistence import save

        save(self, path, resumable=resumable)

    @classmethod
    def load(cls, path: str | os.PathLike[str], *, device: str | torch.device = "cpu") -> Network:
        """Load validated fitted state, preserving saved computation dtype.

        Loading defaults to CPU and does not implicitly restore the source device.
        Both capabilities can predict, fine-tune and capture selected parameter
        groups. Only a complete resumable payload permits resume and inspection
        of training coordinates; ``can_resume`` reports the validated capability.
        """
        from entlearn.network.persistence import load

        return cls._publish(load(path, device=device))

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        """Render a block table and fit summary when printed through Rich.

        Rich is optional and loaded only for this display. The summary reads stored
        configuration and diagnostics, never tensor values or prediction caches.
        """
        from entlearn.network.display import render_network

        yield render_network(self, width=options.max_width)

    @torch.inference_mode()
    def plot(
        self,
        block: str | None = None,
        X_cont: torch.Tensor | None = None,
        *,
        X_cat: Sequence[torch.Tensor] | None = None,
    ) -> Figure:
        """Plot one fitted block, or the loss and every block of the Network.

        Plotting is optional; install ``entlearn[plotting]`` to enable it. Each block is one
        row of panels, drawn with the functions in :mod:`entlearn.plotting`:

        - an input: its feature weights, its centroids over the first two features and,
          with ``X_cont``, the observations coloured by their affiliations;
        - a manifold input: :func:`~entlearn.plotting.plot_manifold`, which needs ``X_cont``;
        - a hidden block: its incoming transition matrix and, with ``X_cont``, the
          observations coloured by their affiliations in that block;
        - a head: its class transition matrix or its output centroids.

        Args:
            block: A block name, or ``None`` for the loss followed by every block.
            X_cont: Observations to draw, on the Network device and in its computation dtype.
            X_cat: Categorical columns paired with ``X_cont``.

        Returns:
            A Plotly figure.

        Raises:
            ModuleNotFoundError: If Plotly is not installed.
            ValueError: If the block name is unknown, or a manifold input has no ``X_cont``.
        """
        try:
            from entlearn.plotting.network import plot_network
        except ModuleNotFoundError as exc:
            if exc.name == "plotly" or (exc.name or "").startswith("plotly."):
                raise ModuleNotFoundError(
                    "Network.plot requires Plotly; install it with `pip install 'entlearn[plotting]'`"
                ) from None
            raise
        return plot_network(self, block=block, X_cont=X_cont, X_cat=X_cat)

    @classmethod
    def _publish(cls, record: _NetworkRecord) -> Network:
        """Construct the finished Network, the type's one construction point, and release caches.

        Each retained member becomes a child Network holding only its own selection record.
        """
        selection = record.selection
        if not selection.outcomes:
            raise ValueError("incomplete fitted diagnostics: selection has not finished")
        record.fitted.graph.release_caches()
        trajectory, at = record.fitted.trajectory, selection.warnings_at
        network = object.__new__(cls)
        network._fitted = record.fitted
        network._selection = selection
        network._diagnostics = FitDiagnostics(
            loss_history=trajectory.loss_history,
            n_iter=trajectory.n_iter,
            converged=trajectory.converged,
            warnings=trajectory.warnings[:at] + selection.warnings + trajectory.warnings[at:],
            initialisation_outcomes=selection.outcomes,
            selected_index=selection.selected_index,
        )
        network._initial_states = record.states
        network._members = (
            None
            if record.members is None
            else tuple(
                cls._publish(_NetworkRecord(member, _SelectionHistory((outcome,), outcome.index)))
                for member, outcome in zip(record.members, selection.outcomes, strict=True)
            )
        )
        return network

    def _record(self) -> _NetworkRecord:
        """Return the fitted state and selection history this Network was published from."""
        return _NetworkRecord(
            self._fitted,
            self._selection,
            self._initial_states,
            None if self._members is None else tuple(member._fitted for member in self._members),
        )

    @classmethod
    @torch.inference_mode()
    def fit(
        cls,
        recipe: Recipe,
        X_cont: torch.Tensor,
        y: torch.Tensor,
        *,
        X_cat: Sequence[torch.Tensor] | None = None,
        sample_weights: torch.Tensor | None = None,
        class_weights: torch.Tensor | None = None,
        task_weights: TaskWeights | None = None,
        initial_state: InitialState | None = None,
        n_inits: int = 1,
        n_jobs: int | None = None,
        parallel_backend: Literal["threads", "processes"] = "threads",
        retain: Literal["winner", "states", "members"] = "winner",
        selection_loss: _SelectionLoss | None = None,
        validation_pairs: Sequence[tuple[torch.Tensor, torch.Tensor]] | None = None,
        return_train_score: bool = False,
        init_rows: torch.Tensor | None = None,
        computation_dtype: torch.dtype | None = None,
        predict_config: PredictConfig | None = None,
        max_iter: int = _DEFAULT_MAX_ITER,
        tol: float = _DEFAULT_TOL,
        seed: int = 0,
        logger: logging.Logger | None = None,
        verbose: int = 0,
    ) -> Network:
        """Fit a classification or regression chain.

        Data and controls after ``y`` are keyword-only. Unsupported Recipe and data
        variants fail before any fitted Network is published.

        Every tensor a fit creates is an inference tensor, so a later in-place write on
        one must run under ``torch.inference_mode`` as well.

        Without validation pairs, each full-data candidate is calibrated when required
        before scoring and the winner is deployed without a second fit. Omitted
        ``selection_loss`` minimises the final weighted training objective.

        Validation requires ``selection_loss``. Each candidate shares one original state across
        independent fold fits. Each geometric classification model calibrates on its own
        positive-weight labelled training rows before single-pass scoring with that model's
        temperature. Each fold is scored and discarded before the next fit. The mean validation
        loss wins. With ``return_train_score=True``, exact ties prefer the candidate whose
        mean validation loss exceeds its mean training loss by the least; otherwise, and on
        equal gaps, the earlier candidate wins.
        The winner is refitted on all rows from its original state and calibrates afresh. No
        derived fold temperature transfers. Each retained full-data member does likewise.
        Calibration always minimises single-pass training log-loss, independently of the
        selection metric. A supplied temperature stays fixed.

        A non-converged fit emits one ``ConvergenceWarning`` before publication.
        Multi-initialisation reports every non-converged candidate in one warning
        (including losing candidates when the winner converged).

        Args:
            recipe: Complete model description.
            X_cont: ``(T, D_cont)`` continuous features.
            y: Classification codes or distributions, or a regression target.
            X_cat: Per-feature categorical codes or distributions.
            sample_weights: Optional (strictly positive) weights for the rows.
            class_weights: Optional non-negative weights for the classes.
            task_weights: Regression task weights, or a callable mapping labelled targets
                to their weights. Not valid for classification.
            initial_state: Optional detached initial geometry, requires ``n_inits=1``.
                A supplied state bypasses initialisation selection: validation pairs
                and selection loss are ignored, and this state is fitted once on all rows.
            n_inits: Positive number of independent candidates. One uses the root seed
                directly. Several candidates draw their sub-seeds before dispatch.
            n_jobs: Joblib worker count; ``None`` and ``1`` run serially, ``-1`` uses
                all CPUs available to joblib. One candidate never starts workers.
            parallel_backend: Threads or processes for several candidates. Processes
                require picklable callbacks and captured values. Either way, the warnings
                raised in the candidates are emitted once every candidate has finished.
            retain: What the fit keeps beyond the winner. ``"winner"`` keeps only the
                winner, whose ``initial_state`` is always retained. ``"states"`` also keeps
                every original input geometry in ``initial_states``, in candidate order.
                ``"members"`` keeps every fitted Network in ``members`` and implies
                ``"states"``: each entry in ``initial_states`` references the corresponding
                member's ``initial_state`` without copying its geometry.
                Every mode applies to one candidate (``n_inits=1``)
                too. Lightweight identity, score, training-score and backend records are
                always available in the diagnostics.
            selection_loss: Optional smaller-is-better tensor callback. It receives detached
                labelled predictions and targets, then keyword arguments ``sample_weights``,
                ``class_weights``, ``task_weights``, ``fold`` and ``partition``. ``fold`` is a
                copied train-validation index pair, or ``None`` in-sample. ``partition`` is
                ``"validation"``, or ``"training"`` for in-sample scoring and
                ``return_train_score``. Indices address this fit's rows. Labelled predictions
                preserve the corresponding partition's index order. Targets are class distributions
                or two-dimensional regression values. Row weights are restricted to labelled rows,
                converted to the computation dtype but not normalised or combined. Omitted weight
                families remain ``None``. The callback returns one finite real scalar,
                and does not receive a Network.
            validation_pairs: Optional materialised sequence of train-validation
                pairs. Pairs should be disjoint (and without duplicate entries) int64 tensor on the data
                device and must contain positive-weight
                labelled mass. Folds may overlap one another and need not cover all rows.
                Each fold model calibrates only on its training partition.
                Initialisation-selection folds are independent
                of any enclosing hyperparameter-search folds.
                Retaining all members performs one full-data
                refit per candidate, otherwise only the winner is refitted. Retaining
                original states alone does not add fits.
            return_train_score: Also score each validation fold's training partition, recording
                the mean loss as each outcome's ``train_score``, otherwise ``None``.
            init_rows: Optional int64 indices used only to generate original geometry,
                mutually exclusive with ``initial_state``. ``None`` uses all rows,
                including validation rows, which can leak validation information into
                initialisation. A leakage-free initialisation requires the supplied
                rows to never belong to any validation partition.
            computation_dtype: Optional float32 or float64 computation dtype.
            predict_config: Complete prediction policy. Omission uses task defaults.
            max_iter: Maximum number of complete fit iterations.
            tol: Non-negative relative convergence tolerance.
            seed: Root initialisation seed: any non-boolean integral value in ``[0, 2**63)``.
            logger: Optional caller-configured logger. Its level, handlers and propagation
                are never changed. Omission creates a private logger writing to stderr.
            verbose: Non-negative logging verbosity: 0 is silent, 1 logs the fit summary,
                and 2 or more also logs initial and complete-iteration losses. An injected
                logger's filters still apply. Warnings are independent of verbosity.

        Returns:
            A complete fitted Network.

        Raises:
            ValueError: If the Recipe, the data or a control is unsupported.
        """
        resolved_tol = _validate_fit_controls(recipe, max_iter, tol)
        resolved_logger = _fit_logger(logger, verbose)
        count, root = _validate_selection_controls(
            n_inits, seed, n_jobs, parallel_backend, retain, return_train_score
        )
        if initial_state is not None:
            # The caller has already selected the original geometry. Selection
            # controls must not create another validation/refit cycle.
            validation_pairs = None
            selection_loss = None
        if selection_loss is not None and not callable(selection_loss):
            raise ValueError("selection_loss must be callable or None")
        if validation_pairs is not None and selection_loss is None:
            raise ValueError("validation_pairs requires selection_loss")
        if initial_state is not None and count != 1:
            raise ValueError("initial_state requires n_inits=1")
        if initial_state is not None and init_rows is not None:
            raise ValueError("initial_state and init_rows are mutually exclusive")
        built = _build_recipe(recipe)
        resolved_predict_config = _resolve_predict_config(predict_config, built.head)
        if initial_state is not None and type(initial_state) is not InitialState:
            raise ValueError("initial_state must be an exact InitialState")
        categorical_cardinalities, known_n_classes = staging_hints(
            built, initial_state, X_cont, X_cat, y, computation_dtype
        )
        data = _stage_data(
            built,
            X_cont,
            y,
            X_cat=X_cat,
            sample_weights=sample_weights,
            class_weights=class_weights,
            task_weights=task_weights,
            computation_dtype=computation_dtype,
            categorical_cardinalities=categorical_cardinalities,
            known_n_classes=known_n_classes,
        )
        if initial_state is None:
            validate_profile_capacity(built, data.X_cont.shape[0])
        rows = None if init_rows is None else _row_indices(init_rows, data, "init_rows")
        folds = None if validation_pairs is None else _validation_folds(validation_pairs, data)
        # Bind common fit inputs; selection supplies the seed, state, policy, logging and rows.
        fit_one = partial(
            _fit_candidate,
            built=built,
            data=data,
            allow_conversion=computation_dtype is not None,
            max_iter=int(max_iter),
            tol=resolved_tol,
            init_rows=rows,
            provided=initial_state is not None,
        )
        record = _select_candidate(
            fit_one,
            _plan_seeds(root, count, data.X_cont.device),
            data=data,
            initial_state=initial_state,
            folds=folds,
            selection_loss=selection_loss,
            predict_config=resolved_predict_config,
            return_train_score=return_train_score,
            n_jobs=n_jobs,
            parallel_backend=parallel_backend,
            retain=retain,
            logger=resolved_logger,
            verbose=int(verbose),
        )
        return cls._publish(record)

    @property
    def _graph(self) -> _CompiledGraph:
        """Return the fitted graph without duplicating its ownership."""
        return self._fitted.graph

    @property
    def recipe(self) -> Recipe:
        """Return the immutable description attached to this fit."""
        return self._graph.recipe

    @property
    def device(self) -> torch.device:
        """Return the device on which query tensors must be staged."""
        return self._graph.input.continuous_centroids.device

    @property
    def schema(self) -> DataSchema:
        """Return the fitted data schema."""
        return self._fitted.schema

    @property
    def diagnostics(self) -> FitDiagnostics:
        """Return the immutable fit diagnostics."""
        return self._diagnostics

    @property
    def predict_config(self) -> PredictConfig:
        """Return the complete immutable fitted prediction policy."""
        return self._fitted.predict_config

    @property
    def initial_state(self) -> InitialState:
        """Return the winner's original state, detached from caller and fitted geometry.

        This state is always retained, including for one candidate and for each retained
        member. It records the supplied or generated state before this fit and its pruning.
        """
        return self._fitted.initial_state

    @property
    def initial_states(self) -> tuple[InitialState, ...] | None:
        """Return all original states under ``retain="states"`` or ``"members"``, else ``None``.

        Entries follow candidate order and reference retained members' original states,
        so ``members[i].initial_state`` is the same state as ``initial_states[i]``.
        The winner's ``initial_state`` remains available regardless of this collection.
        """
        return self._initial_states

    @property
    def members(self) -> tuple[Network, ...] | None:
        """Return fitted members in candidate order under ``retain="members"``, else ``None``.

        Each member's original state is ``members[i].initial_state``.
        """
        return self._members

    @property
    def can_resume(self) -> bool:
        """Whether the winner and every retained member have valid row-bound state."""
        return resume_error(self._fitted) is None and all(
            resume_error(member._fitted) is None for member in self._members or ()
        )

    def resume(
        self,
        X_cont: torch.Tensor,
        y: torch.Tensor,
        *,
        X_cat: Sequence[torch.Tensor] | None = None,
        sample_weights: torch.Tensor | None = None,
        class_weights: torch.Tensor | None = None,
        task_weights: TaskWeights | None = None,
        computation_dtype: torch.dtype | None = None,
        max_iter: int | None = None,
        tol: float | None = None,
        logger: logging.Logger | None = None,
        verbose: int = 0,
    ) -> Network:
        """Continue fitted trajectories, returning a new fully fitted Network.

        Uses the existing Recipe and prediction policy. Row count, feature and
        output meanings, active shapes and resumable state must remain compatible.

        Every retained fitted member continues independently. The selected winner,
        original-state collections and historical selection records are preserved.
        Discarded candidates and fold fits are not recreated and selection is not
        repeated. All members must have resumable state.

        An already-converged trajectory takes no extra optimisation step.
        ``max_iter`` caps the total iteration count, including that of earlier calls.
        Omitted stopping controls reuse the source fit's settings. This means a converged
        trajectory cannot advance unless the requested tolerance is tighter.
        New losses are appended and iteration counts
        accumulate.
        Derived calibration and the scoring reference are rebuilt before return;
        a supplied prediction temperature stays fixed.

        Args:
            X_cont: Finite continuous matrix, with the fitted row and feature counts.
                Its device selects placement. copied state moves to this device.
            y: Compatible targets, with the same row count.
            X_cat: Categorical features in fitted order and within known cardinalities.
            sample_weights: Optional (strictly positive) per-row sample weights.
            class_weights: Optional classification weights in fitted class order.
            task_weights: Optional regression row weights or a callable evaluated once
                on labelled targets, shared by all retained trajectories.
            computation_dtype: Optional float32 or float64 override; otherwise use
                ``X_cont.dtype``. Floating data and copied state convert once.
            max_iter: Positive total iteration ceiling per trajectory. ``None``
                reuses the source fit's ceiling. A ceiling at or below the number
                already completed takes no further steps.
            tol: Non-negative relative-loss tolerance, also applied to retained history.
                ``None`` reuses the source fit's tolerance.
            logger: Optional caller-owned logger, never reconfigured.
            verbose: Non-negative logging verbosity, as for ``fit``.

        Raises:
            ValueError: If controls, data or retained state are incompatible.
        """
        resolved_max_iter = self._fitted.max_iter if max_iter is None else max_iter
        resolved_tol = _validate_fit_controls(
            self.recipe, resolved_max_iter, self._fitted.tol if tol is None else tol
        )
        resolved_logger = _fit_logger(logger, verbose)
        record = continue_fit(
            self._record(),
            X_cont,
            y,
            X_cat=X_cat,
            sample_weights=sample_weights,
            class_weights=class_weights,
            task_weights=task_weights,
            computation_dtype=computation_dtype,
            max_iter=int(resolved_max_iter),
            tol=resolved_tol,
            logger=resolved_logger,
            verbose=int(verbose),
        )
        return self._publish(record)

    def capture_current_state(self, *, blocks: Sequence[str] | None = None) -> InitialState:
        """Capture detached starting parameters of selected blocks, or all by default.

        Names select complete parameter groups, not retained members. Unknown or
        duplicate names raise. Rows, instance weights and diagnostics are excluded.
        """
        return capture_current_state(self._fitted, blocks)

    def fine_tune(
        self,
        X_cont: torch.Tensor,
        y: torch.Tensor,
        *,
        recipe: Recipe | None = None,
        X_cat: Sequence[torch.Tensor] | None = None,
        sample_weights: torch.Tensor | None = None,
        class_weights: torch.Tensor | None = None,
        task_weights: TaskWeights | None = None,
        computation_dtype: torch.dtype | None = None,
        max_iter: int = _DEFAULT_MAX_ITER,
        tol: float = _DEFAULT_TOL,
        logger: logging.Logger | None = None,
        verbose: int = 0,
    ) -> Network:
        """Fine-tune active fitted parameters on compatible new rows.

        Returns a newly fitted Network. Row-dependent values and diagnostics
        rebuild even at an unchanged row count. Original states and retained
        members remain ordered, with no reselection or new initialisations.

        Only shape-preserving hyperparameters may change:
        input temperatures and categorical cost scale or manifold alpha, hidden
        epsilon, connection delta and theta_alpha, and regression epsilon_M
        (or fixed W_M with the same number of targets).
        Structure and initialisation controls must remain fixed.

        Feature and output weights transfer exactly. Infinite temperatures freeze
        their current values. Instance
        weights instead start from the new normalised sample weights, or uniform
        weights when omitted, and infinite epsilon_T freezes those new values.

        Derived prediction temperatures recalibrate; supplied ones stay
        fixed. The scoring reference rebuilds. Prediction-only models can fine-tune.

        Raises:
            ValueError: If controls, Recipe, data or fitted parameters are incompatible.
        """
        recipe = self.recipe if recipe is None else recipe
        resolved_tol = _validate_fit_controls(recipe, max_iter, tol)
        resolved_logger = _fit_logger(logger, verbose)
        record = continue_fit(
            self._record(),
            X_cont,
            y,
            X_cat=X_cat,
            sample_weights=sample_weights,
            class_weights=class_weights,
            task_weights=task_weights,
            computation_dtype=computation_dtype,
            max_iter=int(max_iter),
            tol=resolved_tol,
            logger=resolved_logger,
            verbose=int(verbose),
            replacement=recipe,
        )
        return self._publish(record)

    @overload
    def inspect(self, name: TensorInspection) -> dict[str, Tensor]: ...
    @overload
    def inspect(self, name: Literal["categorical_centroids"]) -> dict[str, tuple[Tensor, ...]]: ...
    @overload
    def inspect(self, name: Literal["head_parameters"]) -> dict[str, dict[str, Tensor]]: ...
    @overload
    def inspect(
        self, name: Literal["affiliation_regimes"]
    ) -> dict[str, Literal["hard", "soft"]]: ...
    @torch.inference_mode()
    def inspect(self, name: InspectionName) -> dict[str, Any]:
        """Return detached fitted values keyed by stable owner names.

        Names are ``continuous_centroids``, ``categorical_centroids``, ``feature_weights``,
        ``transition_matrices``, ``head_parameters``, ``manifold_projectors``,
        ``training_affiliations``, ``training_instance_weights`` and ``affiliation_regimes``.
        Categorical centroid tuples are feature-ordered; head parameters map classification
        ``theta`` or regression ``C_y`` and, when present, ``W_M``. Row-bound requests require
        retained training values. Continuous centroids may have zero columns for
        categorical-only input.
        ``affiliation_regimes`` maps each input and hidden block to ``"hard"`` or ``"soft"``:
        the regime its fit resolved from ``epsilon`` and the computation dtype, which
        prediction keeps for any number of query rows.

        Raises:
            ValueError: If the name is unknown or unavailable for the fitted model.
        """
        return inspect(self._graph, name)

    def _query_config(self, predict_config: PredictConfig | None) -> PredictConfig:
        """Return the fitted policy, or resolve a complete call-time replacement."""
        if predict_config is None:
            return self.predict_config
        return _resolve_predict_config(predict_config, self._graph.head.description)

    @torch.inference_mode()
    def score_samples(
        self,
        X_cont: torch.Tensor,
        *,
        X_cat: Sequence[torch.Tensor] | None = None,
        predict_config: PredictConfig | None = None,
    ) -> torch.Tensor:
        """Return inlier percentiles of recovered query instance weights.

        The right-continuous empirical CDF counts training-reference W_T values less
        than or equal to each query weight. Scores lie in ``[0, 1]``; zero means
        below every reference value. The reference is recovered on the training
        input through the final fitted prediction configuration, it is not the fitted
        training instance weights.

        A complete ``predict_config`` replaces query recovery only. The reference
        stays fixed, so an override compares potentially different recovery modes.
        Query weights are relative to the retained training log-normaliser and
        are not normalised over the query batch.

        Raises:
            ValueError: Before prediction if the fit has no instance-weight recovery or
                retained scoring reference, or if query data or policy is invalid.

        Warns:
            UserWarning: If the reference has collapsed within its precision noise band.
        """
        return score_samples(
            self._graph, self._fitted.Wt_ref, X_cont, X_cat, self._query_config(predict_config)
        )

    @torch.inference_mode()
    def predict(
        self,
        X_cont: torch.Tensor,
        *,
        X_cat: Sequence[torch.Tensor] | None = None,
        predict_config: PredictConfig | None = None,
        predict_init: str | int | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return classification distributions or unconstrained regression values.

        Input and hidden affiliations retain their fitted hard or soft regime.
        Single-pass predictions are batch-independent (apart from floating-point round-off).

        Iterative updates
        are row-local, but convergence uses the mean batch objective, so other query
        rows may change the stopping iteration and thus the returned prediction.
        ``predict_config=None`` (including omission) uses the fitted policy.

        ``predict_init`` is the initial prediction used in iterative prediction, and defaults to starting from the geometric
        read-out. A non-boolean integer start is any integral value in ``[0, 2**63)``.
        Classification accepts M-only ``"arithmetic"``, ``"uniform"``,
        an integer seed, or a finite row-stochastic tensor. Regression accepts ``"mean"`` (the equal-weight
        mean of fitted output centroids), an integer seed (random convex
        combinations of those centroids), or a finite real tensor. Tensor starts
        must match the output shape, computation dtype and device. A supplied
        start skips only the first prediction-coordinate update.

        Raises:
            ValueError: If the input does not match the fitted geometry.
        """
        return self.predict_with_details(
            X_cont, X_cat=X_cat, predict_config=predict_config, predict_init=predict_init
        ).prediction

    @torch.inference_mode()
    def predict_with_details(
        self,
        X_cont: torch.Tensor,
        *,
        X_cat: Sequence[torch.Tensor] | None = None,
        predict_config: PredictConfig | None = None,
        predict_init: str | int | torch.Tensor | None = None,
        details: Sequence[str] = (),
    ) -> PredictionResult:
        """Predict and return the requested detached query details.

        Supported names are ``affiliations``, ``instance_weights``,
        ``reconstruction`` and ``diagnostics``. The latter populates ``n_iter``, ``converged`` and
        ``loss_history``. Configuration and starting coordinates follow
        ``predict``.
        """
        return predict(
            self._graph,
            X_cont,
            X_cat_new=X_cat,
            config=self._query_config(predict_config),
            predict_init=predict_init,
            details=details,
        )

    @torch.inference_mode()
    def predict_all(
        self,
        X_cont: torch.Tensor,
        *,
        X_cat: Sequence[torch.Tensor] | None = None,
        predict_config: PredictConfig | None = None,
        predict_init: str | int | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return raw predictions with shape ``(n_members, n_rows, n_outputs)``.

        Members retain candidate order. Omission of ``predict_config`` uses each member's
        own fitted policy, while an override applies to every member.

        Raises:
            ValueError: If members were not retained or query data or policy is invalid.
        """
        if self.members is None:
            raise ValueError('predict_all requires fitting with retain="members"')
        return torch.stack(
            [
                member.predict(
                    X_cont, X_cat=X_cat, predict_config=predict_config, predict_init=predict_init
                )
                for member in self.members
            ]
        )

    @torch.inference_mode()
    def predict_all_with_details(
        self,
        X_cont: torch.Tensor,
        *,
        X_cat: Sequence[torch.Tensor] | None = None,
        predict_config: PredictConfig | None = None,
        predict_init: str | int | torch.Tensor | None = None,
        details: Sequence[str] = (),
    ) -> tuple[PredictionResult, ...]:
        """Return one detailed result per retained member, in candidate order.

        Members may retain different numbers of clusters after pruning, so their details are
        not stacked. Each member uses its own fitted policy unless an
        override is supplied. The same starting-coordinate request applies to all
        members.
        """
        if self.members is None:
            raise ValueError('predict_all_with_details requires fitting with retain="members"')
        return tuple(
            member.predict_with_details(
                X_cont,
                X_cat=X_cat,
                predict_config=predict_config,
                predict_init=predict_init,
                details=details,
            )
            for member in self.members
        )

    @torch.inference_mode()
    def reconstruct(
        self,
        X_cont: torch.Tensor,
        *,
        X_cat: Sequence[torch.Tensor] | None = None,
        predict_config: PredictConfig | None = None,
        predict_init: str | int | torch.Tensor | None = None,
    ) -> ReconstructionResult:
        """Reconstruct features using the selected prediction affiliations.

        Query rows must match the fitted feature order, device and computation dtype.
        Every input returns a detached ``ReconstructionResult`` with continuous values
        and a feature-ordered tuple of categorical probability matrices, empty for
        continuous-only and manifold input. Standard input blends fitted centroids;
        manifold input blends projections onto the clusters' planes.

        Reconstruction uses prediction's final input affiliations: the forward pass
        in single mode, or the last completed refinement iteration in iterative
        mode. Configuration and starting coordinates follow ``predict``.

        Raises:
            ValueError: If query data, configuration or starting coordinates are incompatible.
        """
        result = self.predict_with_details(
            X_cont,
            X_cat=X_cat,
            predict_config=predict_config,
            predict_init=predict_init,
            details=("reconstruction",),
        )
        assert result.reconstruction is not None
        return result.reconstruction

    @classmethod
    @torch.inference_mode()
    def initialise(
        cls,
        recipe: Recipe,
        X_cont: torch.Tensor,
        y: torch.Tensor,
        *,
        X_cat: Sequence[torch.Tensor] | None = None,
        sample_weights: torch.Tensor | None = None,
        class_weights: torch.Tensor | None = None,
        task_weights: TaskWeights | None = None,
        feature_weights: torch.Tensor | None = None,
        continuous_centroids: torch.Tensor | None = None,
        categorical_centroids: Sequence[torch.Tensor] | None = None,
        computation_dtype: torch.dtype | None = None,
        seed: int = 0,
        init_rows: torch.Tensor | None = None,
    ) -> InitialState:
        """Create detached input geometry from a Recipe.

        The generated input cluster count is capped at the eligible row count with a
        ``UserWarning``. Balanced classification counts only labelled rows.
        The immutable Recipe remains unchanged; ``InitialState.input_geometry.K_active``
        records the realised cluster count.

        Args:
            recipe: Complete model description.
            X_cont: ``(T, D_cont)`` float32 or float64 continuous features.
            y: Classification codes or distributions, or regression targets.
            X_cat: Per-feature int64 codes or floating distributions.
            sample_weights: Optional (strictly positive) weights for the rows.
            class_weights: Optional non-negative weights for the classification classes.
            task_weights: Optional non-negative weights for the regression rows, or a
                callable that maps the staged target to them.
            feature_weights: Optional non-negative input feature weights.
            continuous_centroids: Optional continuous centroids.
            categorical_centroids: Optional categorical centroids.
            computation_dtype: Optional float32 or float64 computation dtype.
            seed: Root seed: any non-boolean integral value in ``[0, 2**63)``.
            init_rows: Optional non-empty set of int64 row indices on the data device.
                Geometry uses only these rows, while feature and target meanings
                are validated on the complete data. Omission uses all rows.

        Returns:
            Detached input geometry in the computation dtype.

        Raises:
            ValueError: If the Recipe, the data or a selection control is unsupported.
        """
        resolved_seed = _validated_seed(seed)
        built, data = _prepare_initialisation(
            recipe,
            X_cont,
            y,
            X_cat=X_cat,
            sample_weights=sample_weights,
            class_weights=class_weights,
            task_weights=task_weights,
            feature_weights=feature_weights,
            continuous_centroids=continuous_centroids,
            categorical_centroids=categorical_centroids,
            computation_dtype=computation_dtype,
        )
        rows = None if init_rows is None else _row_indices(init_rows, data, "init_rows")
        if rows is not None and (
            continuous_centroids is not None or categorical_centroids is not None
        ):
            raise ValueError("init_rows and supplied centroids are mutually exclusive")
        return _seed_initial_state(
            built,
            data,
            feature_weights=feature_weights,
            continuous_centroids=continuous_centroids,
            categorical_centroids=categorical_centroids,
            allow_conversion=computation_dtype is not None,
            seed=resolved_seed,
            rows=rows,
        )
