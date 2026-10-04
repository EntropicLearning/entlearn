"""Public immutable Network state values."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields, replace
from typing import TYPE_CHECKING, Literal

import torch

from entlearn.network.config import PredictConfig
from entlearn.network.transfer import copy_tensor
from entlearn.recipe import (
    ClassificationHead,
    Coupling,
    Hidden,
    Input,
    ManifoldInput,
    RegressionHead,
)

if TYPE_CHECKING:
    from entlearn.network.session import _CompiledGraph


@dataclass(frozen=True, eq=False)
class ReconstructionResult:
    """Detached reconstruction of every input kind, in one record.

    ``continuous`` has shape ``(T, D_cont)``, including zero columns for
    categorical-only input. ``categorical`` is a feature-ordered tuple of
    ``(T, M_i)`` probability tensors. Modality
    order matches the input tensors and ``DataSchema.M_cat``. The tuple is empty
    for continuous-only and manifold input.

    Categorical reconstruction is an arithmetic mixture of fitted centroid
    distributions under the query affiliations.
    """

    continuous: torch.Tensor
    categorical: tuple[torch.Tensor, ...]


@dataclass(frozen=True, eq=False)
class PredictionResult:
    """Predictions and explicitly requested query details.

    Single-pass diagnostics are zero iterations, false convergence and an empty
    history. Iterative history contains one loss per completed iteration, not
    including the initial forward pass.
    """

    prediction: torch.Tensor
    affiliations: Mapping[str, torch.Tensor] | None = None
    instance_weights: torch.Tensor | None = None
    n_iter: int | None = None
    converged: bool | None = None
    loss_history: tuple[float, ...] | None = None
    reconstruction: ReconstructionResult | None = None


@dataclass(frozen=True)
class InitOutcome:
    """The selection record of one candidate initialisation.

    ``Network.fit`` selects the candidate with the lowest ``score``, ranking NaN last.
    Exact ties go to the smaller gap ``score - train_score`` when training scores were
    recorded, and then to the earlier candidate.

    Attributes:
        index: Position in candidate order, starting at zero.
        seed: Integer seed of the candidate.
        score: Selection loss, where smaller is better: the final training objective, the
            ``selection_loss`` value, or its mean over the validation pairs.
        train_score: Mean training-partition loss, recorded under ``return_train_score``
            with validation pairs; ``None`` otherwise.
        requested_backend: The requested ``parallel_backend``.
        effective_backend: The backend used: ``"serial"`` for one worker or one candidate,
            and ``"threads"`` when processes were requested inside a worker.
    """

    index: int
    seed: int
    score: float
    train_score: float | None
    requested_backend: Literal["threads", "processes"]
    effective_backend: Literal["serial", "threads", "processes"]


def _winning_outcome(outcomes: Iterable[InitOutcome]) -> InitOutcome:
    """Return the lowest score, ranking NaN last, then the smaller recorded gap, then the first."""

    def rank(outcome: InitOutcome) -> tuple[bool, float, float]:
        if math.isnan(outcome.score):
            return True, 0.0, 0.0
        gap = 0.0 if outcome.train_score is None else outcome.score - outcome.train_score
        return False, outcome.score, gap

    return min(outcomes, key=rank)


@dataclass(frozen=True)
class FitDiagnostics:
    """The loss and termination record of one fitted Network.

    Attributes:
        loss_history: Initial loss followed by one loss per complete iteration.
        n_iter: Total number of complete iterations.
        converged: Whether the final iteration met the relative tolerance.
        warnings: Warning messages produced during the fit.
        initialisation_outcomes: One selection record per candidate, in seed order: its
            ``index``, ``seed``, ``score`` (the selection loss; smaller is better),
            ``train_score``, and requested and effective parallel backends. A retained
            member holds only its own record.
        selected_index: Candidate index of the record this Network was fitted from:
            the winner's, or a retained member's own.
    """

    loss_history: tuple[float, ...]
    n_iter: int
    converged: bool
    warnings: tuple[str, ...]
    initialisation_outcomes: tuple[InitOutcome, ...] = ()
    selected_index: int = 0

    @property
    def selected_outcome(self) -> InitOutcome:
        """Return the historical record of the candidate this Network was fitted from.

        On the selected Network this is the winner's record, on a retained member it
        is that member's own. With ``return_train_score``, ``score - train_score`` is
        the recorded gap between mean validation and mean training selection loss.

        Raises:
            ValueError: If no record carries ``selected_index``.
        """
        for outcome in self.initialisation_outcomes:
            if outcome.index == self.selected_index:
                return outcome
        raise ValueError("no initialisation outcome records the selected index")


@dataclass(frozen=True)
class DataSchema:
    """The fitted tensor dimensions and active cluster counts.

    Attributes:
        task: ``"classification"`` or ``"regression"``.
        D_cont: Number of continuous features.
        M_cat: Cardinality of each categorical feature, in feature order.
        M: Number of classes, or regression output dimensions.
        K_active: Active cluster count of every block, in stable order; a head's count
            is ``M``, the number of classes.
        computation_dtype: The floating dtype of every real-valued tensor.
    """

    task: Literal["classification", "regression"]
    D_cont: int
    M_cat: tuple[int, ...]
    M: int
    K_active: tuple[tuple[str, int], ...]
    computation_dtype: torch.dtype


@dataclass(frozen=True, eq=False)
class InputGeometry:
    """Detached input geometry and its generation metadata.

    ``Network.initialise`` generates this group and ``Network.capture_current_state``
    captures it from a fitted model. Both return it as ``InitialState.input_geometry``.
    ``captured`` records which: captured geometry has been learned and therefore
    may only be reused at the same active ``K``. A directly constructed instance is
    supplied as ``InitialState(input_geometry=...)``.

    A fresh fit starts from this geometry when its input has the same name and kind, and
    agrees with ``input`` in every setting except ``K``, the fitting temperatures, the
    categorical cost scale and the manifold cost ratio; captured geometry may also change
    ``centroid_strategy``, ``greedy_candidates``, ``balanced`` and ``W_std``. A narrower
    ``K`` takes the leading centroids when ``prefix_reusable``. A wider ``K`` never matches,
    and captured geometry matches only its own ``K``. Otherwise the fit warns and generates
    fresh geometry. A matching ``K`` above ``K_active`` warns and fits only the stored centroids.
    Malformed geometry raises ``ValueError``, including captured geometry without exactly
    ``input.K`` centroids.

    Initialisation and capture return detached tensors that share no storage with a
    model. Editing a tensor in place
    can change a later fit that uses this geometry, but not its source model or a fit
    that has already copied it.

    Attributes:
        input: The requested input block description. Generated geometry may have
            fewer centroids when the eligible row count is below its requested ``K``.
        continuous_centroids: ``(K_active, D_cont)`` continuous centroids.
        categorical_centroids: Per-feature ``(K_active, M_i)`` categorical centroids.
        feature_weights: ``(D_cont + D_cat,)`` feature weights of a standard input;
            ``None`` for a manifold input.
        manifold_projectors: ``(K_active, D_cont, d)`` projectors of a manifold input;
            ``None`` for a standard input.
        captured: Whether the geometry was captured from a fitted ``Network``.
    """

    input: Input | ManifoldInput
    continuous_centroids: torch.Tensor
    categorical_centroids: tuple[torch.Tensor, ...]
    feature_weights: torch.Tensor | None
    manifold_projectors: torch.Tensor | None
    captured: bool

    @property
    def K_active(self) -> int:
        """The number of input clusters, which may be below the requested ``input.K``."""
        return int(self.continuous_centroids.shape[0])

    @property
    def prefix_reusable(self) -> bool:
        """Whether a narrower input may reuse the leading ``k <= K`` centroids.

        Generated standard k-means++, and greedy k-means++ with a fixed
        candidate count, select each cluster centroid independently of the ones after it. Balanced
        allocation and the K-dependent greedy default do not, while captured geometry was
        learned and therefore also not reusable.
        """
        strategy_ok = (
            self.input.centroid_strategy == "kmeans++" or self.input.greedy_candidates is not None
        )
        return not self.captured and not self.input.balanced and strategy_ok


def _input_width(geometry: InputGeometry, description: Input | ManifoldInput) -> int:
    """Return how many leading stored centroids a fit of ``description`` uses."""
    return min(description.K, geometry.K_active)


def _check_cluster_match(geometry: InputGeometry, description: Input | ManifoldInput) -> int:
    """Return the width at which ``geometry`` starts a fit of ``description``.

    Raise a ``ValueError`` that names the mismatch when it cannot.
    """
    stored = geometry.input
    if type(description) is not type(stored):
        raise ValueError(
            f"stored input is {type(stored).__name__}, but the requested input is "
            f"{type(description).__name__}; reuse requires the same input kind"
        )
    fitting = {"K", "epsilon", "epsilon_D", "epsilon_T", "delta_cat", "alpha"}
    if geometry.captured:
        fitting |= {"centroid_strategy", "greedy_candidates", "balanced", "W_std"}
    for field in fields(stored):
        if field.name not in fitting and getattr(stored, field.name) != getattr(
            description, field.name
        ):
            raise ValueError(
                f"stored input parameter {field.name!r} is {getattr(stored, field.name)!r}, "
                f"but the requested value is {getattr(description, field.name)!r}; "
                "generate a new InitialState to change this setting"
            )
    if geometry.captured:
        if description.K != stored.K:
            raise ValueError(
                f"this InitialState was captured from a fitted model with {stored.K} "
                f"centroids; the requested K={description.K} must equal {stored.K} to reuse "
                "learned geometry"
            )
    elif description.K > stored.K:
        raise ValueError(
            f"requested K={description.K} exceeds the original request for "
            f"{stored.K} centroids; generate a new InitialState for a larger model"
        )
    elif description.K < stored.K and not geometry.prefix_reusable:
        raise ValueError(
            f"cannot take the first {description.K} stored centroids: balanced selection "
            "and greedy k-means++ without a fixed greedy_candidates value depend on "
            f"the original requested count ({stored.K}). Keep that count or "
            "generate a new InitialState"
        )
    return _input_width(geometry, description)


@dataclass(frozen=True, eq=False)
class _HiddenParameters:
    """A named hidden block's incoming transition."""

    description: Hidden
    coupling: Coupling
    theta: torch.Tensor


@dataclass(frozen=True, eq=False)
class _ClassificationParameters:
    """A named head's incoming transition."""

    description: ClassificationHead
    theta: torch.Tensor


@dataclass(frozen=True, eq=False)
class _RegressionParameters:
    """A named head's output centroids and output weights; None means uniform ``W_M``."""

    description: RegressionHead
    C_y: torch.Tensor
    W_M: torch.Tensor | None


_DownstreamParameters = _HiddenParameters | _ClassificationParameters | _RegressionParameters


def _copy_parameter_group(group, dtype=None, device=None):
    """Copy a complete parameter group without copying immutable descriptions."""

    def copy_value(value):
        if isinstance(value, torch.Tensor):
            return copy_tensor(value, dtype or value.dtype, device or value.device)
        if isinstance(value, tuple):
            return tuple(copy_value(item) for item in value)
        return value

    copied = {field.name: copy_value(getattr(group, field.name)) for field in fields(group)}
    return replace(group, **copied)


@dataclass(frozen=True, eq=False)
class InitialState:
    """Detached, possibly partial named parameters for starting a fresh fit.

    Capture includes all fitted parameter groups unless blocks are selected.
    Initialisation generates only input geometry and connection seeds. Absent
    groups initialise normally. No rows, affiliations, instance weights, caches,
    diagnostic history or retained members are included. ``input_geometry`` is
    ``None`` when the state holds no input group.
    """

    input_geometry: InputGeometry | None = None
    parameters: tuple[_DownstreamParameters, ...] = ()
    connection_sub_seeds: tuple[tuple[str, int], ...] = ()

    @property
    def block_names(self) -> tuple[str, ...]:
        """Stable names of supplied groups, with input first when present."""
        names = () if self.input_geometry is None else (self.input_geometry.input.name,)
        return names + tuple(group.description.name for group in self.parameters)

    def __deepcopy__(self, memo: dict[int, object]) -> InitialState:
        """Create owned inference storage once, without an intermediate tensor copy."""
        result = object.__new__(type(self))
        result.__setstate__(self.__dict__)
        return result

    @torch.inference_mode()
    def __setstate__(self, state: dict[str, object]) -> None:
        """Restore owned inference storage while preserving enclosing pickle identities."""
        geometry = state["input_geometry"]
        parameters = state["parameters"]
        if not isinstance(parameters, tuple):
            raise ValueError("InitialState.parameters must be a tuple")
        self.__dict__.update(
            input_geometry=None if geometry is None else _copy_parameter_group(geometry),
            parameters=tuple(_copy_parameter_group(group) for group in parameters),
            connection_sub_seeds=state["connection_sub_seeds"],
        )


@dataclass(frozen=True)
class _Trajectory:
    """One fit's loss and termination record, apart from any selection."""

    loss_history: tuple[float, ...]
    n_iter: int
    converged: bool
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class _SelectionHistory:
    """Every candidate's record, the winner's index, and the selection's warnings with their place.

    ``warnings`` follow the winner's first ``warnings_at``; empty ``outcomes`` mean it is unfinished.
    """

    outcomes: tuple[InitOutcome, ...]
    selected_index: int = 0
    warnings: tuple[str, ...] = ()
    warnings_at: int = 0


@dataclass(frozen=True)
class _FittedNetwork:
    """Fitted state, stopping controls and original initialisation."""

    graph: _CompiledGraph
    schema: DataSchema
    trajectory: _Trajectory
    predict_config: PredictConfig
    epsilon_P_source: Literal["supplied", "derived"] | None
    initial_state: InitialState
    Wt_ref: torch.Tensor | None
    max_iter: int
    tol: float


@dataclass(frozen=True, eq=False)
class _NetworkRecord:
    """The selected fit, its selection history, and its retained original states and members.

    ``states`` is ``None`` under ``retain="winner"`` and ``members`` is ``None`` unless
    members are retained; ``members[i]`` was fitted from ``states[i]``, and is ``fitted`` if it won.
    """

    fitted: _FittedNetwork
    selection: _SelectionHistory
    states: tuple[InitialState, ...] | None = None
    members: tuple[_FittedNetwork, ...] | None = None

    def __post_init__(self) -> None:
        """Require the selection's warnings to follow some of the selected fit's own."""
        if not 0 <= self.selection.warnings_at <= len(self.fitted.trajectory.warnings):
            raise ValueError("selection warnings are placed outside the selected fit's warnings")

    @property
    def winner(self) -> int:
        """Return the index of the selected fit's original state, ``0`` when none are retained.

        Raises:
            ValueError: If the retained state at the selected index is not the selected fit's.
        """
        if self.states is None:
            return 0
        index = self.selection.selected_index
        if index >= len(self.states) or self.states[index] is not self.fitted.initial_state:
            raise ValueError("no retained original state is the selected fit's initial state")
        return index
