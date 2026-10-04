"""Immutable descriptions of an entropic-learning graph.

A :class:`Recipe` names its blocks and connections and contains only configuration: no
data, no tensors and no fitted state. ``Block``, ``InputBlock`` and ``Head`` are the
unions the rest of the package dispatches on.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from graphlib import CycleError, TopologicalSorter
from itertools import pairwise
from numbers import Integral, Real
from typing import Any, Literal


def _safe_real_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        return float(value)
    except OverflowError:
        return None


def _require_positive_integer(value: object, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")


def _require_nonnegative_finite(value: object, field: str) -> None:
    scalar = _safe_real_float(value)
    if scalar is None or not math.isfinite(scalar) or scalar < 0:
        raise ValueError(f"{field} must be finite and non-negative")


def _require_weight_temperature(value: object, field: str) -> None:
    scalar = _safe_real_float(value)
    if scalar is None or math.isnan(scalar) or scalar < 0:
        raise ValueError(f"{field} must be non-negative and not NaN")


def _require_positive_finite(value: object, field: str) -> None:
    scalar = _safe_real_float(value)
    if scalar is None or not math.isfinite(scalar) or scalar <= 0:
        raise ValueError(f"{field} must be finite and positive")


def _require_theta_alpha(value: object, field: str) -> None:
    scalar = _safe_real_float(value)
    if scalar is None or not math.isfinite(scalar) or scalar < 1:
        raise ValueError(f"{field} must be finite and at least one")


def _is_positive_finite(value: object) -> bool:
    scalar = _safe_real_float(value)
    return scalar is not None and math.isfinite(scalar) and scalar > 0


def _require_centroid_controls(
    strategy: object, greedy_candidates: object, balanced: object
) -> None:
    if strategy not in ("kmeans++", "greedy-kmeans++"):
        raise ValueError("centroid_strategy must be 'kmeans++' or 'greedy-kmeans++'")
    if not isinstance(balanced, bool):
        raise ValueError("balanced must be a bool")
    if strategy == "kmeans++":
        if greedy_candidates is not None:
            raise ValueError("greedy_candidates must be None for kmeans++")
        return
    if greedy_candidates is None:
        return
    if (
        isinstance(greedy_candidates, bool)
        or not isinstance(greedy_candidates, Integral)
        or greedy_candidates < 2
    ):
        raise ValueError("greedy_candidates must be an integer of at least two for greedy-kmeans++")


class Coupling(StrEnum):
    """Which axis of a transition matrix sums to one.

    ``M`` constrains the columns, so ``theta[:, j]`` is a distribution over the target
    clusters given source cluster ``j``. ``S`` constrains the rows, so each row is one
    target cluster's prototype on the source affiliation simplex. Both deploy through the
    same rule, the target block's own assignment step against its incoming coupling, so
    the choice changes what ``theta`` means, never how the connection propagates.
    """

    M = "M"
    S = "S"


def _coerce_coupling(value: object) -> Coupling:
    try:
        return Coupling(value)
    except ValueError as error:
        raise ValueError("coupling must be 'M' or 'S'") from error


@dataclass(frozen=True)
class Input:
    """Description of a standard input block.

    Attributes:
        K: Number of clusters.
        epsilon: Affiliation temperature, how spread each instance's assignment over the
            clusters is: ``0`` (the default) gives hard affiliations,
            finite positive gives soft affiliations. Must lie in ``[0, inf)``.
        epsilon_D: Feature-weight temperature. Defaults to ``inf``, which holds the
            feature weights at their initial values and does not learn them. A finite
            positive value learns them every iteration. Must lie in ``[0, inf]``.
        epsilon_T: Instance-weight temperature. Defaults to ``inf``, which freezes the
            instance weights at the supplied sample weights (or uniform when none are
            given). A finite positive value learns them every iteration. Must lie in ``[0, inf]``.
        W_std: Standard deviation of the logistic-normal feature-weight initialisation.
            ``0`` (the default) starts from a uniform feature weighting.
        delta_cat: Scale of the categorical channel relative to the continuous one.
        centroid_strategy: Centroid-seeding strategy, ``"kmeans++"`` (the default) or
            ``"greedy-kmeans++"``, which keeps the best of several candidates per seed.
        greedy_candidates: Fixed per-seed candidate count for ``"greedy-kmeans++"``.
            ``None`` (the default) uses the K-dependent ``2 + floor(ln K)`` rule; an
            integer ``>= 2`` fixes the count. Valid only with ``"greedy-kmeans++"``.
        balanced: Whether classification seeding assigns initial clusters equally
            across classes. Regression rejects it.
        name: Block name.
    """

    K: int = 3
    epsilon: float = 0.0
    epsilon_D: float = math.inf
    epsilon_T: float = math.inf
    W_std: float = 0.0
    delta_cat: float = 1.0
    centroid_strategy: Literal["kmeans++", "greedy-kmeans++"] = "kmeans++"
    greedy_candidates: int | None = None
    balanced: bool = False
    name: str = ""

    def __post_init__(self) -> None:
        """Validate the input scalar domains."""
        _require_positive_integer(self.K, "K")
        _require_nonnegative_finite(self.epsilon, "epsilon")
        _require_weight_temperature(self.epsilon_D, "epsilon_D")
        _require_weight_temperature(self.epsilon_T, "epsilon_T")
        _require_nonnegative_finite(self.W_std, "W_std")
        _require_nonnegative_finite(self.delta_cat, "delta_cat")
        _require_centroid_controls(self.centroid_strategy, self.greedy_candidates, self.balanced)


@dataclass(frozen=True)
class ManifoldInput:
    """Description of a manifold input clustering block.

    A continuous-only variant of :class:`Input` that clusters using per-cluster
    local subspaces. Each cluster contains an
    orthonormal projector spanning a ``subspace_dimension``-dimensional hyperplane, and
    distances are measured according to cost ``alpha`` along the hyperplane and ``1 + alpha`` off it.

    Attributes:
        K: Number of clusters.
        subspace_dimension: Dimension of every cluster's tangent projector. An integer
            ``>= 1``; it must not exceed the continuous feature count.
        alpha: The in-plane cost ratio of the distance metric. Finite
            and ``>= 0``; ``0`` makes errors on the tangent plane free.
        epsilon: Affiliation temperature, as on :class:`Input`.
        epsilon_T: Instance-weight temperature, as on :class:`Input`.
        centroid_strategy: Centroid-seeding strategy, as on :class:`Input`.
        greedy_candidates: As on :class:`Input`.
        balanced: As on :class:`Input`.
        name: Block name.
    """

    K: int = 3
    subspace_dimension: int = 1
    alpha: float = 0.1
    epsilon: float = 0.0
    epsilon_T: float = math.inf
    centroid_strategy: Literal["kmeans++", "greedy-kmeans++"] = "kmeans++"
    greedy_candidates: int | None = None
    balanced: bool = False
    name: str = ""

    def __post_init__(self) -> None:
        """Validate the manifold-input scalar domains."""
        _require_positive_integer(self.K, "K")
        _require_positive_integer(self.subspace_dimension, "subspace_dimension")
        _require_nonnegative_finite(self.alpha, "alpha")
        _require_nonnegative_finite(self.epsilon, "epsilon")
        _require_weight_temperature(self.epsilon_T, "epsilon_T")
        _require_centroid_controls(self.centroid_strategy, self.greedy_candidates, self.balanced)


@dataclass(frozen=True)
class Hidden:
    """Description of a hidden clustering block.

    Attributes:
        K: Number of clusters.
        epsilon: Affiliation temperature, as in :class:`Input`.
        name: Block name.
    """

    K: int = 3
    epsilon: float = 0.0
    name: str = ""

    def __post_init__(self) -> None:
        """Validate the hidden-block attributes."""
        _require_positive_integer(self.K, "K")
        _require_nonnegative_finite(self.epsilon, "epsilon")


@dataclass(frozen=True)
class ClassificationHead:
    """Description of a classification head.

    Attributes:
        coupling: Which axis of the head's own transition matrix ``theta_out`` sums to
            one. ``M`` (the default) constrains its columns, so ``theta_out[:, k]`` is
            cluster ``k``'s distribution over the classes. ``S`` constrains its rows, making
            each row one class's prototype on the source affiliation simplex. A string
            ``"M"`` or ``"S"`` is accepted.
        n_classes: Declared number of classes. ``None`` (the default) infers it from the labelled
            target codes at fit time. A declared value keeps gaps in the class codes valid.
        name: Block name.
    """

    coupling: Coupling = Coupling.M
    n_classes: int | None = None
    name: str = ""

    def __post_init__(self) -> None:
        """Coerce the coupling and validate the declared class count."""
        object.__setattr__(self, "coupling", _coerce_coupling(self.coupling))
        if self.n_classes is not None:
            _require_positive_integer(self.n_classes, "n_classes")


@dataclass(frozen=True)
class RegressionHead:
    """Description of a regression head.

    Attributes:
        epsilon_M: Output-weight temperature. Defaults to ``inf``, which fixes the output
            weights ``W_M`` without learning them; uniform, or the fixed ``W_M`` when
            one is given. A finite positive value learns them every iteration, weighting
            the per-dimension residuals of a multi-output target.
        W_M: Fixed output weights, one positive value per output dimension, stored as a
            tuple of the values as supplied. A fit normalises them to sum to one when it
            stages the data. ``None`` (the default) learns the weights according to
            ``epsilon_M``.
        name: Block name.
    """

    epsilon_M: float = math.inf
    W_M: tuple[float, ...] | None = None
    name: str = ""

    def __post_init__(self) -> None:
        """Validate and normalise the fixed output weights."""
        _require_weight_temperature(self.epsilon_M, "epsilon_M")
        if self.W_M is None:
            return
        if isinstance(self.W_M, (str, bytes)) or not isinstance(self.W_M, Sequence):
            raise ValueError("W_M must be a numeric sequence")
        weights = tuple(self.W_M)
        if not weights or any(not _is_positive_finite(weight) for weight in weights):
            raise ValueError("W_M must be a non-empty finite positive numeric tuple")
        if not math.isinf(self.epsilon_M):
            raise ValueError("fixed W_M requires epsilon_M to be infinite")
        object.__setattr__(self, "W_M", weights)


# Type aliases
Block = Input | ManifoldInput | Hidden | ClassificationHead | RegressionHead
InputBlock = Input | ManifoldInput
Head = ClassificationHead | RegressionHead
Numeric = float | int


@dataclass(frozen=True)
class Connection:
    """Description of a structural connection between two blocks.

    A connection into a hidden block configures transition state owned by that target, consisting of
    the coupling and pseudocount. A connection into a head has neither. The
    head owns its output parameters and consumes only ``delta``.

    Attributes:
        name: Connection name.
        source: Name of the upstream block.
        target: Name of the downstream block.
        delta: Coupling strength between the two blocks. Finite and positive; defaults to ``1``.
        coupling: Which axis of the transition matrix sums to one; see :class:`Coupling`.
            When the target is a hidden block, ``None`` (the default) becomes ``M`` in the
            :class:`Recipe`, which stores the resolved value. A connection into a head
            cannot carry a coupling. A string ``"M"`` or ``"S"`` is accepted.
        theta_alpha: Dirichlet pseudocount strength for the transition matrix; ``1``
            disables the prior exactly. Finite and ``>= 1``. When the target is a hidden
            block, ``None`` (the default) becomes ``1`` in the :class:`Recipe`, which
            stores the resolved value. A connection into a head cannot carry it.
    """

    name: str
    source: str
    target: str
    delta: float = 1.0
    coupling: Coupling | None = None
    theta_alpha: float | None = None

    def __post_init__(self) -> None:
        """Validate the connection names and scalar domains."""
        if not all(
            isinstance(value, str) and value for value in (self.name, self.source, self.target)
        ):
            raise ValueError("connection name, source and target must be non-empty strings")
        _require_positive_finite(self.delta, "delta")
        if self.coupling is not None:
            object.__setattr__(self, "coupling", _coerce_coupling(self.coupling))
        if self.theta_alpha is not None:
            _require_theta_alpha(self.theta_alpha, "theta_alpha")


@dataclass(frozen=True)
class Recipe:
    """An immutable, directed acyclic graph of block and connection descriptions.

    Blocks and connections are stored as tuples in declared order, and the declared order
    is the primary stable graph order (see :meth:`stable_order`). Construction validates
    names, references, block kinds, connection fields, source and head placement,
    acyclicity and task combinations, and raises ``ValueError`` on the first violation. On
    every hidden-target connection it stores the defaults for the fields left as ``None``:
    ``M`` for the coupling and ``1`` for ``theta_alpha``, as :meth:`chain` does. A
    structurally valid graph the builder cannot yet execute is accepted here and rejected
    once at ``Network`` entry.

    Attributes:
        blocks: The block descriptions, in declared order.
        connections: The connection descriptions, in declared order.
    """

    blocks: tuple[Block, ...]
    connections: tuple[Connection, ...]

    def __post_init__(self) -> None:
        """Canonicalise descriptions and validate graph structure."""
        object.__setattr__(self, "blocks", tuple(self.blocks))
        object.__setattr__(self, "connections", tuple(self.connections))
        if not all(
            type(block) in {Input, ManifoldInput, Hidden, ClassificationHead, RegressionHead}
            for block in self.blocks
        ):
            raise ValueError("blocks must be valid Recipe block descriptions")
        if not all(type(connection) is Connection for connection in self.connections):
            raise ValueError("connections must be valid Recipe connection descriptions")
        self._validate_names(self.blocks, "block")
        self._validate_names(self.connections, "connection")

        blocks = {block.name: block for block in self.blocks}
        if not any(isinstance(block, InputBlock) for block in self.blocks):
            raise ValueError("a Recipe requires at least one input block")
        heads = [block for block in self.blocks if isinstance(block, Head)]
        if not heads:
            raise ValueError("a Recipe requires at least one head")
        if len({type(head) for head in heads}) != 1:
            raise ValueError("a Recipe cannot mix classification and regression heads")
        if isinstance(heads[0], RegressionHead) and any(
            block.balanced for block in self.blocks if isinstance(block, InputBlock)
        ):
            raise ValueError("regression recipes cannot use balanced input initialisation")

        predecessors = {name: set() for name in blocks}
        successors = {name: set() for name in blocks}
        connections = []
        for connection in self.connections:
            if connection.source not in blocks or connection.target not in blocks:
                raise ValueError("connection endpoints must name Recipe blocks")
            if connection.source == connection.target:
                raise ValueError(
                    "a connection cannot join a block to itself (recursion is not allowed)"
                )
            source = blocks[connection.source]
            target = blocks[connection.target]
            if isinstance(source, Head):
                raise ValueError("a head cannot be a connection source")
            if isinstance(target, InputBlock):
                raise ValueError("an input block cannot be a connection target")
            if isinstance(target, Hidden):
                coupling, theta_alpha = connection.coupling, connection.theta_alpha
                if coupling is None or theta_alpha is None:
                    connection = replace(
                        connection,
                        coupling=Coupling.M if coupling is None else coupling,
                        theta_alpha=1.0 if theta_alpha is None else theta_alpha,
                    )
            elif connection.coupling is not None or connection.theta_alpha is not None:
                raise ValueError("a head-target connection cannot carry coupling or theta_alpha")
            connections.append(connection)
            predecessors[connection.target].add(connection.source)
            successors[connection.source].add(connection.target)
        object.__setattr__(self, "connections", tuple(connections))
        self.stable_order()
        for name, block in blocks.items():
            incoming = predecessors[name]
            outgoing = successors[name]
            if not incoming and not isinstance(block, InputBlock):
                raise ValueError("every graph source must be an input block")
            if isinstance(block, InputBlock) and not outgoing:
                raise ValueError("every input block must have an outgoing connection")
            if not outgoing and not isinstance(block, Head):
                raise ValueError("every graph output must be a head")

    def stable_order(self) -> tuple[str, ...]:
        """Return the block names in topological order. Declared order breaks ties.

        Raises:
            ValueError: If the connections form a cycle.
        """
        index = {block.name: position for position, block in enumerate(self.blocks)}
        predecessors = {name: set[str]() for name in index}
        for connection in self.connections:
            predecessors[connection.target].add(connection.source)
        sorter = TopologicalSorter(predecessors)
        try:
            sorter.prepare()
        except CycleError as error:
            raise ValueError("Recipe connections must be acyclic") from error
        order: list[str] = []
        while sorter.is_active():
            ready = sorted(sorter.get_ready(), key=lambda name: index[name])
            order.extend(ready)
            sorter.done(*ready)
        return tuple(order)

    def replace_block(self, name: str, /, **changes: Any) -> Recipe:
        """Return a Recipe with one block replaced. Every other description is shared.

        Raises:
            KeyError: If no block has ``name``.
            ValueError: If the replacement makes the Recipe invalid.
        """
        if name not in {block.name for block in self.blocks}:
            raise KeyError(f"no block named {name!r}")
        blocks = tuple(
            replace(block, **changes) if block.name == name else block for block in self.blocks
        )
        return replace(self, blocks=blocks)

    def replace_connection(self, name: str, /, **changes: Any) -> Recipe:
        """Return a Recipe with one connection replaced. Every other description is shared.

        Raises:
            KeyError: If no connection has ``name``.
            ValueError: If the replacement makes the Recipe invalid.
        """
        if name not in {connection.name for connection in self.connections}:
            raise KeyError(f"no connection named {name!r}")
        connections = tuple(
            replace(connection, **changes) if connection.name == name else connection
            for connection in self.connections
        )
        return replace(self, connections=connections)

    @classmethod
    def chain(
        cls,
        *blocks: Block,
        coupling: Coupling | str | Sequence[Coupling | str] | None = None,
        delta: Numeric | Sequence[Numeric] = 1.0,
        theta_alpha: Numeric | Sequence[Numeric] = 1.0,
    ) -> Recipe:
        """Create a Recipe as a chain of input-to-head block descriptions.

        Each block takes a positional default name (``"input"``, ``"hidden_1"``, ...,
        ``"output"``) unless it provides one, and consecutive blocks are joined by a
        connection named ``"{source}_to_{target}"`` by default. ``delta`` applies to every
        connection, and ``coupling`` and ``theta_alpha`` to every hidden-target
        connection. Each is one value broadcast over those connections or one value per
        connection, in order. By default every hidden-target connection takes
        ``coupling="M"``. a chain without a hidden block rejects ``coupling``.

        Args:
            *blocks: The block descriptions in chain order, an input block first and a
                head last.
            coupling: Coupling for every hidden-target connection. ``None`` (the default)
                gives ``M``.
            delta: Coupling strength for every connection.
            theta_alpha: Pseudocount strength for every hidden-target connection.

        Returns:
            The graph-shaped description.

        Raises:
            ValueError: If the blocks do not form an input-to-head chain, or a
                per-connection value is invalid or a sequence has the wrong length.
        """
        if not all(isinstance(block, Block) for block in blocks):
            raise ValueError("a chain requires Recipe block descriptions")
        if (
            len(blocks) < 2
            or not isinstance(blocks[0], InputBlock)
            or not isinstance(blocks[-1], Head)
        ):
            raise ValueError("a chain must start with an input block and end with a head")
        if any(isinstance(block, InputBlock) for block in blocks[1:]) or any(
            isinstance(block, Head) for block in blocks[:-1]
        ):
            raise ValueError("a chain cannot contain an interior input block or head")

        named_blocks: list[Block] = []
        hidden_index = 0
        for block in blocks:
            if isinstance(block, Hidden):
                hidden_index += 1
            if block.name:
                named_blocks.append(block)
            elif isinstance(block, InputBlock):
                named_blocks.append(replace(block, name="input"))
            elif isinstance(block, Hidden):
                named_blocks.append(replace(block, name=f"hidden_{hidden_index}"))
            else:
                named_blocks.append(replace(block, name="output"))

        hidden_couplings = cls._hidden_couplings(coupling, hidden_index)
        deltas = cls._connection_scalars(
            delta, len(named_blocks) - 1, "delta", _require_positive_finite
        )
        theta_alphas = cls._connection_scalars(
            theta_alpha, hidden_index, "theta_alpha", _require_theta_alpha
        )
        hidden = zip(hidden_couplings, theta_alphas, strict=True)
        connections: list[Connection] = []
        for (source, target), strength in zip(pairwise(named_blocks), deltas, strict=True):
            connection_coupling, connection_theta_alpha = (
                next(hidden) if isinstance(target, Hidden) else (None, None)
            )
            connections.append(
                Connection(
                    name=f"{source.name}_to_{target.name}",
                    source=source.name,
                    target=target.name,
                    delta=strength,
                    coupling=connection_coupling,
                    theta_alpha=connection_theta_alpha,
                )
            )
        return cls(blocks=tuple(named_blocks), connections=tuple(connections))

    @staticmethod
    def _connection_scalars(
        value: Numeric | Sequence[Numeric],
        count: int,
        field: str,
        validate: Callable[[object, str], None],
    ) -> tuple[float, ...]:
        if isinstance(value, Real):
            validate(value, field)
            return (float(value),) * count
        elif isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise ValueError(f"{field} must be a scalar or a sequence")
        else:
            values = tuple(value)
        if len(values) != count:
            raise ValueError(f"{field} must provide one value per generated connection")
        for scalar in values:
            validate(scalar, field)
        return tuple(float(scalar) for scalar in values)

    @staticmethod
    def _hidden_couplings(
        coupling: Coupling | str | Sequence[Coupling | str] | None, count: int
    ) -> tuple[Coupling | None, ...]:
        if coupling is None:
            return (None,) * count
        if count == 0:
            raise ValueError("a chain without hidden blocks cannot receive coupling")
        if isinstance(coupling, str):
            return (_coerce_coupling(coupling),) * count
        if isinstance(coupling, bytes) or not isinstance(coupling, Sequence):
            raise ValueError("coupling must be a Coupling or a sequence of Coupling values")
        values = tuple(_coerce_coupling(value) for value in coupling)
        if len(values) != count:
            raise ValueError("coupling must provide one Coupling value per hidden block")
        return values

    @staticmethod
    def _validate_names(values: Sequence[object], kind: str) -> None:
        names = []
        for value in values:
            name = getattr(value, "name", None)
            if not isinstance(name, str) or not name:
                raise ValueError(f"{kind} names must be non-empty strings")
            if "__" in name:
                raise ValueError(f"{kind} names cannot contain '__'")
            if name.startswith("_") or name.endswith("_"):
                raise ValueError(f"{kind} names cannot start or end with '_'")
            names.append(name)
        if len(set(names)) != len(names):
            raise ValueError(f"{kind} names must be unique")
