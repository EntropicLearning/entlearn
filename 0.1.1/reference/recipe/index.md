# Recipe

A `Recipe` describes the blocks, connections and settings of a model.

Two public aliases name a block's role in type annotations: `entlearn.InputBlock` is `Input | ManifoldInput`, and `entlearn.Head` is `ClassificationHead | RegressionHead`. Both aliases are unions of the block description types listed below.

## entlearn.Recipe

```
Recipe(blocks, connections)
```

An immutable, directed acyclic graph of block and connection descriptions.

Blocks and connections are stored as tuples in declared order, and the declared order is the primary stable graph order (see :meth:`stable_order`). Construction validates names, references, block kinds, connection fields, source and head placement, acyclicity and task combinations, and raises `ValueError` on the first violation. On every hidden-target connection it stores the defaults for the fields left as `None`: `M` for the coupling and `1` for `theta_alpha`, as :meth:`chain` does. A connection's `delta_ratio` and a hidden block's `epsilon_ratio` are stored as provided. :meth:`resolve_ratios` returns a recipe with the equivalent `epsilon` and `delta` values. A structurally valid graph the builder cannot yet execute is accepted here and rejected at `Network` entry.

Attributes:

| Name          | Type                     | Description                                     |
| ------------- | ------------------------ | ----------------------------------------------- |
| `blocks`      | `tuple[Block, ...]`      | The block descriptions, in declared order.      |
| `connections` | `tuple[Connection, ...]` | The connection descriptions, in declared order. |

### __post_init__

```
__post_init__()
```

Canonicalise descriptions and validate graph structure.

Source code in `src/entlearn/recipe.py`

```
def __post_init__(self) -> None:  # noqa: C901
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
    self.resolve_ratios()  # raises on a ratio the graph cannot resolve
```

### chain

```
chain(
    *blocks,
    coupling=None,
    delta=1.0,
    theta_alpha=1.0,
    delta_ratio=None,
)
```

Create a Recipe as a chain of input-to-head block descriptions.

Each block takes a positional default name (`"input"`, `"hidden_1"`, ..., `"output"`) unless it provides one, and consecutive blocks are joined by a connection named `"{source}_to_{target}"` by default. `delta` applies to every connection without a ratio, and `coupling` and `theta_alpha` to every hidden-target connection. Each is one value broadcast over those connections or one value per connection, in order. By default every hidden-target connection takes `coupling="M"`. A chain without a hidden block rejects `coupling`.

`delta_ratio` describes a connection's `delta` as a multiple of the previous connection's, as on :class:`Connection`. One value applies to every connection after the first, so `delta_ratio=3` makes each `delta` three times the one before it. A sequence holds one value per connection in order, `None` where `delta` gives the strength. `delta` then gives only the connections without a ratio: one value applies to each of them, and a sequence holds `None` where a ratio gives the strength.

Parameters:

| Name          | Type       | Description                                                                  | Default                                                  |
| ------------- | ---------- | ---------------------------------------------------------------------------- | -------------------------------------------------------- |
| `*blocks`     | `Block`    | The block descriptions in chain order, an input block first and a head last. | `()`                                                     |
| `coupling`    | \`Coupling | str                                                                          | Sequence\[Coupling                                       |
| `delta`       | \`Numeric  | Sequence\[Numeric                                                            | None\]\`                                                 |
| `theta_alpha` | \`Numeric  | Sequence[Numeric]\`                                                          | Pseudocount strength for every hidden-target connection. |
| `delta_ratio` | \`Numeric  | Sequence\[Numeric                                                            | None\]                                                   |

Returns:

| Type     | Description                   |
| -------- | ----------------------------- |
| `Recipe` | The graph-shaped description. |

Raises:

| Type         | Description                                                                                                                |
| ------------ | -------------------------------------------------------------------------------------------------------------------------- |
| `ValueError` | If the blocks do not form an input-to-head chain, or a per-connection value is invalid or a sequence has the wrong length. |

Source code in `src/entlearn/recipe.py`

```
@classmethod
def chain(
    cls,
    *blocks: Block,
    coupling: Coupling | str | Sequence[Coupling | str] | None = None,
    delta: Numeric | Sequence[Numeric | None] = 1.0,
    theta_alpha: Numeric | Sequence[Numeric] = 1.0,
    delta_ratio: Numeric | Sequence[Numeric | None] | None = None,
) -> Recipe:
    """Create a Recipe as a chain of input-to-head block descriptions.

    Each block takes a positional default name (``"input"``, ``"hidden_1"``, ...,
    ``"output"``) unless it provides one, and consecutive blocks are joined by a
    connection named ``"{source}_to_{target}"`` by default. ``delta`` applies to every
    connection without a ratio, and ``coupling`` and ``theta_alpha`` to every hidden-target
    connection. Each is one value broadcast over those connections or one value per
    connection, in order. By default every hidden-target connection takes
    ``coupling="M"``. A chain without a hidden block rejects ``coupling``.

    ``delta_ratio`` describes a connection's ``delta`` as a multiple of the previous
    connection's, as on :class:`Connection`. One value applies to every connection
    after the first, so ``delta_ratio=3`` makes each ``delta`` three times the one
    before it. A sequence holds one value per connection in order, ``None`` where
    ``delta`` gives the strength. ``delta`` then gives only the connections without a
    ratio: one value applies to each of them, and a sequence holds ``None`` where a
    ratio gives the strength.

    Args:
        *blocks: The block descriptions in chain order, an input block first and a
            head last.
        coupling: Coupling for every hidden-target connection. ``None`` (the default)
            gives ``M``.
        delta: Coupling strength for every connection without a ratio.
        theta_alpha: Pseudocount strength for every hidden-target connection.
        delta_ratio: Each connection's ``delta`` divided by the previous
            connection's. ``None`` (the default) gives no ratios.

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
    count = len(named_blocks) - 1
    deltas, ratios = cls._chain_strengths(delta, delta_ratio, count)
    theta_alphas = cls._connection_scalars(
        theta_alpha, hidden_index, "theta_alpha", _require_theta_alpha
    )
    hidden = zip(hidden_couplings, theta_alphas, strict=True)
    connections: list[Connection] = []
    for (source, target), strength, ratio in zip(
        pairwise(named_blocks), deltas, ratios, strict=True
    ):
        connection_coupling, connection_theta_alpha = (
            next(hidden) if isinstance(target, Hidden) else (None, None)
        )
        connections.append(
            Connection(
                name=f"{source.name}_to_{target.name}",
                source=source.name,
                target=target.name,
                delta=strength,
                delta_ratio=ratio,
                coupling=connection_coupling,
                theta_alpha=connection_theta_alpha,
            )
        )
    return cls(blocks=tuple(named_blocks), connections=tuple(connections))
```

### replace_block

```
replace_block(name, /, **changes)
```

Return a Recipe with one block replaced. Every other description is shared.

Raises:

| Type         | Description                                  |
| ------------ | -------------------------------------------- |
| `KeyError`   | If no block has name.                        |
| `ValueError` | If the replacement makes the Recipe invalid. |

Source code in `src/entlearn/recipe.py`

```
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
```

### replace_connection

```
replace_connection(name, /, **changes)
```

Return a Recipe with one connection replaced. Every other description is shared.

Raises:

| Type         | Description                                  |
| ------------ | -------------------------------------------- |
| `KeyError`   | If no connection has name.                   |
| `ValueError` | If the replacement makes the Recipe invalid. |

Source code in `src/entlearn/recipe.py`

```
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
```

### resolve_ratios

```
resolve_ratios()
```

Return the Recipe with all ratios replaced by the equivalent explicit value.

Every resolved connection sets `delta_ratio=None`. A chain of ratios resolves from its first explicit `delta`. A hidden block that sets `epsilon_ratio` takes `epsilon = epsilon_ratio * delta`, using the resolved `delta`. Every resolved block sets `epsilon_ratio=None`. A fit uses these values, and the fitted `Network` keeps the original Recipe. A Recipe without a ratio is returned unchanged.

Source code in `src/entlearn/recipe.py`

```
def resolve_ratios(self) -> Recipe:
    """Return the Recipe with all ratios replaced by the equivalent explicit value.

    Every resolved connection sets ``delta_ratio=None``. A chain of ratios resolves from its first
    explicit ``delta``. A hidden block that sets ``epsilon_ratio`` takes
    ``epsilon = epsilon_ratio * delta``, using the resolved ``delta``. Every resolved
    block sets ``epsilon_ratio=None``. A fit uses these values, and the fitted
    ``Network`` keeps the original Recipe. A Recipe without a ratio is returned unchanged.
    """
    deltas = self._resolved_deltas()
    blocks = tuple(
        replace(block, epsilon=self._ratio_epsilon(block, deltas), epsilon_ratio=None)
        if isinstance(block, Hidden) and block.epsilon_ratio is not None
        else block
        for block in self.blocks
    )
    connections = tuple(
        replace(connection, delta=deltas[connection.name], delta_ratio=None)
        if connection.delta_ratio is not None
        else connection
        for connection in self.connections
    )
    if blocks == self.blocks and connections == self.connections:
        return self
    return replace(self, blocks=blocks, connections=connections)
```

### stable_order

```
stable_order()
```

Return the block names in topological order. Declared order breaks ties.

Raises:

| Type         | Description                      |
| ------------ | -------------------------------- |
| `ValueError` | If the connections form a cycle. |

Source code in `src/entlearn/recipe.py`

```
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
```

## entlearn.Input

```
Input(
    K=3,
    epsilon=0.0,
    epsilon_D=inf,
    epsilon_T=inf,
    W_std=0.0,
    delta_cat=None,
    centroid_strategy="kmeans++",
    greedy_candidates=None,
    balanced=False,
    name="",
)
```

Description of a standard input block.

Attributes:

| Name                | Type                                     | Description                                                                                                                                                                                                                                                  |
| ------------------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `K`                 | `int`                                    | Number of clusters.                                                                                                                                                                                                                                          |
| `epsilon`           | `float`                                  | Affiliation temperature, how spread each instance's assignment over the clusters is: 0 (the default) gives hard affiliations, finite positive gives soft affiliations. Must lie in \[0, inf).                                                                |
| `epsilon_D`         | `float`                                  | Feature-weight temperature. Defaults to inf, which holds the feature weights at their initial values and does not learn them. A finite positive value learns them every iteration. Must lie in [0, inf].                                                     |
| `epsilon_T`         | `float`                                  | Instance-weight temperature. Defaults to inf, which freezes the instance weights at the supplied sample weights (or uniform when none are given). A finite positive value learns them every iteration, relative to the sample weights. Must lie in [0, inf]. |
| `W_std`             | `float`                                  | Standard deviation of the logistic-normal feature-weight initialisation. 0 (the default) starts from a uniform feature weighting.                                                                                                                            |
| `delta_cat`         | \`float                                  | None\`                                                                                                                                                                                                                                                       |
| `centroid_strategy` | `Literal['kmeans++', 'greedy-kmeans++']` | Centroid-seeding strategy, "kmeans++" (the default) or "greedy-kmeans++", which keeps the best of several candidates per seed.                                                                                                                               |
| `greedy_candidates` | \`int                                    | None\`                                                                                                                                                                                                                                                       |
| `balanced`          | `bool`                                   | Whether classification seeding assigns initial clusters equally across classes. Regression rejects it.                                                                                                                                                       |
| `name`              | `str`                                    | Block name.                                                                                                                                                                                                                                                  |

### __post_init__

```
__post_init__()
```

Validate the input scalar domains.

Source code in `src/entlearn/recipe.py`

```
def __post_init__(self) -> None:
    """Validate the input scalar domains."""
    _require_positive_integer(self.K, "K")
    _require_nonnegative_finite(self.epsilon, "epsilon")
    _require_weight_temperature(self.epsilon_D, "epsilon_D")
    _require_weight_temperature(self.epsilon_T, "epsilon_T")
    _require_nonnegative_finite(self.W_std, "W_std")
    if self.delta_cat is not None:
        _require_nonnegative_finite(self.delta_cat, "delta_cat")
    _require_centroid_controls(self.centroid_strategy, self.greedy_candidates, self.balanced)
```

## entlearn.ManifoldInput

```
ManifoldInput(
    K=3,
    subspace_dimension=1,
    alpha=0.1,
    epsilon=0.0,
    epsilon_T=inf,
    centroid_strategy="kmeans++",
    greedy_candidates=None,
    balanced=False,
    name="",
)
```

Description of a manifold input clustering block.

A continuous-only variant of :class:`Input` that clusters using per-cluster local subspaces. Each cluster contains an orthonormal projector spanning a `subspace_dimension`-dimensional hyperplane, and distances are measured according to cost `alpha` along the hyperplane and `1 + alpha` off it, averaged over the continuous features as the standard input's distance is.

Attributes:

| Name                 | Type                                     | Description                                                                                                       |
| -------------------- | ---------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `K`                  | `int`                                    | Number of clusters.                                                                                               |
| `subspace_dimension` | `int`                                    | Dimension of every cluster's tangent projector. An integer >= 1; it must not exceed the continuous feature count. |
| `alpha`              | `float`                                  | The in-plane cost ratio of the distance metric. Finite and >= 0; 0 makes errors on the tangent plane free.        |
| `epsilon`            | `float`                                  | Affiliation temperature, as on :class:Input.                                                                      |
| `epsilon_T`          | `float`                                  | Instance-weight temperature, as on :class:Input.                                                                  |
| `centroid_strategy`  | `Literal['kmeans++', 'greedy-kmeans++']` | Centroid-seeding strategy, as on :class:Input.                                                                    |
| `greedy_candidates`  | \`int                                    | None\`                                                                                                            |
| `balanced`           | `bool`                                   | As on :class:Input.                                                                                               |
| `name`               | `str`                                    | Block name.                                                                                                       |

### __post_init__

```
__post_init__()
```

Validate the manifold-input scalar domains.

Source code in `src/entlearn/recipe.py`

```
def __post_init__(self) -> None:
    """Validate the manifold-input scalar domains."""
    _require_positive_integer(self.K, "K")
    _require_positive_integer(self.subspace_dimension, "subspace_dimension")
    _require_nonnegative_finite(self.alpha, "alpha")
    _require_nonnegative_finite(self.epsilon, "epsilon")
    _require_weight_temperature(self.epsilon_T, "epsilon_T")
    _require_centroid_controls(self.centroid_strategy, self.greedy_candidates, self.balanced)
```

## entlearn.Hidden

```
Hidden(K=3, epsilon=None, epsilon_ratio=None, name='')
```

Description of a hidden clustering block.

A hidden block's affiliation temperature can be given either as a value, `epsilon`, or as a ratio to the strength of the incoming connection, `epsilon_ratio`.

Attributes:

| Name            | Type    | Description         |
| --------------- | ------- | ------------------- |
| `K`             | `int`   | Number of clusters. |
| `epsilon`       | \`float | None\`              |
| `epsilon_ratio` | \`float | None\`              |
| `name`          | `str`   | Block name.         |

### __post_init__

```
__post_init__()
```

Validate the hidden-block attributes.

Source code in `src/entlearn/recipe.py`

```
def __post_init__(self) -> None:
    """Validate the hidden-block attributes."""
    _require_positive_integer(self.K, "K")
    if self.epsilon_ratio is None:
        if self.epsilon is None:
            object.__setattr__(self, "epsilon", 0.0)
        _require_nonnegative_finite(self.epsilon, "epsilon")
    else:
        _require_nonnegative_finite(self.epsilon_ratio, "epsilon_ratio")
        if self.epsilon is not None:
            raise ValueError(
                "epsilon_ratio cannot be combined with an explicit epsilon; set "
                "epsilon=None, or epsilon_ratio=None to use epsilon"
            )
```

## entlearn.ClassificationHead

```
ClassificationHead(
    coupling=M, n_classes=None, theta_alpha=1.0, name=""
)
```

Description of a classification head.

Attributes:

| Name          | Type       | Description                                                                                                                                                                                                                                                                                                     |
| ------------- | ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `coupling`    | `Coupling` | Which axis of the head's own transition matrix theta_out sums to one. M (the default) constrains its columns, so theta_out[:, k] is cluster k's distribution over the classes. S constrains its rows, making each row one class's prototype on the source affiliation simplex. A string "M" or "S" is accepted. |
| `n_classes`   | \`int      | None\`                                                                                                                                                                                                                                                                                                          |
| `theta_alpha` | `float`    | Dirichlet pseudocount strength for theta_out; 1 (the default) disables the prior exactly. Finite and >= 1. A value above one keeps every class probability positive, so a class a cluster has not seen costs a finite amount rather than -log of the dtype's smallest normal number.                            |
| `name`        | `str`      | Block name.                                                                                                                                                                                                                                                                                                     |

### __post_init__

```
__post_init__()
```

Coerce the coupling and validate the declared class count and prior.

Source code in `src/entlearn/recipe.py`

```
def __post_init__(self) -> None:
    """Coerce the coupling and validate the declared class count and prior."""
    object.__setattr__(self, "coupling", _coerce_coupling(self.coupling))
    if self.n_classes is not None:
        _require_positive_integer(self.n_classes, "n_classes")
    _require_theta_alpha(self.theta_alpha, "theta_alpha")
```

## entlearn.RegressionHead

```
RegressionHead(epsilon_M=inf, W_M=None, name='')
```

Description of a regression head.

Attributes:

| Name        | Type                | Description                                                                                                                                                                                                                                                             |
| ----------- | ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `epsilon_M` | `float`             | Output-weight temperature. Defaults to inf, which fixes the output weights W_M without learning them; uniform, or the fixed W_M when one is given. A finite positive value learns them every iteration, weighting the per-dimension residuals of a multi-output target. |
| `W_M`       | \`tuple[float, ...] | None\`                                                                                                                                                                                                                                                                  |
| `name`      | `str`               | Block name.                                                                                                                                                                                                                                                             |

### __post_init__

```
__post_init__()
```

Validate and normalise the fixed output weights.

Source code in `src/entlearn/recipe.py`

```
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
```

## entlearn.Connection

```
Connection(
    name,
    source,
    target,
    delta=None,
    delta_ratio=None,
    coupling=None,
    theta_alpha=None,
)
```

Description of a structural connection between two blocks.

A connection into a hidden block configures transition state owned by that target, consisting of the coupling and pseudocount. A connection into a head has neither. The head owns its output parameters, and a classification head its own coupling and pseudocount, so the connection owns only `delta`.

The coupling strength can be given either as a value, `delta`, or as a ratio to the strength of the previous connection, `delta_ratio`.

Attributes:

| Name          | Type       | Description                   |
| ------------- | ---------- | ----------------------------- |
| `name`        | `str`      | Connection name.              |
| `source`      | `str`      | Name of the upstream block.   |
| `target`      | `str`      | Name of the downstream block. |
| `delta`       | \`float    | None\`                        |
| `delta_ratio` | \`float    | None\`                        |
| `coupling`    | \`Coupling | None\`                        |
| `theta_alpha` | \`float    | None\`                        |

### __post_init__

```
__post_init__()
```

Validate the connection names and scalar domains.

Source code in `src/entlearn/recipe.py`

```
def __post_init__(self) -> None:
    """Validate the connection names and scalar domains."""
    if not all(
        isinstance(value, str) and value for value in (self.name, self.source, self.target)
    ):
        raise ValueError("connection name, source and target must be non-empty strings")
    if self.delta_ratio is None:
        if self.delta is None:
            object.__setattr__(self, "delta", 1.0)
        _require_positive_finite(self.delta, "delta")
    else:
        _require_positive_finite(self.delta_ratio, "delta_ratio")
        if self.delta is not None:
            raise ValueError(
                "delta_ratio cannot be combined with an explicit delta; set delta=None, "
                "or delta_ratio=None to use delta"
            )
    if self.coupling is not None:
        object.__setattr__(self, "coupling", _coerce_coupling(self.coupling))
    if self.theta_alpha is not None:
        _require_theta_alpha(self.theta_alpha, "theta_alpha")
```

## entlearn.Coupling

Bases: `StrEnum`

Which axis of a transition matrix sums to one.

`M` constrains the columns, so `theta[:, j]` is a distribution over the target clusters given source cluster `j`. `S` constrains the rows, so each row is one target cluster's prototype on the source affiliation simplex. Both deploy through the same rule, the target block's own assignment step against its incoming coupling, so the choice changes what `theta` means, never how the connection propagates.
