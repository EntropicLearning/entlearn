# Plotting

Install `entlearn[plotting]` for these Plotly figures and for [`Network.plot`](https://entropiclearning.github.io/entlearn/0.1.0/reference/network/#entlearn.Network.plot). See the [plotting guide](https://entropiclearning.github.io/entlearn/0.1.0/guide/plotting/index.md) for examples.

## entlearn.plotting.plot_loss

```
plot_loss(loss_history, *, fig=None, row=None, col=None)
```

Plot the loss history.

Parameters:

| Name           | Type              | Description                                              | Default                                                                      |
| -------------- | ----------------- | -------------------------------------------------------- | ---------------------------------------------------------------------------- |
| `loss_history` | `Sequence[float]` | The loss values, as in network.diagnostics.loss_history. | *required*                                                                   |
| `fig`          | \`Figure          | None\`                                                   | A figure to draw into, such as one from make_subplots; by default a new one. |
| `row`          | \`int             | None\`                                                   | The subplot's row in fig.                                                    |
| `col`          | \`int             | None\`                                                   | The subplot's column in fig.                                                 |

Source code in `src/entlearn/plotting/basic.py`

```
def plot_loss(
    loss_history: Sequence[float],
    *,
    fig: go.Figure | None = None,
    row: int | None = None,
    col: int | None = None,
) -> go.Figure:
    """Plot the loss history.

    Args:
        loss_history: The loss values, as in ``network.diagnostics.loss_history``.
        fig: A figure to draw into, such as one from ``make_subplots``; by default a
            new one.
        row: The subplot's row in ``fig``.
        col: The subplot's column in ``fig``.
    """
    figure, at = _target(fig, row, col)
    figure.add_trace(
        go.Scatter(y=list(loss_history), mode="lines+markers", name="loss", showlegend=False), **at
    )
    figure.update_xaxes(title_text="iteration", **at)
    figure.update_yaxes(title_text=r"$\mathcal{L}$", **at)
    return figure
```

## entlearn.plotting.plot_feature_importance

```
plot_feature_importance(
    weights, names=None, *, fig=None, row=None, col=None
)
```

Plot one bar per feature weight.

Parameters:

| Name      | Type            | Description                                                                                                          | Default                                                                      |
| --------- | --------------- | -------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| `weights` | `Any`           | One weight per feature, such as network.inspect("feature_weights")["input"] or an estimator's feature_importances\_. | *required*                                                                   |
| `names`   | \`Sequence[str] | None\`                                                                                                               | Feature names in the same order; by default the feature indices.             |
| `fig`     | \`Figure        | None\`                                                                                                               | A figure to draw into, such as one from make_subplots; by default a new one. |
| `row`     | \`int           | None\`                                                                                                               | The subplot's row in fig.                                                    |
| `col`     | \`int           | None\`                                                                                                               | The subplot's column in fig.                                                 |

Source code in `src/entlearn/plotting/basic.py`

```
def plot_feature_importance(
    weights: Any,
    names: Sequence[str] | None = None,
    *,
    fig: go.Figure | None = None,
    row: int | None = None,
    col: int | None = None,
) -> go.Figure:
    """Plot one bar per feature weight.

    Args:
        weights: One weight per feature, such as
            ``network.inspect("feature_weights")["input"]`` or an estimator's
            ``feature_importances_``.
        names: Feature names in the same order; by default the feature indices.
        fig: A figure to draw into, such as one from ``make_subplots``; by default a
            new one.
        row: The subplot's row in ``fig``.
        col: The subplot's column in ``fig``.
    """
    values = _numpy(weights)
    figure, at = _target(fig, row, col)
    labels = list(names) if names is not None else [str(d) for d in range(len(values))]
    figure.add_trace(go.Bar(x=labels, y=values, name="W", showlegend=False), **at)
    figure.update_xaxes(title_text="feature", type="category", **at)
    figure.update_yaxes(title_text=r"$w$", **at)
    return figure
```

## entlearn.plotting.plot_centroids

```
plot_centroids(
    centroids,
    X=None,
    *,
    theta=None,
    affiliations=None,
    wt=None,
    features=(0, 1),
    fig=None,
    row=None,
    col=None,
)
```

Plot the continuous centroids over two features, optionally with the data.

Without `theta` each centroid has its own colour. With `theta`, the transition matrix leaving this block, a centroid takes the colours of the clusters or classes it leads to, mixed in proportion. Observations with `affiliations` are coloured the same way through their clusters.

Parameters:

| Name           | Type              | Description                                                                             | Default                                                                      |
| -------------- | ----------------- | --------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| `centroids`    | `Any`             | The centroids, one row per cluster.                                                     | *required*                                                                   |
| `X`            | `Any`             | Observations to draw behind the centroids.                                              | `None`                                                                       |
| `theta`        | `Any`             | Transition matrix with one column per centroid and one row per target cluster or class. | `None`                                                                       |
| `affiliations` | `Any`             | The observations' affiliations, one column per centroid.                                | `None`                                                                       |
| `wt`           | `Any`             | Instance weights of the observations; larger weights draw larger markers.               | `None`                                                                       |
| `features`     | `tuple[int, int]` | The two feature axes.                                                                   | `(0, 1)`                                                                     |
| `fig`          | \`Figure          | None\`                                                                                  | A figure to draw into, such as one from make_subplots; by default a new one. |
| `row`          | \`int             | None\`                                                                                  | The subplot's row in fig.                                                    |
| `col`          | \`int             | None\`                                                                                  | The subplot's column in fig.                                                 |

Source code in `src/entlearn/plotting/basic.py`

```
def plot_centroids(
    centroids: Any,
    X: Any = None,
    *,
    theta: Any = None,
    affiliations: Any = None,
    wt: Any = None,
    features: tuple[int, int] = (0, 1),
    fig: go.Figure | None = None,
    row: int | None = None,
    col: int | None = None,
) -> go.Figure:
    """Plot the continuous centroids over two features, optionally with the data.

    Without ``theta`` each centroid has its own colour. With ``theta``, the transition
    matrix leaving this block, a centroid takes the colours of the clusters or classes
    it leads to, mixed in proportion. Observations with ``affiliations`` are coloured the
    same way through their clusters.

    Args:
        centroids: The centroids, one row per cluster.
        X: Observations to draw behind the centroids.
        theta: Transition matrix with one column per centroid and one row per target
            cluster or class.
        affiliations: The observations' affiliations, one column per centroid.
        wt: Instance weights of the observations; larger weights draw larger markers.
        features: The two feature axes.
        fig: A figure to draw into, such as one from ``make_subplots``; by default a
            new one.
        row: The subplot's row in ``fig``.
        col: The subplot's column in ``fig``.
    """
    C = _numpy(centroids)[:, list(features)]
    figure, at = _target(fig, row, col)
    leads_to = None if theta is None else _numpy(theta).T
    if leads_to is None:
        centroid_colours = _blend(np.eye(len(C)), CLUSTER_COLOURS)
    else:
        centroid_colours = _blend(leads_to, CLASS_COLOURS)
    if X is not None:
        points = _numpy(X)[:, list(features)]
        colour: Any = "lightgrey"
        if affiliations is not None:
            gamma = _numpy(affiliations)
            if leads_to is None:
                colour = _blend(gamma, CLUSTER_COLOURS)
            else:
                leads = leads_to / np.clip(leads_to.sum(axis=1, keepdims=True), _TINY, None)
                colour = _blend(gamma @ leads, CLASS_COLOURS)
        size: Any = 6
        if wt is not None:
            weights = _numpy(wt)
            size = 3 + 9 * np.sqrt(weights / max(weights.max(), _TINY))
        figure.add_trace(
            go.Scatter(
                x=points[:, 0],
                y=points[:, 1],
                mode="markers",
                name="observations",
                showlegend=False,
                marker=dict(color=colour, size=size, line=dict(width=0.5, color="black")),
            ),
            **at,
        )
    figure.add_trace(
        go.Scatter(
            x=C[:, 0],
            y=C[:, 1],
            mode="markers",
            name="centroids",
            showlegend=False,
            marker=dict(color=centroid_colours, **CENTROID_MARKER),
            text=[f"cluster {k}" for k in range(len(C))],
            hoverinfo="text",
        ),
        **at,
    )
    figure.update_xaxes(title_text=f"feature {features[0]}", **at)
    figure.update_yaxes(title_text=f"feature {features[1]}", **at)
    _equal_aspect(figure, row, col)
    return figure
```

## entlearn.plotting.plot_affiliations

```
plot_affiliations(
    X,
    affiliations,
    centroids=None,
    *,
    classes=False,
    features=(0, 1),
    fig=None,
    row=None,
    col=None,
)
```

Colour observations by their affiliations over two features.

Each cluster has a colour, and each observation mixes the colours of its clusters in proportion to its affiliations.

Parameters:

| Name           | Type              | Description                                                                                                                         | Default                                                                      |
| -------------- | ----------------- | ----------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| `X`            | `Any`             | The observations.                                                                                                                   | *required*                                                                   |
| `affiliations` | `Any`             | Their affiliations, one row per observation and one column per cluster, from any block.                                             | *required*                                                                   |
| `centroids`    | `Any`             | The centroids to mark, one row per cluster, which must live in the same space as X.                                                 | `None`                                                                       |
| `classes`      | `bool`            | Whether the columns are classes, such as the probabilities from network.predict; they then take the class colours of plot_decision. | `False`                                                                      |
| `features`     | `tuple[int, int]` | The two features on the axes.                                                                                                       | `(0, 1)`                                                                     |
| `fig`          | \`Figure          | None\`                                                                                                                              | A figure to draw into, such as one from make_subplots; by default a new one. |
| `row`          | \`int             | None\`                                                                                                                              | The subplot's row in fig.                                                    |
| `col`          | \`int             | None\`                                                                                                                              | The subplot's column in fig.                                                 |

Source code in `src/entlearn/plotting/basic.py`

```
def plot_affiliations(
    X: Any,
    affiliations: Any,
    centroids: Any = None,
    *,
    classes: bool = False,
    features: tuple[int, int] = (0, 1),
    fig: go.Figure | None = None,
    row: int | None = None,
    col: int | None = None,
) -> go.Figure:
    """Colour observations by their affiliations over two features.

    Each cluster has a colour, and each observation mixes the colours of its clusters in
    proportion to its affiliations.

    Args:
        X: The observations.
        affiliations: Their affiliations, one row per observation and one column per
            cluster, from any block.
        centroids: The centroids to mark, one row per cluster, which must live in the same
            space as ``X``.
        classes: Whether the columns are classes, such as the probabilities from
            ``network.predict``; they then take the class colours of ``plot_decision``.
        features: The two features on the axes.
        fig: A figure to draw into, such as one from ``make_subplots``; by default a
            new one.
        row: The subplot's row in ``fig``.
        col: The subplot's column in ``fig``.
    """
    points = _numpy(X)[:, list(features)]
    gamma = _numpy(affiliations)
    figure, at = _target(fig, row, col)
    palette, label = (CLASS_COLOURS, "class") if classes else (CLUSTER_COLOURS, "cluster")
    figure.add_trace(
        go.Scatter(
            x=points[:, 0],
            y=points[:, 1],
            mode="markers",
            name="observations",
            showlegend=False,
            marker=dict(color=_blend(gamma, palette), size=6),
            text=[f"{label} {k}" for k in gamma.argmax(axis=1)],
            hoverinfo="text",
        ),
        **at,
    )
    if centroids is not None:
        C = _numpy(centroids)[:, list(features)]
        figure.add_trace(
            go.Scatter(
                x=C[:, 0],
                y=C[:, 1],
                mode="markers",
                name="centroids",
                showlegend=False,
                marker=dict(color=_blend(np.eye(len(C)), CLUSTER_COLOURS), **CENTROID_MARKER),
            ),
            **at,
        )
    figure.update_xaxes(title_text=f"feature {features[0]}", **at)
    figure.update_yaxes(title_text=f"feature {features[1]}", **at)
    _equal_aspect(figure, row, col)
    return figure
```

## entlearn.plotting.plot_theta

```
plot_theta(
    theta,
    *,
    x_title="source cluster",
    y_title="target cluster",
    fig=None,
    row=None,
    col=None,
)
```

Plot a transition matrix as a heatmap, one column per source cluster, with a colour bar.

Parameters:

| Name      | Type     | Description                                                                                                                                   | Default                                                                      |
| --------- | -------- | --------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| `theta`   | `Any`    | The matrix, one row per target cluster (or class, or output) and one column per source cluster, as in network.inspect("transition_matrices"). | *required*                                                                   |
| `x_title` | `str`    | The title of the horizontal axis.                                                                                                             | `'source cluster'`                                                           |
| `y_title` | `str`    | The title of the vertical axis.                                                                                                               | `'target cluster'`                                                           |
| `fig`     | \`Figure | None\`                                                                                                                                        | A figure to draw into, such as one from make_subplots; by default a new one. |
| `row`     | \`int    | None\`                                                                                                                                        | The subplot's row in fig.                                                    |
| `col`     | \`int    | None\`                                                                                                                                        | The subplot's column in fig.                                                 |

Source code in `src/entlearn/plotting/basic.py`

```
def plot_theta(
    theta: Any,
    *,
    x_title: str = "source cluster",
    y_title: str = "target cluster",
    fig: go.Figure | None = None,
    row: int | None = None,
    col: int | None = None,
) -> go.Figure:
    """Plot a transition matrix as a heatmap, one column per source cluster, with a colour bar.

    Args:
        theta: The matrix, one row per target cluster (or class, or output) and one column
            per source cluster, as in ``network.inspect("transition_matrices")``.
        x_title: The title of the horizontal axis.
        y_title: The title of the vertical axis.
        fig: A figure to draw into, such as one from ``make_subplots``; by default a
            new one.
        row: The subplot's row in ``fig``.
        col: The subplot's column in ``fig``.
    """
    values = _numpy(theta)
    figure, at = _target(fig, row, col)
    colorbar: dict[str, Any] = dict(thickness=8, len=0.8)
    if row is not None:
        # In a subplot, the colour bar sits beside its own panel instead of the figure's edge.
        axes = figure.get_subplot(row, col)
        (_, right), (bottom, top) = axes.xaxis.domain, axes.yaxis.domain
        colorbar.update(
            x=right + 0.01, xanchor="left", y=(bottom + top) / 2, len=0.8 * (top - bottom)
        )
    figure.add_trace(
        go.Heatmap(
            z=values,
            colorscale="Blues",
            colorbar=colorbar,
            showlegend=False,
            hovertemplate=f"{x_title} %{{x}}<br>{y_title} %{{y}}<br>%{{z:.3f}}<extra></extra>",
        ),
        **at,
    )
    figure.update_xaxes(title_text=x_title, dtick=1, **at)
    figure.update_yaxes(title_text=y_title, dtick=1, autorange="reversed", **at)
    return figure
```

## entlearn.plotting.plot_decision

```
plot_decision(
    network,
    X,
    y=None,
    *,
    X_cat=None,
    features=(0, 1),
    resolution=40,
)
```

Plot which class a classifier predicts across two features, and how confidently.

Each class has a colour, and the map mixes them in proportion to the predicted probabilities. The other continuous features are held at their medians over `X`, and each categorical feature at its most frequent category. When the network learned instance weights, a second panel shows `network.score_samples` over the same grid.

Parameters:

| Name         | Type               | Description                                                                                      | Default                                                                                                               |
| ------------ | ------------------ | ------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------- |
| `network`    | `Network`          | A fitted classification network.                                                                 | *required*                                                                                                            |
| `X`          | `Tensor`           | Observations on the network's device and dtype; they set the plotted range and are drawn on top. | *required*                                                                                                            |
| `y`          | `Any`              | Their class labels, which colour the observations; by default the predicted classes.             | `None`                                                                                                                |
| `X_cat`      | \`Sequence[Tensor] | None\`                                                                                           | Their categorical features, one tensor of category codes per feature, for a network fitted with categorical features. |
| `features`   | `tuple[int, int]`  | The two features on the axes.                                                                    | `(0, 1)`                                                                                                              |
| `resolution` | `int`              | Grid points per axis. The browser smooths the map between them.                                  | `40`                                                                                                                  |

Raises:

| Type         | Description                         |
| ------------ | ----------------------------------- |
| `ValueError` | If the network is not a classifier. |

Source code in `src/entlearn/plotting/decision.py`

```
def plot_decision(
    network: Network,
    X: torch.Tensor,
    y: Any = None,
    *,
    X_cat: Sequence[torch.Tensor] | None = None,
    features: tuple[int, int] = (0, 1),
    resolution: int = 40,
) -> go.Figure:
    """Plot which class a classifier predicts across two features, and how confidently.

    Each class has a colour, and the map mixes them in proportion to the predicted
    probabilities.
    The other continuous features
    are held at their medians over ``X``, and each categorical feature at its most
    frequent category. When the network learned instance weights, a second
    panel shows ``network.score_samples`` over the same grid.

    Args:
        network: A fitted classification network.
        X: Observations on the network's device and dtype; they set the plotted range and
            are drawn on top.
        y: Their class labels, which colour the observations; by default the predicted
            classes.
        X_cat: Their categorical features, one tensor of category codes per feature, for a
            network fitted with categorical features.
        features: The two features on the axes.
        resolution: Grid points per axis. The browser smooths the map between them.

    Raises:
        ValueError: If the network is not a classifier.
    """
    if network.schema.task != "classification":
        raise ValueError("plot_decision needs a classification network")
    first, second = features
    low, high = X[:, [first, second]].amin(0), X[:, [first, second]].amax(0)
    low, high = low - 0.1 * (high - low), high + 0.1 * (high - low)
    along = [
        torch.linspace(float(low[i]), float(high[i]), resolution, dtype=X.dtype) for i in (0, 1)
    ]
    gx, gy = torch.meshgrid(*along, indexing="xy")
    grid = X.median(dim=0).values.repeat(gx.numel(), 1)
    grid[:, first], grid[:, second] = gx.flatten().to(X.device), gy.flatten().to(X.device)
    grid_cat = None
    if X_cat is not None:
        grid_cat = [codes.mode().values.repeat(len(grid)) for codes in X_cat]

    probabilities = _numpy(network.predict(grid, X_cat=grid_cat))
    classes = probabilities.shape[1]
    rgb = (probabilities @ _palette(CLASS_COLOURS, classes)).reshape(resolution, resolution, 3)
    xs, ys = _numpy(along[0]), _numpy(along[1])
    scored = _scores_available(network)
    fig = make_subplots(
        rows=1,
        cols=2 if scored else 1,
        horizontal_spacing=0.12,
        subplot_titles=("Predicted class", "Confidence") if scored else ("Predicted class",),
    )
    fig.add_trace(
        go.Image(
            source=_png(rgb),
            zsmooth="fast",
            x0=xs[0],
            dx=xs[1] - xs[0],
            y0=ys[0],
            dy=ys[1] - ys[0],
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Contour(
            z=probabilities.argmax(axis=1).reshape(resolution, resolution),
            x=xs,
            y=ys,
            contours=dict(coloring="none", start=0.5, end=classes - 1.5, size=1),
            line=dict(color="black", width=1),
            showscale=False,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    if scored:
        fig.add_trace(
            go.Contour(
                z=_numpy(network.score_samples(grid, X_cat=grid_cat)).reshape(
                    resolution, resolution
                ),
                x=xs,
                y=ys,
                colorscale="YlOrRd",
                zmin=0,
                zmax=1,
                hoverinfo="skip",
                colorbar=dict(title="inlier", len=0.6, thickness=8),
            ),
            row=1,
            col=2,
        )

    points = _numpy(X)[:, [first, second]]
    if y is None:
        labels = _numpy(network.predict(X, X_cat=X_cat)).argmax(axis=1)
    else:
        labels = _numpy(y).astype(int)
    size: Any = 7 if not scored else 4 + 12 * _numpy(network.score_samples(X, X_cat=X_cat))
    for col in (1, 2) if scored else (1,):
        fig.add_trace(
            go.Scatter(
                x=points[:, 0],
                y=points[:, 1],
                mode="markers",
                showlegend=False,
                marker=dict(
                    color=_blend(np.eye(classes)[labels], CLASS_COLOURS),
                    size=size,
                    line=dict(width=1, color="black"),
                ),
                text=[f"class {c}" for c in labels],
                hoverinfo="text",
            ),
            row=1,
            col=col,
        )
        fig.update_xaxes(title_text=f"feature {first}", range=[xs[0], xs[-1]], row=1, col=col)
        fig.update_yaxes(title_text=f"feature {second}", range=[ys[0], ys[-1]], row=1, col=col)
        _equal_aspect(fig, 1, col)
    return fig
```

## entlearn.plotting.plot_manifold

```
plot_manifold(
    network,
    X,
    y=None,
    *,
    probes=None,
    features=(0, 1, 2),
    output=0,
    block="input",
    volume_resolution=30,
    fig=None,
    row=None,
    col=None,
)
```

Plot a manifold input's local manifolds, what they carry, and where new points land.

The figure is a 3-D scene over three features, with layers you switch on and off from the legend:

- **Manifolds**: each cluster's local plane, or line when `subspace_dimension` is 1. With instance weights, a manifold also stops where `network.score_samples` falls below the lowest score among `X`, and it fades as the score drops.
- **Manifolds (prediction)** (hidden at first): the same manifolds, but coloured by what the network predicts at each point on them. A regression network uses its output `output` on the colour scale of `y`, a classifier mixes the class colours of `plot_decision` in proportion to the predicted probabilities.
- **Observations**: the observations `X`, coloured by `y`. When `X` is the training data, they are drawn larger the larger their fitted instance weight.
- **Probes**: with `probes`, each probe's end, where the network reconstructs it, sized and coloured by its score; hidden layers show each probe's origin and its movement from origin to end.
- **Inlier fade** (hidden by default, with instance weights): the score over the whole box, drawn as a cloud.

Parameters:

| Name                | Type                   | Description                                                                                                                                | Default                                                                      |
| ------------------- | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------- |
| `network`           | `Network`              | A fitted network whose input block is a ManifoldInput.                                                                                     | *required*                                                                   |
| `X`                 | `Tensor`               | The observations to draw, usually the training data, on the network's device and dtype.                                                    | *required*                                                                   |
| `y`                 | `Any`                  | The values that colour them, such as their targets or predictions.                                                                         | `None`                                                                       |
| `probes`            | \`Tensor               | None\`                                                                                                                                     | Further points to send through the network.                                  |
| `features`          | `tuple[int, int, int]` | The three continuous feature axes.                                                                                                         | `(0, 1, 2)`                                                                  |
| `output`            | `int`                  | For a regression network, the output that colours the prediction manifolds, without y, its predictions over X also set their colour scale. | `0`                                                                          |
| `block`             | `str`                  | The name of the manifold input block.                                                                                                      | `'input'`                                                                    |
| `volume_resolution` | `int`                  | Grid points per axis for the inlier fade.                                                                                                  | `30`                                                                         |
| `fig`               | \`Figure               | None\`                                                                                                                                     | A figure to draw into, such as one from make_subplots; by default a new one. |
| `row`               | \`int                  | None\`                                                                                                                                     | The 3-D subplot's row in fig.                                                |
| `col`               | \`int                  | None\`                                                                                                                                     | The 3-D subplot's column in fig.                                             |

Raises:

| Type         | Description                                                                                                                                                                     |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ValueError` | If the input is not a manifold input, its subspace_dimension is above 2, features does not name three different continuous features, or output is not an output of the network. |

Source code in `src/entlearn/plotting/manifold.py`

```
def plot_manifold(
    network: Network,
    X: torch.Tensor,
    y: Any = None,
    *,
    probes: torch.Tensor | None = None,
    features: tuple[int, int, int] = (0, 1, 2),
    output: int = 0,
    block: str = "input",
    volume_resolution: int = 30,
    fig: go.Figure | None = None,
    row: int | None = None,
    col: int | None = None,
) -> go.Figure:
    """Plot a manifold input's local manifolds, what they carry, and where new points land.

    The figure is a 3-D scene over three features, with layers you switch on and off
    from the legend:

    - **Manifolds**: each cluster's local plane, or line when ``subspace_dimension`` is 1.
      With instance weights, a manifold also stops where ``network.score_samples`` falls
      below the lowest score among ``X``, and it fades as the score drops.
    - **Manifolds (prediction)** (hidden at first): the same manifolds, but coloured by
      what the network predicts at each point on them. A regression network uses its
      output ``output`` on the colour scale of ``y``, a classifier mixes the class colours of
      ``plot_decision`` in proportion to the predicted probabilities.
    - **Observations**: the observations ``X``, coloured by ``y``. When ``X`` is the
      training data, they are drawn larger the larger their fitted instance weight.
    - **Probes**: with ``probes``, each probe's end, where the network reconstructs it,
      sized and coloured by its score; hidden layers show each probe's origin and its
      movement from origin to end.
    - **Inlier fade** (hidden by default, with instance weights): the score over the
      whole box, drawn as a cloud.

    Args:
        network: A fitted network whose input block is a ``ManifoldInput``.
        X: The observations to draw, usually the training data, on the network's device
            and dtype.
        y: The values that colour them, such as their targets or predictions.
        probes: Further points to send through the network.
        features: The three continuous feature axes.
        output: For a regression network, the output that colours the prediction manifolds,
            without ``y``, its predictions over ``X`` also set their colour scale.
        block: The name of the manifold input block.
        volume_resolution: Grid points per axis for the inlier fade.
        fig: A figure to draw into, such as one from ``make_subplots``; by default a
            new one.
        row: The 3-D subplot's row in ``fig``.
        col: The 3-D subplot's column in ``fig``.

    Raises:
        ValueError: If the input is not a manifold input, its ``subspace_dimension`` is
            above 2, ``features`` does not name three different continuous features, or
            ``output`` is not an output of the network.
    """
    C = network.inspect("continuous_centroids")[block]
    bases = network.inspect("manifold_projectors")[block]
    axes = list(features)
    if len(axes) != 3 or len(set(axes)) != 3 or not all(0 <= i < C.shape[1] for i in axes):
        raise ValueError(
            f"features must name three different continuous features, from 0 to {C.shape[1] - 1}"
        )
    if bases.shape[2] > 2:
        raise ValueError(
            f"plot_manifold draws lines and planes only; block {block!r} has "
            f"subspace_dimension={bases.shape[2]}, which cannot be drawn in three dimensions"
        )
    figure, at = _target(fig, row, col)
    scored = _scores_available(network)
    floor = float(network.score_samples(X).min()) if scored else 0.0
    low, high = X[:, axes].amin(0), X[:, axes].amax(0)
    box = torch.stack([low - 0.25 * (high - low), high + 0.25 * (high - low)])
    classification = network.schema.task == "classification"
    if classification:
        classes = network.predict(X[:1]).shape[1]
        shade = {}
    else:
        outputs = network.predict(X[:1]).shape[1]
        if not 0 <= output < outputs:
            raise ValueError(f"output must be from 0 to {outputs - 1}")
        values = _numpy(network.predict(X)[:, output] if y is None else y).reshape(-1)
        shade = dict(cmin=float(values.min()), cmax=float(values.max()))

    # Manifolds: rays from each centroid along its plane, each running out to where the plane
    # stops being feasible (another cluster wins, or the score falls below the floor).
    d = bases.shape[2]
    reach = float(torch.linalg.vector_norm(box[1] - box[0]))
    if d == 2:
        angle = torch.linspace(0, 2 * torch.pi, _DIRECTIONS, dtype=X.dtype, device=X.device)
        rays = torch.stack([angle.cos(), angle.sin()], dim=1)
    else:
        rays = torch.tensor([[1.0], [-1.0]], dtype=X.dtype, device=X.device)
    fraction = torch.linspace(0, 1, _STEPS, dtype=X.dtype, device=X.device)

    def last_feasible(k: int, on_plane: torch.Tensor, start: torch.Tensor, stop: torch.Tensor):
        """Per ray, the farthest sampled distance in [start, stop] before the plane ends."""
        t = start[:, None] + (stop - start)[:, None] * fraction  # (R, steps)
        points = C[k] + (t[..., None] * on_plane[:, None, :]).reshape(-1, C.shape[1])
        gamma = network.predict_with_details(points, details=("affiliations",)).affiliations
        assert gamma is not None
        feasible = gamma[block].argmax(dim=1) == k
        feasible &= ((points[:, axes] >= box[0]) & (points[:, axes] <= box[1])).all(dim=1)
        if scored:
            feasible &= network.score_samples(points) >= floor
        feasible = feasible.reshape(t.shape)
        first_out = torch.where(feasible.all(dim=1), _STEPS, (~feasible).int().argmax(dim=1))
        return t.gather(1, (first_out - 1).clamp(min=0)[:, None]).squeeze(1)

    step = reach / (_STEPS - 1)
    for k in range(len(C)):
        on_plane = rays @ bases[k][:, :d].T  # (R, D) unit directions in the plane
        coarse = last_feasible(
            k, on_plane, torch.zeros_like(rays[:, 0]), torch.full_like(rays[:, 0], reach)
        )
        radius = last_feasible(k, on_plane, coarse, coarse + step)
        rings = torch.linspace(0, 1, _RINGS, dtype=X.dtype, device=X.device)
        surface = C[k] + rings[:, None, None] * (radius[:, None] * on_plane)[None]  # (rings, R, D)
        score = network.score_samples(surface.reshape(-1, C.shape[1])) if scored else None
        xyz = _numpy(surface[..., axes])
        colour = qualitative.Dark24[k % len(qualitative.Dark24)]
        legend = dict(
            name="Manifolds", legendgroup="manifolds", showlegend=k == 0, hoverinfo="name"
        )
        predicted = dict(
            name="Manifolds (prediction)",
            legendgroup="manifolds_prediction",
            showlegend=k == 0,
            visible="legendonly",
            hoverinfo="name",
        )
        prediction = network.predict(surface.reshape(-1, C.shape[1]))
        if classification:
            tone = _blend(_numpy(prediction), CLASS_COLOURS)
            mesh_tone: dict[str, Any] = dict(vertexcolor=tone)
            line_tone: dict[str, Any] = dict()
        else:
            tone = _numpy(prediction[:, output])
            mesh_tone = dict(intensity=tone, colorscale="Viridis", showscale=False, **shade)
            line_tone = dict(colorscale="Viridis", **shade)
        vertices = xyz.reshape(-1, 3)
        if d == 2:
            figure.add_trace(
                go.Surface(
                    x=xyz[..., 0],
                    y=xyz[..., 1],
                    z=xyz[..., 2],
                    surfacecolor=None if score is None else _numpy(score).reshape(_RINGS, -1),
                    cmin=0,
                    cmax=1,
                    colorscale=[[0, colour], [1, colour]],
                    showscale=False,
                    opacity=None if scored else 0.6,
                    opacityscale=[[0, 0.1], [0.5, 0.8], [1, 0.8]] if scored else None,
                    **legend,
                ),
                **at,
            )
            first, second, third = _grid_faces(*xyz.shape[:2])
            figure.add_trace(
                go.Mesh3d(
                    x=vertices[:, 0],
                    y=vertices[:, 1],
                    z=vertices[:, 2],
                    i=first,
                    j=second,
                    k=third,
                    opacity=0.8,
                    **mesh_tone,
                    **predicted,
                ),
                **at,
            )
        else:
            ends = xyz[-1]
            # One path from the far end of ray -1, through the centroid, to the end of ray +1.
            path = np.concatenate([np.arange(_RINGS)[::-1] * 2 + 1, np.arange(1, _RINGS) * 2])
            figure.add_trace(
                go.Scatter3d(
                    x=ends[:, 0],
                    y=ends[:, 1],
                    z=ends[:, 2],
                    mode="lines",
                    line=dict(color=colour, width=6),
                    **legend,
                ),
                **at,
            )
            figure.add_trace(
                go.Scatter3d(
                    x=vertices[path, 0],
                    y=vertices[path, 1],
                    z=vertices[path, 2],
                    mode="lines",
                    line=dict(width=8, color=np.asarray(tone)[path].tolist(), **line_tone),
                    **predicted,
                ),
                **at,
            )
    centres = _numpy(C[:, axes])
    figure.add_trace(
        go.Scatter3d(
            x=centres[:, 0],
            y=centres[:, 1],
            z=centres[:, 2],
            mode="markers",
            marker=dict(size=4, color="black", symbol="diamond"),
            name="Manifolds",
            legendgroup="manifolds",
            showlegend=False,
            text=[f"cluster {k}" for k in range(len(C))],
            hoverinfo="text",
        ),
        **at,
    )

    # Observations: coloured by target, larger with larger instance weight.
    rows = _numpy(X[:, axes])
    size: Any = 4
    try:
        weights = _numpy(network.inspect("training_instance_weights")[block])
    except ValueError:
        weights = None
    if weights is not None and len(weights) == len(rows):
        size = 2 + 8 * np.sqrt(weights / max(weights.max(), np.finfo(float).tiny))
    observed: dict[str, Any] = dict(size=size)
    if y is not None and classification:
        labels = _numpy(y).astype(int).reshape(-1)
        observed.update(color=_blend(np.eye(classes)[labels], CLASS_COLOURS))
    elif y is not None:
        colorbar = dict(title="target", len=0.45, y=0.75, thickness=10)
        observed.update(color=values, colorscale="Viridis", colorbar=colorbar, **shade)
    figure.add_trace(
        go.Scatter3d(
            x=rows[:, 0],
            y=rows[:, 1],
            z=rows[:, 2],
            mode="markers",
            name="Observations",
            marker=observed,
        ),
        **at,
    )

    # Probes: where each starts, and where the network reconstructs it.
    if probes is not None:
        landing = network.predict_with_details(probes, details=("reconstruction",)).reconstruction
        assert landing is not None
        start, end = _numpy(probes[:, axes]), _numpy(landing.continuous[:, axes])
        score = _numpy(network.score_samples(probes)) if scored else np.full(len(start), 0.5)
        segments = np.stack([start, end, np.full_like(start, np.nan)], axis=1).reshape(-1, 3)
        figure.add_trace(
            go.Scatter3d(
                x=start[:, 0],
                y=start[:, 1],
                z=start[:, 2],
                mode="markers",
                name="Probes (origin)",
                visible="legendonly",
                marker=dict(size=2, color="grey", opacity=0.4),
            ),
            **at,
        )
        figure.add_trace(
            go.Scatter3d(
                x=segments[:, 0],
                y=segments[:, 1],
                z=segments[:, 2],
                mode="lines",
                name="Probes (movement)",
                visible="legendonly",
                hoverinfo="skip",
                line=dict(color="rgba(120,120,120,0.25)", width=1),
            ),
            **at,
        )
        figure.add_trace(
            go.Scatter3d(
                x=end[:, 0],
                y=end[:, 1],
                z=end[:, 2],
                mode="markers",
                name="Probes (end)",
                marker=dict(
                    size=1 + 9 * score,
                    color=score,
                    colorscale="YlOrRd",
                    cmin=0,
                    cmax=1,
                    colorbar=dict(title="inlier", len=0.45, y=0.25, thickness=10),
                    showscale=scored,
                ),
                text=[f"inlier {s:.2f}" for s in score],
                hoverinfo="text",
            ),
            **at,
        )

    # Inlier fade: the score over a grid of the whole box.
    if scored:
        cube = _on_grid(X, box, volume_resolution, axes)
        figure.add_trace(
            go.Volume(
                x=_numpy(cube[:, 0]).astype(np.float32),
                y=_numpy(cube[:, 1]).astype(np.float32),
                z=_numpy(cube[:, 2]).astype(np.float32),
                value=_numpy(network.score_samples(cube)).astype(np.float32),
                visible="legendonly",
                showlegend=True,
                showscale=False,
                hoverinfo="skip",
                name="Inlier fade",
                isomin=0.01,
                isomax=1.0,
                surface_count=12,
                colorscale=[[0, "#9ecae1"], [1, "#08306b"]],
                opacity=0.4,
                opacityscale=[[0, 0], [0.05, 0.3], [0.5, 0.8], [1, 1]],
            ),
            **at,
        )

    low_edge, high_edge = _numpy(box[0]), _numpy(box[1])
    figure.update_scenes(
        aspectmode="data",
        dragmode="turntable",
        **{
            f"{axis}axis": dict(title=f"feature {axes[i]}", range=[low_edge[i], high_edge[i]])
            for i, axis in enumerate("xyz")
        },
        **at,
    )
    figure.update_layout(legend=dict(x=0, y=1), margin=dict(l=0, r=0, t=40, b=0))
    return figure
```

## entlearn.plotting.plot_parallel

```
plot_parallel(
    network,
    X,
    y=None,
    *,
    X_cat=None,
    scaled=False,
    show_data=True,
    colour="cluster",
    names=None,
    categorical_names=None,
    block="input",
)
```

Draw every input centroid across all features, with the data behind it.

Each centroid is one line through one axis per continuous feature, one 0-to-1 axis per level of each categorical feature (grouped under a band), and then the head: the probability of each class, or each output. With `y`, a last axis shows each observation's label or target, and for a centroid the class or output it leads to. The observations are drawn behind as transparent lines, and the feature weights are written under their axes. Drag along an axis to keep only the lines in that range.

Parameters:

| Name                | Type                          | Description                                                                                                                                                                                                      | Default                                                                     |
| ------------------- | ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| `network`           | `Network`                     | A fitted network with a standard or manifold input.                                                                                                                                                              | *required*                                                                  |
| `X`                 | `Tensor`                      | The observations' continuous features, on the network's device and dtype.                                                                                                                                        | *required*                                                                  |
| `y`                 | `Any`                         | Their labels or targets, drawn on a last axis.                                                                                                                                                                   | `None`                                                                      |
| `X_cat`             | \`Sequence[Tensor]            | None\`                                                                                                                                                                                                           | Their categorical features, for a network fitted with categorical features. |
| `scaled`            | `bool`                        | Multiply each feature axis by the square root of its weight and give all of them one range, so their spreads follow the distances the network uses.                                                              | `False`                                                                     |
| `show_data`         | `bool`                        | Draw the observations. Without them the axes keep the same ranges.                                                                                                                                               | `True`                                                                      |
| `colour`            | `Literal['cluster', 'class']` | Colour the lines by "cluster": an observation by its most likely cluster; or, for a classifier, by "class": an observation by its label (its predicted class without y) and a centroid by the class it leads to. | `'cluster'`                                                                 |
| `names`             | \`Sequence[str]               | None\`                                                                                                                                                                                                           | Names of the continuous features; by default x0, x1, ...                    |
| `categorical_names` | \`Sequence[str]               | None\`                                                                                                                                                                                                           | Names of the categorical features; by default c0, c1, ...                   |
| `block`             | `str`                         | The name of the input block.                                                                                                                                                                                     | `'input'`                                                                   |

Raises:

| Type         | Description                                                                                                              |
| ------------ | ------------------------------------------------------------------------------------------------------------------------ |
| `ValueError` | If colour is not "cluster" or "class", "class" is asked of a regression network, or the names do not match the features. |

Source code in `src/entlearn/plotting/parallel.py`

```
def plot_parallel(
    network: Network,
    X: torch.Tensor,
    y: Any = None,
    *,
    X_cat: Sequence[torch.Tensor] | None = None,
    scaled: bool = False,
    show_data: bool = True,
    colour: Literal["cluster", "class"] = "cluster",
    names: Sequence[str] | None = None,
    categorical_names: Sequence[str] | None = None,
    block: str = "input",
) -> go.Figure:
    """Draw every input centroid across all features, with the data behind it.

    Each centroid is one line through one axis per continuous feature, one 0-to-1 axis per
    level of each categorical feature (grouped under a band), and then the head: the
    probability of each class, or each output. With ``y``, a last axis shows each
    observation's label or target, and for a centroid the class or output it leads to.
    The observations are drawn behind as transparent lines, and the feature weights are
    written under their axes. Drag along an axis to keep only the lines in that range.

    Args:
        network: A fitted network with a standard or manifold input.
        X: The observations' continuous features, on the network's device and dtype.
        y: Their labels or targets, drawn on a last axis.
        X_cat: Their categorical features, for a network fitted with categorical features.
        scaled: Multiply each feature axis by the square root of its weight and give all of
            them one range, so their spreads follow the distances the network uses.
        show_data: Draw the observations. Without them the axes keep the same ranges.
        colour: Colour the lines by ``"cluster"``: an observation by its most likely
            cluster; or, for a classifier, by ``"class"``: an observation by its label
            (its predicted class without ``y``) and a centroid by the class it leads to.
        names: Names of the continuous features; by default ``x0``, ``x1``, ...
        categorical_names: Names of the categorical features; by default ``c0``, ``c1``, ...
        block: The name of the input block.

    Raises:
        ValueError: If ``colour`` is not ``"cluster"`` or ``"class"``, ``"class"`` is
            asked of a regression network, or the names do not match the features.
    """
    if colour not in ("cluster", "class"):
        raise ValueError('colour must be "cluster" or "class"')
    X_cat = list(X_cat or [])
    C = network.inspect("continuous_centroids")[block]
    C_cats = network.inspect("categorical_centroids")[block] if network.schema.M_cat else ()
    K, T, D = C.shape[0], X.shape[0], C.shape[1]
    names = list(names) if names is not None else [f"x{d}" for d in range(D)]
    categorical_names = (
        list(categorical_names)
        if categorical_names is not None
        else [f"c{j}" for j in range(len(C_cats))]
    )
    if len(names) != D or len(categorical_names) != len(C_cats):
        raise ValueError(
            f"expected {D} continuous and {len(C_cats)} categorical feature names, "
            f"got {len(names)} and {len(categorical_names)}"
        )
    lead, classification = _leads_to(network, block)
    if colour == "class" and not classification:
        raise ValueError('colour="class" needs a classification network')
    result = network.predict_with_details(X, X_cat=X_cat or None, details=("affiliations",))
    assert result.affiliations is not None
    cluster = _numpy(result.affiliations[block]).argmax(axis=1)
    prediction = _numpy(result.prediction)
    weights = _numpy(feature_importances(network))

    # Each axis: its values for the T observations then the K centroids, and its weight.
    columns: dict[str, np.ndarray] = {"cluster": np.concatenate([cluster, np.arange(K)])}
    weight_of: dict[str, float] = {}
    levels: dict[str, str] = {}
    for d, name in enumerate(names):
        columns[name] = np.concatenate([_numpy(X[:, d]), _numpy(C[:, d])])
        weight_of[name] = float(weights[d])
    groups = []
    for j, (name, codes, C_cat) in enumerate(zip(categorical_names, X_cat, C_cats, strict=True)):
        one_hot = np.eye(C_cat.shape[1])[_numpy(codes).astype(int)]
        keys = []
        for level in range(C_cat.shape[1]):
            key = f"{name}={level}"
            columns[key] = np.concatenate([one_hot[:, level], _numpy(C_cat[:, level])])
            weight_of[key], levels[key] = float(weights[D + j]), str(level)
            keys.append(key)
        groups.append((name, float(weights[D + j]), keys))
    head = _numpy(lead).T  # (K, classes or outputs)
    shown = (
        range(1, head.shape[1]) if classification and head.shape[1] == 2 else range(head.shape[1])
    )
    for m in shown:
        key = f"P(class {m})" if classification else f"output {m}"
        columns[key] = np.concatenate([prediction[:, m], head[:, m]])
    if y is not None:
        labels = _numpy(y).reshape(T, -1)
        for m in range(labels.shape[1]):
            key = "label" if classification else f"target {m}"
            centroid = head.argmax(axis=1) if classification else head[:, m]
            columns[key] = np.concatenate([labels[:, m], centroid])

    # Ranges come from observations and centroids together, so hiding the data keeps them.
    keep = slice(None) if show_data else slice(T, None)
    scale = {key: np.sqrt(w) if scaled else 1.0 for key, w in weight_of.items()}
    shared = np.concatenate([columns[key] * scale[key] for key in weight_of])
    dimensions = []
    for key, values in columns.items():
        label = levels.get(key, key)
        dimension: dict[str, Any] = dict(label=label)
        if key in weight_of:
            values = values * scale[key]
            if scaled:
                dimension.update(label=f"√w·{label}", range=[shared.min(), shared.max()])
            elif key in levels:
                dimension.update(range=[0, 1])
            else:
                dimension.update(range=[values.min(), values.max()])
            dimension.update(tickformat=".2f")
        elif key == "cluster":
            dimension.update(tickvals=list(range(K)), range=[-0.5, K - 0.5])
        elif key.startswith("P("):
            dimension.update(range=[0, 1])
        elif key == "label":
            dimension.update(tickvals=list(range(head.shape[1])), range=[-0.1, head.shape[1] - 0.9])
        dimension.update(values=values[keep])
        dimensions.append(dimension)

    # One colour scale: the observations' colours made transparent, then the centroids'.
    if colour == "cluster":
        palette = _palette(CLUSTER_COLOURS, K)
        data_index, centroid_index = cluster, np.arange(K)
    else:
        palette = _palette(CLASS_COLOURS, head.shape[1])
        data_index = (
            _numpy(y).reshape(-1).astype(int) if y is not None else prediction.argmax(axis=1)
        )
        centroid_index = head.argmax(axis=1)
    n = len(palette)
    stops = [f"rgba({r:.0f},{g:.0f},{b:.0f},{_DATA_ALPHA})" for r, g, b in palette]
    stops += [f"rgb({r:.0f},{g:.0f},{b:.0f})" for r, g, b in palette]
    line_colour = np.concatenate([data_index, centroid_index + n])[keep]
    fig = go.Figure(
        go.Parcoords(
            line=dict(
                color=line_colour,
                colorscale=[[i / (2 * n - 1), c] for i, c in enumerate(stops)],
                cmin=0,
                cmax=2 * n - 1,
            ),
            dimensions=dimensions,
        )
    )

    # The weights under the feature axes; a band and a name under each categorical feature.
    order = list(columns)
    step = 1 / (len(order) - 1)
    note = dict(xref="paper", yref="paper", showarrow=False, font=dict(size=11))
    for name in names:
        fig.add_annotation(
            x=order.index(name) * step, y=-0.09, text=f"{weight_of[name]:.2f}", **note
        )
    for name, weight, keys in groups:
        first, last = order.index(keys[0]) * step, order.index(keys[-1]) * step
        fig.add_shape(
            type="rect",
            xref="paper",
            yref="paper",
            x0=first - 0.4 * step,
            x1=last + 0.4 * step,
            y0=-0.03,
            y1=1.0,
            fillcolor="rgba(120,120,120,0.10)",
            line_width=0,
            layer="below",
        )
        fig.add_annotation(x=(first + last) / 2, y=-0.09, text=f"{weight:.2f}", **note)
        fig.add_annotation(
            x=(first + last) / 2, y=-0.15, text=f"<b>{name}</b>{' ≈' if scaled else ''}", **note
        )
    fig.add_annotation(
        x=-0.01,
        xanchor="right",
        y=-0.09,
        text="weight",
        **{**note, "font": dict(size=11, color="grey")},
    )
    fig.update_layout(height=600, margin=dict(l=70, r=70, t=60, b=90))
    return fig
```
