# Plotting

Plots can help you inspect a fitted model and follow its predictions. Install the `plotting` extra to use them:

```
pip install "entlearn[plotting]"
```

Every plot is a Plotly figure, which displays directly in a notebook and can be saved with `figure.write_html("figure.html")`.

## Plot a fitted network

Plot a block using a fitted `network` and a continuous input tensor `X`:

```
input_figure = network.plot(block="input", X_cont=X)
```

Each block has a different plot to illustrate its parameters. Omit `block` to show the loss history followed by every block, and omit `X_cont` to show only the fitted parameters. For a network fitted with categorical features, pass them as `X_cat`, as for `network.predict`.

## Plot one part of a network

The panels of `network.plot` are also available on their own. Each takes fitted tensors, which you can read with `network.inspect`:

```
from entlearn.helpers.reporting import feature_importances
from entlearn.plotting import plot_centroids, plot_feature_importance, plot_theta

centroids = network.inspect("continuous_centroids")["input"]
affiliations = network.inspect("training_affiliations")["input"]
theta = network.inspect("transition_matrices")["input_to_hidden_1"]

plot_feature_importance(feature_importances(network))
plot_centroids(centroids, X, theta=theta, affiliations=affiliations)
plot_theta(theta)
```

Pass `wt`, the training instance weights, to draw more relevant observations larger. `plot_affiliations` colours observations by the affiliations of any block, or with `classes=True` by class probabilities in the colours of `plot_decision`, and `plot_loss` draws `network.diagnostics.loss_history`.

To place several of these in one figure, create it with Plotly's `make_subplots` and pass `fig`, `row` and `col`.

## Decision and confidence

For a classifier with at least two continuous features, plot predictions over a two-feature grid:

```
from entlearn.plotting import plot_decision

decision = plot_decision(network, X, y)
```

Each class has a colour, and the map mixes them in proportion to the predicted probabilities, so a confident region shows its class colour and an uncertain one a blend. For the two dimensional grid to work, the other continuous features are held at their medians. When the network learned instance weights, a second panel shows its [inlier percentiles](https://entropiclearning.github.io/entlearn/0.1.0/guide/networks/#reconstruction-and-inlier-percentiles) over the same grid.

## Every centroid in every feature

Use parallel coordinates to show every centroid across all features:

```
from entlearn.plotting import plot_parallel

overview = plot_parallel(network, X, y, X_cat=X_cat)
```

Each line represents a centroid. Continuous features have one axis each. Categorical features have one axis from 0 to 1 per category, grouped under a band. The final axes show the class probabilities or regression outputs associated with the cluster.

Pass `scaled=True` to multiply each feature axis by the square root of its weight on one shared range, `show_data=False` to keep only the centroids, and `colour="class"` to colour a classifier's lines by class instead of by cluster.

## Local manifold geometry

For a fitted `manifold_network` and observations `X` with at least three continuous features, you can display the local manifolds in three dimensions: planes, or lines when `subspace_dimension` is 1. A `subspace_dimension` above 2 cannot be drawn, and `plot_manifold` raises an error.

```
from entlearn.plotting import plot_manifold

geometry = plot_manifold(manifold_network, X, y, probes=probes)
```

Neighbouring manifolds meet like the cells of a curved Voronoi diagram.

When the network learned instance weights, a manifold also ends where the inlier percentile falls below that of the least typical observation in `X`, and it fades as the percentile drops. The observations are coloured by the supplied `y` (prediction can be used, as well as original labels), and training observations with larger fitted instance weights appear larger.

Pass additional points, such as test observations, as `probes` to show their reconstructions. Their size and colour indicate their inlier percentiles.

Click the legend to show each probe's origin and its movement to its end, or the inlier percentile over the whole space, drawn as a cloud. The legend can also colour the manifolds by what the network predicts at each location. With more than three features, the view shows the first three, but you can choose others with `features`.

See [Manifold](https://entropiclearning.github.io/entlearn/0.1.0/guide/manifold/index.md) for the geometry and [Tutorial 6](https://entropiclearning.github.io/entlearn/0.1.0/tutorials/series/#tutorial-6) for a worked example. See the [plotting reference](https://entropiclearning.github.io/entlearn/0.1.0/reference/plotting/index.md) for all options.
