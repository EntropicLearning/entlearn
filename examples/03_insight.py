import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", css_file="entlearn.css")


@app.cell
def _():
    import torch
    from sklearn.model_selection import train_test_split

    return torch, train_test_split


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Inspecting a fitted network

    Now, let us briefly return to the example from the "First steps" notebook. What can we learn from a fitted network?
    """)
    return


@app.cell
def _(torch, train_test_split):
    from entlearn import ClassificationHead, Input, Network, Recipe

    X, y = make_worms(noise_dims=4, seed=7)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, stratify=y, random_state=12)
    train = torch.tensor(X_train, dtype=torch.float64)
    targets = torch.tensor(y_train, dtype=torch.int64)

    recipe = Recipe.chain(
        Input(K=3, epsilon=0.01, epsilon_D=0.002, epsilon_T=0.05),
        ClassificationHead(),
    )

    network = Network.fit(recipe, train, targets, n_inits=5, seed=7, max_iter=150)
    return X_train, network, y_train


@app.cell(hide_code=True)
def _(mo, network):
    mo.md(rf"""
    ## Fit diagnostics

    Every call to fit returns the diagnostics, including the convergence of the fitting procedure. In this case, it took {network.diagnostics.n_iter} iterations for convergence
    """)
    return


@app.cell
def _(network):
    from entlearn.plotting import plot_loss

    plot_loss(network.diagnostics.loss_history).update_traces(line_color="orange")
    return


@app.cell(hide_code=True)
def _(X_train, centroids, mo):
    mo.md(rf"""
    ## Centroids

    we can extract the centroids. In this case, their shape is {centroids.shape}, as the original dimensionality of the dataset is {X_train.shape[1]}.
    """)
    return


@app.cell
def _(X_train, network):
    from entlearn.plotting import plot_centroids

    centroids = network.inspect("continuous_centroids")["input"]
    plot_centroids(centroids, X_train).update_traces(
        marker_color="orange", selector=dict(name="centroids")
    )
    return centroids, plot_centroids


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Categorical centroids
    In this dataset, there is no categorical feature, but if there were, we would be able to see the categorical centroids using:
    ```python
    network.inspect("categorical_centroids")
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Theta

    In a shallow network, theta links the input centroids to the output probabilities. Per se, it is a matrix
    """)
    return


@app.cell
def _(network):
    from entlearn.plotting import plot_theta

    theta = network.inspect("head_parameters")["output"]["theta"]
    plot_theta(theta, x_title="cluster", y_title="class").update_traces(colorscale="Oranges")
    return (theta,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    but it can be useful to color each centroid by the classes it leads to:
    """)
    return


@app.cell(hide_code=True)
def _(X_train, centroids, plot_centroids, theta):
    plot_centroids(centroids, X_train, theta=theta)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Calibration

    After the fit, the network also calibrates the temperature of its output probabilities, `epsilon_P`, on the training data. It is stored in `predict_config`.
    It can be changed at prediction time with a `PredictConfig`, without refitting. Note that `PredictConfig()` replaces the calibrated temperature, it does not keep it.
    """)
    return


@app.cell
def _(network):
    network.predict_config
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Affiliations

    We can also recover the affiliations of the training data points to the centroids.
    """)
    return


@app.cell
def _(X_train, centroids, network):
    from entlearn.plotting import plot_affiliations

    affiliations = network.inspect("training_affiliations")["input"]
    plot_affiliations(X_train, affiliations, centroids)
    return (affiliations,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now, putting the last three steps together, we can color each training point using the affiliation-weighted probabilities, and overlay the centroids
    """)
    return


@app.cell
def _(X_train, affiliations, centroids, plot_centroids, theta):
    plot_centroids(centroids, X_train, theta=theta, affiliations=affiliations)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Feature weights

    Moreover, when assigning points, the network also learns how much each feature matters. These feature weights sum to one.
    Here, almost all the weight goes to the first two features, while the four noise features get close to zero: the model found the informative coordinates by itself.
    """)
    return


@app.cell
def _(network):
    from entlearn.plotting import plot_feature_importance

    feature_weights = network.inspect("feature_weights")["input"]
    plot_feature_importance(feature_weights).update_traces(marker_color="orange")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training instance weights

    Similarly, the network learns a weight for each training point. These instance weights also sum to one, and points that the centroids describe poorly get less weight.
    Below, the larger a point's weight, the larger it is drawn.
    """)
    return


@app.cell
def _(network):
    inst_weights = network.inspect("training_instance_weights")["input"]
    return (inst_weights,)


@app.cell
def _(X_train, affiliations, centroids, inst_weights, plot_centroids, theta):
    plot_centroids(centroids, X_train, theta=theta, affiliations=affiliations, wt=inst_weights)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As you can see, the points far from the centroids, at the ends of the worms, are the smallest. The same idea gives the confidence seen in the "First steps" notebook: a new point that the centroids describe poorly gets a low score.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Everything at once

    Finally, parallel coordinates show every centroid in every feature, with the training points behind them and their labels on the last axis. Drag along an axis to keep only the lines in that range.
    """)
    return


@app.cell
def _(X_train, network, torch, y_train):
    from entlearn.plotting import plot_parallel

    plot_parallel(network, torch.tensor(X_train), y_train)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Utility cells
    The cells below define additional imports and the data generator. They are not relevant for the demonstration, but are left for the curious reader.
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.function(hide_code=True)
def make_worms(noise_dims: int = 4, seed: int = 7):
    """Three worms along the diagonal (middle one class 1), plus `noise_dims` noise columns."""
    import torch

    g = torch.Generator().manual_seed(seed)

    along = torch.tensor([1.0, -1.0], dtype=torch.float64) / 2**0.5
    across = torch.tensor([1.0, 1.0], dtype=torch.float64) / 2**0.5
    points, labels = [], []
    for centre, label in [(0.25, 0), (0.5, 1), (0.75, 0)]:
        length = 0.16 * torch.randn(55, 1, generator=g, dtype=torch.float64)
        width = 0.0365 * torch.randn(55, 1, generator=g, dtype=torch.float64)
        points.append(centre + length * along + width * across)
        labels.append(torch.full((55,), label))
    X, y = torch.cat(points), torch.cat(labels)

    # Always draw all 32 noise columns, so the first ones do not depend on noise_dims.
    low, high = X.min(), X.max()
    noise = low + (high - low) * torch.rand(len(X), 32, generator=g, dtype=torch.float64)
    X = torch.cat([X, noise[:, :noise_dims]], dim=1)
    return ((X + 0.3) / 1.6).numpy(), y.numpy()


if __name__ == "__main__":
    app.run()
