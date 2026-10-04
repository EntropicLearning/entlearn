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
    # First steps

    ## Dataset

    In this notebook, we will explore a classification task using a synthetic dataset.
    Only the first two coordinates are informative, the model must be able to identify them.

    In the plot below, test points are indicated in orange
    """)
    return


@app.cell
def _(torch, train_test_split):
    X, y = make_worms(noise_dims=4, seed=7)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=12
    )
    train = torch.tensor(X_train, dtype=torch.float64)
    test = torch.tensor(X_test, dtype=torch.float64)
    targets = torch.tensor(y_train, dtype=torch.int64)
    return X_test, X_train, targets, test, train, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    dim_1 = mo.ui.slider(
        1, 6, value=1, label="Dimension to plot as x", full_width=True, show_value=True
    )
    dim_2 = mo.ui.slider(
        1, 6, value=2, label="Dimension to plot as y", full_width=True, show_value=True
    )
    mo.hstack([dim_1, dim_2])
    return dim_1, dim_2


@app.cell
def _(X_test, X_train, plot_data, y_test, y_train):
    plot_data(X_train, y_train, X_test, y_test)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Recipe

    For this notebook, we will create a simple shallow network, using a `chain` recipe.
    For now, do not worry too much about the parameters, they will be explored in subsequent notebooks.

    The key point is the definition of a recipe using a sequence of blocks.
    """)
    return


@app.cell
def _():
    from entlearn import ClassificationHead, Input, Recipe

    recipe = Recipe.chain(
        Input(K=9, epsilon=0.015, epsilon_D=0.01, epsilon_T=0.15),
        ClassificationHead(),
    )
    return (recipe,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Fit
    Once that is done, we can create and fit a network using `Network.fit()`, and passing as arguments the recipe we just created, the data, the labels and (if we like) some other arguments. As before, do not worry about them now, just know that this is where the **fit parameters** can be specified.
    """)
    return


@app.cell
def _(recipe, targets, train):
    from entlearn import Network

    network = Network.fit(recipe, train, targets, n_inits=5, seed=7, max_iter=150)
    return (network,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## That's it!
    We can now inspect the network.
    """)
    return


@app.cell
def _(network):
    network
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note that if you use `print` from `rich` instead of the normal one, you will get a nicer representation
    """)
    return


@app.cell
def _(network):
    from rich import print

    print(network)
    return (print,)


@app.cell(hide_code=True)
def _(accuracy, mo):
    mo.md(rf"""
    ## Predicting

    Prediction can be performed using the `predict` function (note that `Network`'s predict function returns probabilities), if you are interested in hard labels, you can simply take the argmax. This model reaches {accuracy:.1%} of accuracy on the test set
    """)
    return


@app.cell
def _(network, test, torch, y_test):
    probabilities = network.predict(test)
    predictions = probabilities.argmax(dim=1)
    accuracy = float((predictions == torch.tensor(y_test)).double().mean())
    return (accuracy,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Confidence
    EON models report also their confidence on test predictions, which can be obtained as follows:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let us inspect the outputs more carefully. In the cells below there are two plots:

    - The left plot illustrates the decision function returned by the network
    - The right one instead is the equivalent visualization for the model confidence.

    The test points are also plotted, colored by their prediction probabilities and scaled by the model's confidence.
    """)
    return


@app.cell
def _(network, test, y_test):
    from entlearn.plotting import plot_decision

    plot_decision(network, test, y_test)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Save and reload

    The obtained network can be easily saved to disk:
    ```python
    network.save("network.safetensors")
    ```
    ... and restored:
    ```
    restored = Network.load("network.safetensors")
    ```
    To continue training later, save with `resumable=True`: the reloaded network can `resume` on the same data or `fine_tune` on new data.
    ```python
    network.save("checkpoint.safetensors", resumable=True)
    network = Network.load("checkpoint.safetensors").resume(train, targets, max_iter=50)
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Estimators

    The estimators can be used in a similar manner. In fact, the same recipe can be used with both `Network` and `EONClassifier`.

    `EONClassifier` and `EONRegressor` follow the scikit-learn interface: they accept pandas or Polars data frames with named columns, categorical columns (`categorical_features="from_dtype"`) and string labels.
    They can also be pickled, and continue a previous fit with `warm_start="resume"` or `"fine_tune"`.
    """)
    return


@app.cell
def _(X_train, print, recipe, y_train):
    from entlearn.scikit_adapter import EONClassifier

    estimator = EONClassifier(recipe, n_inits=5, random_state=7, max_iter=150)
    estimator.fit(X_train, y_train)

    # The network inside the estimator is just a regular Network
    print(estimator.network_)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Utility cells
    The cells below define additional imports, plotting helpers and the data generator. They are not relevant for the demonstration, but are left for the curious reader.
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import plotly.graph_objects as go

    return go, mo


@app.cell(hide_code=True)
def _(dim_1, dim_2, go):
    def plot_data(X_train, y_train, X_test, y_test):

        fig = go.Figure(
            go.Scatter(
                x=X_train[:, dim_1.value - 1],
                y=X_train[:, dim_2.value - 1],
                mode="markers",
                showlegend=False,
                marker=dict(
                    color=y_train,
                    colorscale="RdBu",
                    line=dict(width=2, color="black"),
                    showscale=False,
                ),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=X_test[:, dim_1.value - 1],
                y=X_test[:, dim_2.value - 1],
                mode="markers",
                showlegend=False,
                marker=dict(
                    color=y_test,
                    colorscale="RdBu",
                    size=10,
                    line=dict(width=2, color="orange"),
                ),
            ),
        )

        fig.update_layout(
            xaxis_title=f"Feature {dim_1.value}", yaxis_title=f"Feature {dim_2.value}"
        )
        fig.update_layout(title="Data")
        fig.update_xaxes(scaleanchor="y", scaleratio=1)
        return fig

    return (plot_data,)


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
