import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", css_file="entlearn.css")


@app.cell
def _():
    import torch
    from rich import print

    return print, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Creating recipes

    Recipes can receive arbitrary sequences of blocks, but there are rules.

    There are three flavours of blocks: input, hidden and heads.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.mermaid("""
    flowchart LR
        I[Input] --> H[Hidden]
        H --> C[Head]
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Inputs
    As inputs, you can select either an `Input` or a `ManifoldInput`.
    """)
    return


@app.cell
def _():
    from entlearn import Input, ManifoldInput

    Input()
    return Input, ManifoldInput


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    An `Input` accepts as main parameters:

    |Parameter name | Meaning|
    |---------------|--------|
    |`K`|The number of clusters to be used in the block|
    |`epsilon`|Temperature for the *affiliations*|
    |`epsilon_D`|Temperature for the *feature* weights|
    |`epsilon_T`|Temperature for the *instance* weights|
    |`delta_cat`|Importance of categorical features relative to the real-valued ones|
    """)
    return


@app.cell
def _(ManifoldInput):
    ManifoldInput()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The parameters for `ManifoldInput` are explained in the corresponding notebook.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Hidden

    There is currently only one type of hidden layer
    """)
    return


@app.cell
def _():
    from entlearn import Hidden

    Hidden()
    return (Hidden,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Its main parameters are `K` and `epsilon`, which have the same meaning as in the `Input`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Heads

    The two types of available heads are `ClassificationHead` and `RegressionHead`
    """)
    return


@app.cell
def _():
    from entlearn import ClassificationHead, RegressionHead

    ClassificationHead()
    return ClassificationHead, RegressionHead


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A `ClassificationHead` accepts as main parameters:

    |Parameter name | Meaning|
    |---------------|--------|
    |`coupling`|Which way `theta` sums to one (`M`: each column is a distribution over the targets, `S`: each row is a distribution over the sources)|
    |`n_classes`|The number of classes|
    """)
    return


@app.cell
def _(RegressionHead):
    RegressionHead()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A `RegressionHead` accepts as main parameters:

    |Parameter name | Meaning|
    |---------------|--------|
    |`epsilon_M`|Temperature for the *output weights*|
    |`W_M`|Fixed weights for output dimensions|
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Connecting blocks

    Blocks need to be connected together.
    A list of connections need to be provided, specifying the input and outputs.

    This can be done manually as shown below:
    """)
    return


@app.cell
def _(ClassificationHead, Hidden, Input, print):
    from entlearn import Connection, Recipe

    recipe = Recipe(
        blocks=(
            Input(K=8, name="in"),
            Hidden(K=4, name="hid"),
            ClassificationHead(name="out"),
        ),
        connections=(
            Connection("in_to_hid", "in", "hid"),
            Connection("hid_to_out", "hid", "out"),
        ),
    )

    print(recipe)
    return (Recipe,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Initialisation

    `Network.initialise` creates the starting point of a fit, without fitting.
    Each connection gets its own seed, which depends only on the main seed and the connection name: a deeper recipe starts from the same point as a shallower one.
    """)
    return


@app.cell
def _(ClassificationHead, Hidden, Input, Recipe, make_worms, print, torch):
    from entlearn import Network

    X, y = make_worms(noise_dims=0)
    X, y = torch.tensor(X), torch.tensor(y)

    shallow = Recipe.chain(Input(K=3), Hidden(K=2), ClassificationHead())
    deeper = Recipe.chain(Input(K=3), Hidden(K=2), Hidden(K=2), ClassificationHead())

    print(Network.initialise(shallow, X, y, seed=17).connection_sub_seeds)
    print(Network.initialise(deeper, X, y, seed=17).connection_sub_seeds)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## General rules

    In general, the following diagram illustrates the allowed connections, which enable creation of classification or regression networks with either standard inputs or manifold inputs and arbitrary depth.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.mermaid("""
    ---
    config:
      flowchart:
        curve: basis
    ---
    flowchart TD
        subgraph inputs [Inputs]
            I[Input]
            M[Manifold]
        end

        subgraph heads [Heads]
            C[Classification]
            R[Regression]
        end

        I --> H
        M --> H
        H --> C
        H --> R
        I --> C
        I --> R
        H --> H
        M --> C
        M --> R
        inputs -.->|shallow| heads
    """)
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


@app.cell(hide_code=True)
def _():
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

    return (make_worms,)


if __name__ == "__main__":
    app.run()
