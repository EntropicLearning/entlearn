"""Every plotting function draws a figure for every block type, without changing its inputs."""

import plotly.graph_objects as go
import pytest
import torch
from conftest import DEVICE, DTYPE

from entlearn import (
    ClassificationHead,
    Hidden,
    Input,
    ManifoldInput,
    Network,
    Recipe,
    RegressionHead,
)
from entlearn.plotting import (
    plot_affiliations,
    plot_centroids,
    plot_decision,
    plot_feature_importance,
    plot_loss,
    plot_manifold,
    plot_parallel,
    plot_theta,
)


@pytest.fixture(scope="module")
def classifier():
    generator = torch.Generator().manual_seed(3)
    X = torch.rand(90, 3, generator=generator, dtype=DTYPE).to(DEVICE)
    y = (X[:, 0] > 0.5).long() + (X[:, 1] > 0.6).long()
    recipe = Recipe.chain(Input(K=6, epsilon_T=0.2), Hidden(K=4), ClassificationHead())
    return Network.fit(recipe, X, y, seed=1, max_iter=20), X, y


@pytest.fixture(scope="module")
def categorical():
    generator = torch.Generator().manual_seed(5)
    X = torch.rand(90, 3, generator=generator, dtype=DTYPE).to(DEVICE)
    codes = torch.randint(0, 3, (90,), generator=generator).to(DEVICE)
    y = ((X[:, 0] > 0.5) ^ (codes == 1)).long()
    recipe = Recipe.chain(Input(K=6, epsilon_T=0.2), Hidden(K=4), ClassificationHead())
    return Network.fit(recipe, X, y, X_cat=[codes], seed=1, max_iter=20), X, [codes], y


@pytest.fixture(scope="module")
def manifold():
    generator = torch.Generator().manual_seed(4)
    t = torch.rand(80, 1, generator=generator, dtype=DTYPE)
    X = torch.cat([t, t**2, 0.05 * torch.rand(80, 1, generator=generator, dtype=DTYPE)], dim=1)
    recipe = Recipe.chain(ManifoldInput(K=4, subspace_dimension=2, epsilon_T=0.5), RegressionHead())
    return Network.fit(recipe, X.to(DEVICE), t.to(DEVICE), seed=1, max_iter=20), X.to(DEVICE), t


class TestSmallFunctions:
    def test_each_draws_a_figure_and_leaves_its_input_alone(self, classifier):
        network, X, _ = classifier
        centroids = network.inspect("continuous_centroids")["input"]
        gamma = network.inspect("training_affiliations")["input"]
        theta = network.inspect("transition_matrices")["input_to_hidden_1"]
        before = centroids.clone()
        figures = [
            plot_loss(network.diagnostics.loss_history),
            plot_feature_importance(network.inspect("feature_weights")["input"]),
            plot_centroids(centroids, X, theta=theta, affiliations=gamma, wt=torch.ones(len(X))),
            plot_affiliations(X, gamma, centroids),
            plot_theta(theta),
        ]
        assert all(isinstance(figure, go.Figure) and figure.data for figure in figures)
        assert torch.equal(centroids, before)

    def test_draws_into_a_subplot(self, classifier):
        from plotly.subplots import make_subplots

        network, _, _ = classifier
        figure = make_subplots(rows=1, cols=2)
        plot_theta(
            network.inspect("transition_matrices")["input_to_hidden_1"], fig=figure, row=1, col=2
        )
        assert figure.data[0].xaxis == "x2"
        assert figure.data[0].colorbar.x > figure.layout.xaxis2.domain[1]

    def test_class_probabilities_take_the_class_colours(self, classifier):
        network, X, y = classifier
        labelled = plot_decision(network, X, y).data[-1].marker.color
        figure = plot_affiliations(X, torch.nn.functional.one_hot(y), classes=True)
        assert list(figure.data[0].marker.color) == list(labelled)


class TestNetworkFigures:
    @pytest.mark.parametrize("block", ["input", "hidden_1", "output", None])
    def test_classifier_blocks(self, classifier, block):
        network, X, _ = classifier
        assert network.plot(block=block).data
        assert network.plot(block=block, X_cont=X).data

    def test_manifold_and_regression_blocks(self, manifold):
        network, X, _ = manifold
        titles = {annotation.text for annotation in network.plot(X_cont=X).layout.annotations}
        assert "input · feature participation" in titles
        with pytest.raises(ValueError, match="X_cont"):
            network.plot(block="input")

    def test_unknown_block(self, classifier):
        with pytest.raises(ValueError, match="unknown block"):
            classifier[0].plot(block="missing")


class TestCategoricalData:
    def test_every_plot_takes_categorical_features(self, categorical):
        network, X, X_cat, y = categorical
        centroids = network.inspect("continuous_centroids")["input"]
        gamma = network.predict_with_details(X, X_cat=X_cat, details=("affiliations",))
        assert gamma.affiliations is not None
        affiliations = gamma.affiliations["input"]
        figures = [
            network.plot(X_cont=X, X_cat=X_cat),
            plot_decision(network, X, y, X_cat=X_cat),
            plot_centroids(centroids, X, affiliations=affiliations),
            plot_affiliations(X, affiliations, centroids),
        ]
        assert all(figure.data for figure in figures)
        assert len(plot_decision(network, X, X_cat=X_cat).data) == 5


class TestParallel:
    def test_every_centroid_crosses_every_axis(self, categorical):
        network, X, X_cat, y = categorical
        K, T = network.inspect("continuous_centroids")["input"].shape[0], len(X)
        dimensions = plot_parallel(network, X, y, X_cat=X_cat).data[0].dimensions
        labels = [dimension.label for dimension in dimensions]
        assert labels == ["cluster", "x0", "x1", "x2", "0", "1", "2", "P(class 1)", "label"]
        assert all(len(dimension.values) == T + K for dimension in dimensions)
        hidden = plot_parallel(network, X, y, X_cat=X_cat, show_data=False).data[0].dimensions
        assert all(len(dimension.values) == K for dimension in hidden)
        assert [d.range for d in hidden] == [d.range for d in dimensions]
        scaled = plot_parallel(network, X, X_cat=X_cat, scaled=True, colour="class")
        assert scaled.data[0].dimensions[1].label == "√w·x0"

    def test_rejects_an_unknown_or_impossible_colouring(self, categorical, manifold):
        network, X, X_cat, _ = categorical
        with pytest.raises(ValueError, match="colour"):
            plot_parallel(network, X, X_cat=X_cat, colour="size")  # type: ignore[arg-type]
        regression, X_manifold, _ = manifold
        with pytest.raises(ValueError, match="classification"):
            plot_parallel(regression, X_manifold, colour="class")


class TestDecisionAndManifold:
    def test_decision_with_and_without_scores(self, classifier):
        network, X, y = classifier
        assert len(plot_decision(network, X, y).data) == 5
        unscored = Network.fit(Recipe.chain(Input(K=4), ClassificationHead()), X, y, seed=1)
        assert len(plot_decision(unscored, X).data) == 3
        with pytest.raises(ValueError, match="classification"):
            plot_decision(
                Network.fit(Recipe.chain(Input(K=3), RegressionHead()), X, X[:, :1], seed=1), X
            )

    def test_manifold_with_probes(self, manifold):
        network, X, t = manifold
        probes = torch.rand(30, 3, dtype=X.dtype, device=X.device)
        figure = plot_manifold(network, X, t, probes=probes)
        names = {trace.name for trace in figure.data}
        assert {"Manifolds", "Observations", "Probes (end)", "Inlier fade"} <= names
        assert figure.layout.scene.dragmode == "turntable"

    def test_manifold_chooses_its_features_and_output(self, manifold):
        _, X, t = manifold
        recipe = Recipe.chain(ManifoldInput(K=4, subspace_dimension=2), RegressionHead())
        network = Network.fit(recipe, X, torch.cat([t, -t], dim=1).to(X.device), seed=1)
        figure = plot_manifold(network, X, features=(2, 0, 1), output=1)
        observations = next(trace for trace in figure.data if trace.name == "Observations")
        planes = next(trace for trace in figure.data if trace.name == "Manifolds (prediction)")
        second = network.predict(X)[:, 1]
        assert figure.layout.scene.xaxis.title.text == "feature 2"
        assert list(observations.x) == X[:, 2].tolist()
        assert (planes.cmin, planes.cmax) == (float(second.min()), float(second.max()))
        for bad in [dict(features=(0, 0, 1)), dict(features=(0, 1, 3)), dict(output=2)]:
            with pytest.raises(ValueError, match=r"features|output"):
                plot_manifold(network, X, **bad)

    def test_manifold_planes_coloured_by_prediction(self, manifold):
        network, X, t = manifold
        traces = plot_manifold(network, X, t).data
        planes = [trace for trace in traces if trace.name == "Manifolds (prediction)"]
        observations = next(trace for trace in traces if trace.name == "Observations")
        assert planes and all(trace.visible == "legendonly" for trace in planes)
        scale = (observations.marker.cmin, observations.marker.cmax)
        assert all((trace.cmin, trace.cmax) == scale for trace in planes)
        assert scale == (float(t.min()), float(t.max()))

    def test_manifold_classifier_planes_mix_the_class_colours(self, manifold):
        _, X, t = manifold
        y = (t[:, 0] > 0.5).long().to(X.device)
        recipe = Recipe.chain(ManifoldInput(K=4, subspace_dimension=2), ClassificationHead())
        network = Network.fit(recipe, X, y, seed=1, max_iter=20)
        traces = plot_manifold(network, X, y).data
        planes = [trace for trace in traces if trace.name == "Manifolds (prediction)"]
        observations = next(trace for trace in traces if trace.name == "Observations")
        labelled = plot_decision(network, X, y).data[-1].marker.color
        assert planes and all(trace.vertexcolor is not None for trace in planes)
        assert list(observations.marker.color) == list(labelled)

    def test_manifold_classifier_lines_are_coloured_lines(self, manifold):
        _, X, t = manifold
        y = (t[:, 0] > 0.5).long().to(X.device)
        recipe = Recipe.chain(ManifoldInput(K=3, subspace_dimension=1), ClassificationHead())
        network = Network.fit(recipe, X, y, seed=1, max_iter=20)
        traces = plot_manifold(network, X, y).data
        lines = [trace for trace in traces if trace.name == "Manifolds (prediction)"]
        assert lines and all(trace.mode == "lines" for trace in lines)
        assert all(len(trace.line.color) == len(trace.x) for trace in lines)

    def test_manifold_refuses_three_dimensional_manifolds(self, manifold):
        _, X, t = manifold
        recipe = Recipe.chain(ManifoldInput(K=2, subspace_dimension=3), RegressionHead())
        network = Network.fit(recipe, X, t, seed=1, max_iter=20)
        with pytest.raises(ValueError, match="subspace_dimension=3"):
            plot_manifold(network, X)
