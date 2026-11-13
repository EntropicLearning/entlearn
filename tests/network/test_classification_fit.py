"""Public behaviour of a fitted shallow classification Network."""

from dataclasses import replace
from itertools import pairwise

import pytest
import torch
from _alloc import assert_no_float64, assert_zero_alloc
from conftest import DEVICE, DTYPE, FIXED_BUDGET
from network._fixtures import (
    BLOB_SPREADS,
    assert_deep_coordinates_non_increasing,
    assert_head_then_input_non_increasing,
    assert_non_increasing,
    assert_simplex_rows,
    blobs,
    classification_recipe,
    deep_classification_recipe,
    materialise,
)

from entlearn import (
    ClassificationHead,
    Coupling,
    Input,
    Network,
    PredictConfig,
    Recipe,
)
from entlearn.network.data import _ClassificationSupervision
from entlearn.network.fit import _fit_iteration_, _loss
from entlearn.network.predict import predict
from entlearn.primitives.normalise import _tiny as _tiny_value

# The temperature regimes the allocation probe uses, matching the monotonicity grid: the
# fixed temperatures take an early return that the finite ones do not.
_ALLOC_TEMPERATURES = (
    (0.1, float("inf"), float("inf")),
    (0.1, 0.2, float("inf")),
    (0.1, float("inf"), 0.2),
    (0.1, 0.2, 0.2),
)


def _separable() -> tuple[torch.Tensor, torch.Tensor]:
    """Return a one-feature, two-class problem with an obvious split."""
    X_cont = torch.tensor([[0.0], [0.1], [0.2], [0.8], [0.9], [1.0]], dtype=DTYPE, device=DEVICE)
    y = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int64, device=DEVICE)
    return X_cont, y


def _tiny() -> tuple[torch.Tensor, torch.Tensor]:
    """Return the smallest two-class problem the fit accepts."""
    X_cont = torch.tensor([[0.0], [0.2], [0.8], [1.0]], dtype=DTYPE, device=DEVICE)
    y = torch.tensor([0, 0, 1, 1], dtype=torch.int64, device=DEVICE)
    return X_cont, y


class TestShallowClassificationFit:
    def test_predicts_useful_class_distributions(self) -> None:
        X_cont, y = _separable()
        recipe = classification_recipe(Input(K=2, epsilon=0.05))

        network = Network.fit(recipe, X_cont, y, max_iter=20, seed=7)
        probabilities = network.predict(X_cont)

        assert_simplex_rows(probabilities)
        assert torch.equal(probabilities.argmax(dim=1), y)

    def test_diagnostics_record_complete_iterations(self) -> None:
        X_cont, y = _tiny()

        network = Network.fit(classification_recipe(), X_cont, y, max_iter=3, tol=0.0)
        diagnostics = network.diagnostics

        assert diagnostics.n_iter == len(diagnostics.loss_history) - 1
        assert 1 <= diagnostics.n_iter <= 3
        for previous, current in pairwise(diagnostics.loss_history):
            assert_non_increasing(previous, current, "iteration")

    def test_a_converged_fit_stops_early_and_records_convergence(self) -> None:
        X_cont, y = _separable()

        network = Network.fit(classification_recipe(), X_cont, y, max_iter=50, tol=1e-6, seed=7)

        assert network.diagnostics.converged is True
        assert network.diagnostics.n_iter < 50
        assert network.diagnostics.warnings == ()

    @FIXED_BUDGET
    def test_finite_weight_temperatures_learn_and_infinite_temperatures_pin(self) -> None:
        X_cont = torch.tensor(
            [[0.0, 0.0], [0.1, 0.1], [0.2, 0.3], [0.8, 0.7], [0.9, 0.8], [1.0, 1.0]],
            dtype=DTYPE,
            device=DEVICE,
        )
        y = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int64, device=DEVICE)
        frozen_recipe = classification_recipe(
            Input(K=2, epsilon=0.05, epsilon_D=float("inf"), epsilon_T=float("inf"))
        )
        learned_recipe = classification_recipe(
            Input(K=2, epsilon=0.05, epsilon_D=0.2, epsilon_T=0.2)
        )
        frozen_state = Network.initialise(frozen_recipe, X_cont, y, seed=3)
        learned_state = Network.initialise(learned_recipe, X_cont, y, seed=3)

        frozen = Network.fit(frozen_recipe, X_cont, y, initial_state=frozen_state, max_iter=2)
        learned = Network.fit(learned_recipe, X_cont, y, initial_state=learned_state, max_iter=2)

        uniform_rows = torch.full((6,), 1.0 / 6, dtype=DTYPE, device=DEVICE)
        assert torch.equal(
            frozen.inspect("feature_weights")["input"], frozen_state.input_geometry.feature_weights
        )
        assert torch.equal(frozen.inspect("training_instance_weights")["input"], uniform_rows)
        assert not torch.equal(
            learned.inspect("feature_weights")["input"],
            learned_state.input_geometry.feature_weights,
        )
        assert not torch.equal(learned.inspect("training_instance_weights")["input"], uniform_rows)
        assert torch.equal(
            learned_state.input_geometry.feature_weights,
            torch.full((2,), 0.5, dtype=DTYPE, device=DEVICE),
        )

    @FIXED_BUDGET
    def test_pruning_updates_the_active_schema_and_keeps_prediction_usable(self) -> None:
        X_cont, y = _tiny()
        recipe = classification_recipe(Input(K=3))
        state = Network.initialise(recipe, X_cont, y, seed=2)
        coincident = (
            state.input_geometry.continuous_centroids[:1]
            .expand_as(state.input_geometry.continuous_centroids)
            .clone()
        )

        network = Network.fit(
            recipe,
            X_cont,
            y,
            initial_state=replace(
                state, input_geometry=replace(state.input_geometry, continuous_centroids=coincident)
            ),
            max_iter=1,
        )
        probabilities = network.predict(X_cont)

        assert dict(network.schema.K_active)["input"] == 1
        assert_simplex_rows(probabilities)

    @FIXED_BUDGET
    def test_fitted_recipe_and_device_support_fresh_queries_and_refits(self) -> None:
        X_cont, y = _tiny()
        recipe = classification_recipe(Input(K=2))
        network = Network.fit(recipe, X_cont, y, max_iter=1)

        # A query arriving in the other float dtype converts with the fitted placement.
        other = torch.float32 if DTYPE is torch.float64 else torch.float64
        query = X_cont.to(other).to(device=network.device, dtype=network.schema.computation_dtype)
        torch.testing.assert_close(network.predict(query), network.predict(X_cont))
        refit = Network.fit(network.recipe, X_cont, y, max_iter=1)
        torch.testing.assert_close(refit.predict(query), network.predict(query))
        assert network.recipe == recipe


class TestFitStructuralContract:
    def test_data_and_controls_are_keyword_only_after_the_target(self) -> None:
        X_cont, y = _tiny()

        with pytest.raises(TypeError, match="takes 4 positional arguments"):
            # The extra positional argument is the point of the test.
            Network.fit(classification_recipe(), X_cont, y, None)  # ty: ignore[too-many-positional-arguments]


class TestClassificationFitInvariants:
    @pytest.mark.parametrize("coupling", (Coupling.M, Coupling.S))
    def test_head_seed_is_uniform_despite_imbalanced_weighted_targets(self, coupling) -> None:
        X_cont = torch.tensor([[0.0], [0.2], [0.8], [1.0]], dtype=DTYPE, device=DEVICE)
        y = torch.tensor([0, 0, 0, 1], dtype=torch.int64, device=DEVICE)
        sample_weights = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=DTYPE, device=DEVICE)
        class_weights = torch.tensor([1.0, 2.0], dtype=DTYPE, device=DEVICE)
        session = materialise(
            Recipe.chain(Input(K=3), ClassificationHead(coupling=coupling)),
            X_cont,
            y,
            sample_weights=sample_weights,
            class_weights=class_weights,
            seed=2,
        )

        head = session.graph.head
        axis_size = head.theta.shape[0 if coupling is Coupling.M else 1]
        torch.testing.assert_close(head.theta, torch.full_like(head.theta, 1.0 / axis_size))
        probabilities = predict(
            session.graph,
            X_cont,
            X_cat_new=None,
            config=PredictConfig(
                output_mode="arithmetic" if coupling is Coupling.M else "geometric"
            ),
        ).prediction
        torch.testing.assert_close(probabilities, torch.full_like(probabilities, 0.5))

    def test_head_loss_and_update_read_the_latest_source_affiliations(self) -> None:
        X_cont, y = _separable()
        session = materialise(classification_recipe(Input(K=2, epsilon=0.1)), X_cont, y, seed=4)
        head, source = session.graph.head, session.graph.input
        supervision = session.data.supervision
        assert isinstance(supervision, _ClassificationSupervision)
        head.update_parameters_(session)
        _loss(session)  # counts the labels of these affiliations
        stale = supervision.weighted_target.T @ source.gamma
        source.update_parameters_(session)
        counts = supervision.weighted_target.T @ source.gamma
        assert not torch.allclose(counts, stale)

        expected_loss = -session.graph.head_delta * (counts * head.log_theta).sum()

        session.workspace.loss.zero_()
        head.partial_loss_(session)
        head.update_parameters_(session)

        torch.testing.assert_close(session.workspace.loss, expected_loss)
        torch.testing.assert_close(head.theta, counts / counts.sum(0))

    @pytest.mark.parametrize(
        "temperatures",
        _ALLOC_TEMPERATURES,
        ids=lambda case: f"g{case[0]}-wd{case[1]}-wt{case[2]}",
    )
    @pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weighted"])
    def test_complete_iteration_allocates_no_tensor_storage(self, temperatures, weighted) -> None:
        epsilon, epsilon_D, epsilon_T = temperatures
        X_cont, y = _separable()
        recipe = classification_recipe(
            Input(K=2, epsilon=epsilon, epsilon_D=epsilon_D, epsilon_T=epsilon_T)
        )
        rows = X_cont.shape[0]
        weights = torch.arange(1, rows + 1, dtype=DTYPE, device=DEVICE) if weighted else None
        session = materialise(recipe, X_cont, y, sample_weights=weights, seed=4, warmup=1)

        assert_zero_alloc(_fit_iteration_, session)

    @FIXED_BUDGET
    def test_float32_fit_and_prediction_create_no_float64_tensors(self) -> None:
        X_cont = torch.tensor([[0.0], [0.2], [0.8], [1.0]], dtype=torch.float32)
        y = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
        recipe = classification_recipe(Input(K=2, epsilon=0.1))

        network = assert_no_float64(Network.fit, recipe, X_cont, y, max_iter=2, seed=2)
        probabilities = assert_no_float64(network.predict, X_cont)

        assert probabilities.dtype is torch.float32


class TestHeadPrior:
    @pytest.mark.parametrize("coupling", (Coupling.M, Coupling.S))
    @pytest.mark.parametrize("weighted", (False, True), ids=("unweighted", "weighted"))
    @pytest.mark.parametrize("theta_alpha", (1.0, 1.7))
    def test_update_and_loss_add_the_pseudocount_at_unit_total_scale(
        self, coupling, weighted, theta_alpha
    ) -> None:
        X_cont, y = _separable()
        sample_weights = (
            torch.arange(1, X_cont.shape[0] + 1, dtype=DTYPE, device=DEVICE) if weighted else None
        )
        recipe = Recipe.chain(
            Input(K=3, epsilon=0.1), ClassificationHead(coupling=coupling, theta_alpha=theta_alpha)
        )
        session = materialise(recipe, X_cont, y, sample_weights=sample_weights, seed=4)
        head = session.graph.head
        supervision = session.data.supervision
        assert isinstance(supervision, _ClassificationSupervision)
        counts = supervision.weighted_target.T @ session.graph.input.gamma
        # The weighted target sums to the unit total, as a transition's counts do.
        torch.testing.assert_close(counts.sum(), torch.ones((), dtype=DTYPE, device=DEVICE))
        prior = (theta_alpha - 1.0) / counts.numel()
        axis = 0 if coupling is Coupling.M else 1
        expected = (counts + prior) / (counts + prior).sum(dim=axis, keepdim=True)

        head.update_parameters_(session)

        torch.testing.assert_close(head.theta, expected)
        logs = expected.clamp_min(_tiny_value(DTYPE)).log()
        expected_loss = -session.graph.head_delta * ((counts * logs).sum() + prior * logs.sum())
        session.workspace.loss.zero_()
        head.partial_loss_(session)
        torch.testing.assert_close(session.workspace.loss, expected_loss)

    @pytest.mark.parametrize("theta_alpha", (1.0, 2.0))
    def test_a_prior_keeps_every_class_probability_of_a_pure_cluster_positive(
        self, theta_alpha
    ) -> None:
        X_cont, y = _separable()
        recipe = Recipe.chain(Input(K=2, epsilon=0.0), ClassificationHead(theta_alpha=theta_alpha))
        network = Network.fit(recipe, X_cont, y, seed=4, max_iter=5)
        theta = network.inspect("head_parameters")["output"]["theta"]
        assert bool((theta > 0).all()) is (theta_alpha > 1.0)

    @pytest.mark.parametrize("epsilon", (0.0, 0.2), ids=("hard", "soft"))
    def test_each_coordinate_update_is_non_increasing(self, epsilon) -> None:
        X_cont, y = blobs(3, BLOB_SPREADS["overlapping"], seed=3)
        shallow = Recipe.chain(Input(K=4, epsilon=epsilon), ClassificationHead(theta_alpha=2.0))
        assert_head_then_input_non_increasing(materialise(shallow, X_cont, y, seed=3))
        deep = deep_classification_recipe(
            (4, 3), epsilon=epsilon, theta_alpha=1.5, head_theta_alpha=2.0
        )
        assert_deep_coordinates_non_increasing(materialise(deep, X_cont, y, seed=3, warmup=2))

    def test_pruning_keeps_the_pseudocount_of_the_initial_widths(self) -> None:
        X_cont, y = _separable()
        recipe = Recipe.chain(Input(K=3, epsilon=0.1), ClassificationHead(theta_alpha=2.0))
        head = materialise(recipe, X_cont, y, seed=4).graph.head
        pseudocount = head.pseudocount
        keep = torch.tensor([True, False, True], device=DEVICE)

        head.prune_source(keep, torch.empty(2, dtype=DTYPE, device=DEVICE))

        assert head.theta.shape == (2, 2)
        assert head.initial_widths == (2, 3)
        assert head.pseudocount == pseudocount == 1.0 / 6

    @FIXED_BUDGET
    def test_the_prior_and_its_widths_survive_a_save_and_load(self, tmp_path) -> None:
        X_cont, y = _separable()
        recipe = Recipe.chain(Input(K=3, epsilon=0.1), ClassificationHead(theta_alpha=2.0))
        source = Network.fit(recipe, X_cont, y, seed=4, max_iter=3)
        path = tmp_path / "prior.safetensors"
        source.save(path)
        restored = Network.load(path, device=DEVICE)

        assert restored.recipe == source.recipe
        assert restored._graph.head.initial_widths == source._graph.head.initial_widths
        expected = source.fine_tune(X_cont, y, max_iter=2)
        actual = restored.fine_tune(X_cont, y, max_iter=2)
        torch.testing.assert_close(actual.predict(X_cont), expected.predict(X_cont), rtol=0, atol=0)
