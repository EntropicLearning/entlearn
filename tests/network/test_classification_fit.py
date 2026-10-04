"""Public behaviour of a fitted shallow classification Network."""

from dataclasses import FrozenInstanceError, replace
from itertools import pairwise

import pytest
import torch
from _alloc import assert_no_float64, assert_zero_alloc
from conftest import DEVICE, DTYPE
from network._fixtures import (
    assert_non_increasing,
    assert_simplex_rows,
    classification_recipe,
    materialise,
)

from entlearn import (
    ClassificationHead,
    Coupling,
    Input,
    LossIncreaseWarning,
    Network,
    PredictConfig,
    Recipe,
)
from entlearn.network.blocks import input as inp
from entlearn.network.fit import _fit_iteration_
from entlearn.network.predict import predict

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

    def test_finite_weight_temperatures_learn_and_infinite_temperatures_pin(self) -> None:
        X_cont = torch.tensor(
            [[0.0, 0.0], [0.1, 1.0], [0.2, 3.0], [0.8, 7.0], [0.9, 8.0], [1.0, 10.0]],
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
            frozen._graph.input.feature_weights, frozen_state.input_geometry.feature_weights
        )
        assert torch.equal(frozen._graph.input.instance_weights, uniform_rows)
        assert not torch.equal(
            learned._graph.input.feature_weights, learned_state.input_geometry.feature_weights
        )
        assert not torch.equal(learned._graph.input.instance_weights, uniform_rows)
        assert torch.equal(
            learned_state.input_geometry.feature_weights,
            torch.full((2,), 0.5, dtype=DTYPE, device=DEVICE),
        )

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

    def test_fit_diagnostics_are_immutable(self) -> None:
        X_cont, y = _tiny()
        diagnostics = Network.fit(classification_recipe(), X_cont, y, max_iter=1).diagnostics

        with pytest.raises(FrozenInstanceError):
            diagnostics.n_iter = 2

    def test_fitted_recipe_and_device_support_fresh_queries_and_refits(self) -> None:
        X_cont, y = _tiny()
        recipe = classification_recipe(Input(K=2))
        network = Network.fit(recipe, X_cont, y, max_iter=1)

        query = X_cont.to(device=network.device, dtype=network.schema.computation_dtype)
        torch.testing.assert_close(network.predict(query), network.predict(X_cont))
        refit = Network.fit(network.recipe, X_cont, y, max_iter=1)
        torch.testing.assert_close(refit.predict(query), network.predict(query))
        assert network.recipe is recipe
        with pytest.raises(AttributeError):
            network.recipe = recipe.replace_block("input", K=3)
        with pytest.raises(AttributeError):
            network.device = torch.device("cpu")

    def test_a_loss_increase_warns_and_reaches_the_diagnostics(self) -> None:
        X_cont, y = _separable()
        recipe = classification_recipe(Input(K=2, epsilon=0.05))
        session = materialise(recipe, X_cont, y, seed=7, warmup=1)
        # Manipulate the head so the next iteration starts from a worse transition than the
        # one the coordinate steps left behind.
        session.graph.head.theta.copy_(torch.tensor([[0.99, 0.01], [0.01, 0.99]], dtype=DTYPE))

        from entlearn.network.fit import _loss, _record_loss_warning

        history = [_loss(session) - 1.0, _loss(session)]
        with pytest.warns(LossIncreaseWarning, match="loss increased"):
            _record_loss_warning(session, history)

        assert session.warning_messages and "loss increased" in session.warning_messages[0]


class TestFitStructuralContract:
    def test_the_head_updates_before_the_input_block(self) -> None:
        X_cont, y = _separable()
        recipe = classification_recipe(Input(K=2, epsilon=0.05))
        session = materialise(recipe, X_cont, y, seed=7)
        head = session.graph.head
        block = session.graph.input
        gamma_before = block.gamma.clone()

        _fit_iteration_(session)
        after_iteration = head.theta.clone()

        # Recompute the head from the affiliations that stood before the iteration. It
        # matches only because the head read them before the input block moved them on.
        rerun = materialise(recipe, X_cont, y, seed=7)
        assert torch.equal(rerun.graph.input.gamma, gamma_before)
        rerun.graph.head.update_parameters_(rerun)
        assert torch.equal(rerun.graph.head.theta, after_iteration)

    def test_no_terminal_transition_object_exists(self) -> None:
        X_cont, y = _tiny()
        network = Network.fit(classification_recipe(), X_cont, y, max_iter=1)
        graph = network._graph

        assert set(graph.blocks) == {"input", "output"}
        assert graph.terminal.coupling is None
        assert graph.terminal.theta_alpha is None
        assert not hasattr(graph, "transitions")

    def test_publication_releases_the_rebuildable_caches(self) -> None:
        X_cont, y = _tiny()
        network = Network.fit(classification_recipe(), X_cont, y, max_iter=1)

        assert network._graph.input.cache is None
        with pytest.raises(RuntimeError, match="holds no cache"):
            inp.refresh_cache_(network._graph.input, None)

    def test_data_and_controls_are_keyword_only_after_the_target(self) -> None:
        X_cont, y = _tiny()

        with pytest.raises(TypeError):
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

    @pytest.mark.parametrize(
        "temperatures",
        _ALLOC_TEMPERATURES,
        ids=lambda case: f"g{case[0]}-wd{case[1]}-wt{case[2]}",
    )
    def test_complete_iteration_allocates_no_tensor_storage(self, temperatures) -> None:
        epsilon, epsilon_D, epsilon_T = temperatures
        X_cont, y = _separable()
        recipe = classification_recipe(
            Input(K=2, epsilon=epsilon, epsilon_D=epsilon_D, epsilon_T=epsilon_T)
        )
        session = materialise(recipe, X_cont, y, seed=4, warmup=1)

        assert_zero_alloc(_fit_iteration_, session)

    def test_float32_fit_and_prediction_create_no_float64_tensors(self) -> None:
        X_cont = torch.tensor([[0.0], [0.2], [0.8], [1.0]], dtype=torch.float32)
        y = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
        recipe = classification_recipe(Input(K=2, epsilon=0.1))

        network = assert_no_float64(Network.fit, recipe, X_cont, y, max_iter=2, seed=2)
        probabilities = assert_no_float64(network.predict, X_cont)

        assert probabilities.dtype is torch.float32
