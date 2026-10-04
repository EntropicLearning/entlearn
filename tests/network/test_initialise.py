"""Public contract tests for Network initialisation."""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
import torch
from network._fixtures import classification_recipe, continuous_classification_data

from entlearn import (
    ClassificationHead,
    Coupling,
    DataSchema,
    Hidden,
    InitialState,
    Input,
    Network,
    Recipe,
    RegressionHead,
)


def _initial_state(**overrides) -> InitialState:
    fields = dict(
        input=Input(K=3),
        continuous_centroids=torch.zeros(3, 2),
        categorical_centroids=(),
        feature_weights=torch.full((2,), 0.5),
        manifold_projectors=None,
        captured=False,
    )
    fields.update(overrides)
    generated = Network.initialise(
        Recipe.chain(Input(K=3), ClassificationHead(coupling=Coupling.M)),
        torch.zeros(3, 2),
        torch.tensor([0, 1, 0]),
    )
    return replace(
        generated,
        input_geometry=replace(generated.input_geometry, **fields),
        connection_sub_seeds=(("input_to_output", 7),),
    )


class TestNetworkLifecycle:
    def test_rejects_direct_construction(self):
        with pytest.raises(TypeError, match="cannot be constructed directly"):
            Network()

    def test_requires_an_exact_recipe(self):
        with pytest.raises(ValueError, match="Recipe must be an exact Recipe"):
            Network.initialise(None, object(), object())


class TestLifecyclePreliminaryChecks:
    @pytest.mark.parametrize("seed", (True, -1, 2**63), ids=("bool", "negative", "oversized"))
    def test_rejects_invalid_root_seed_before_data_access(self, seed):
        with pytest.raises(ValueError, match="seed must be an integer in"):
            Network.initialise(classification_recipe(K=2), object(), object(), seed=seed)

    @pytest.mark.parametrize("n_inits", (1, 2))
    def test_numpy_integer_seed_plans_the_same_seeds_as_int(self, n_inits):
        X, y = continuous_classification_data()
        recipe = classification_recipe(K=2)

        def planned(seed):
            fitted = Network.fit(recipe, X, y, seed=seed, n_inits=n_inits, max_iter=1)
            return tuple(outcome.seed for outcome in fitted.diagnostics.initialisation_outcomes)

        expected = planned(12)
        assert len(expected) == n_inits
        assert planned(np.int64(12)) == expected

    def test_several_candidates_draw_their_seeds_below_the_bound_from_the_root(self):
        X, y = continuous_classification_data()
        fitted = Network.fit(classification_recipe(K=2), X, y, seed=12, n_inits=3, max_iter=1)
        generator = torch.Generator(device=X.device).manual_seed(12)
        expected = torch.randint(2**63 - 1, (3,), generator=generator, device=X.device)
        seeds = [outcome.seed for outcome in fitted.diagnostics.initialisation_outcomes]
        assert seeds == expected.tolist()


class TestStateValues:
    def test_data_schema_is_frozen(self):
        schema = DataSchema(
            task="classification",
            D_cont=2,
            M_cat=(3, 4),
            M=5,
            K_active=(("input", 3), ("hidden", 2), ("output", 5)),
            computation_dtype=torch.float32,
        )

        with pytest.raises(FrozenInstanceError):
            schema.M = 6

    def test_initial_state_is_frozen_and_reads_its_active_k_from_the_centroids(self):
        state = _initial_state(input=Input(K=5), continuous_centroids=torch.zeros(3, 2))

        assert state.input_geometry.K_active == 3
        with pytest.raises(FrozenInstanceError):
            state.input_geometry = None
        with pytest.raises(FrozenInstanceError):
            state.input_geometry.captured = True

    def test_comparing_two_initial_states_answers_instead_of_raising(self):
        recipe = Recipe.chain(Input(K=3), ClassificationHead(coupling=Coupling.M))
        X = torch.rand(6, 2, dtype=torch.float64)
        y = torch.tensor([0, 1, 0, 1, 0, 1])
        first = Network.initialise(recipe, X, y, seed=1)
        second = Network.initialise(recipe, X, y, seed=1)

        assert torch.equal(
            first.input_geometry.continuous_centroids, second.input_geometry.continuous_centroids
        )
        assert first != second
        assert second not in [first]
        assert first.input_geometry != second.input_geometry

    @pytest.mark.parametrize(
        ("input_block", "captured", "expected"),
        (
            pytest.param(Input(K=3), False, True, id="kmeans++"),
            pytest.param(
                Input(K=3, centroid_strategy="greedy-kmeans++", greedy_candidates=2),
                False,
                True,
                id="fixed-greedy",
            ),
            pytest.param(
                Input(K=3, centroid_strategy="greedy-kmeans++"),
                False,
                False,
                id="k-dependent-greedy",
            ),
            pytest.param(Input(K=3, balanced=True), False, False, id="balanced"),
            pytest.param(Input(K=3), True, False, id="captured"),
        ),
    )
    def test_prefix_reuse_follows_the_seeding_strategy_and_provenance(
        self, input_block, captured, expected
    ):
        state = _initial_state(input=input_block, captured=captured)

        assert state.input_geometry.prefix_reusable is expected


class TestStandardContinuousInitialisation:
    @pytest.mark.parametrize(
        ("recipe", "y"),
        (
            pytest.param(classification_recipe(K=2), torch.tensor(0), id="classification"),
            pytest.param(
                Recipe.chain(Input(K=2), RegressionHead()),
                torch.tensor(0.0, dtype=torch.float64),
                id="regression",
            ),
        ),
    )
    def test_rejects_a_scalar_target(self, recipe, y):
        X_cont = torch.tensor([[0.0], [1.0]], dtype=torch.float64)

        with pytest.raises(ValueError, match="y must be at least one-dimensional"):
            Network.initialise(recipe, X_cont, y)

    @pytest.mark.parametrize("non_tensor", ["X_cont", "y"])
    def test_rejects_non_tensor_features_or_targets(self, non_tensor):
        X_cont, y = continuous_classification_data()
        if non_tensor == "X_cont":
            X_cont = X_cont.tolist()
        else:
            y = y.tolist()

        with pytest.raises(ValueError, match="X_cont and y must be tensors"):
            Network.initialise(classification_recipe(K=2), X_cont, y)

    def test_initialises_classification_from_a_chain(self):
        X_cont, y = continuous_classification_data()
        recipe = classification_recipe(K=2)

        state = Network.initialise(recipe, X_cont, y, seed=4)

        assert isinstance(state, InitialState)
        assert state.input_geometry.input == recipe.blocks[0]
        assert state.input_geometry.continuous_centroids.shape == (2, 2)
        assert state.input_geometry.K_active == 2
        assert torch.equal(
            state.input_geometry.feature_weights, torch.full((2,), 0.5, dtype=torch.float64)
        )
        assert state.input_geometry.categorical_centroids == ()
        assert state.input_geometry.manifold_projectors is None
        assert not state.input_geometry.captured
        assert state.input_geometry.prefix_reusable
        assert tuple(name for name, _ in state.connection_sub_seeds) == ("input_to_output",)

    def test_validates_fixed_regression_weights_before_geometry_capacity(self):
        recipe = Recipe.chain(Input(K=5), RegressionHead(W_M=(1.0,)))
        X_cont = torch.zeros(3, 2, dtype=torch.float64)
        y = torch.zeros(3, 2, dtype=torch.float64)

        with pytest.raises(
            ValueError, match="fixed W_M has length 1 but the output dimension is 2"
        ):
            Network.initialise(recipe, X_cont, y)

    def test_validates_fixed_regression_weights_before_finite_feature_scan(self):
        recipe = Recipe.chain(Input(K=5), RegressionHead(W_M=(1.0,)))
        X_cont = torch.tensor(
            [[float("nan"), 0.0], [0.0, 1.0], [1.0, 0.0]],
            dtype=torch.float64,
        )
        y = torch.zeros(3, 2, dtype=torch.float64)

        with pytest.raises(
            ValueError, match="fixed W_M has length 1 but the output dimension is 2"
        ):
            Network.initialise(recipe, X_cont, y)

    def test_initialises_valid_multi_output_regression_without_output_state(self):
        recipe = Recipe.chain(Input(K=2), RegressionHead(W_M=(1.0, 2.0)))
        X_cont = torch.tensor(
            [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
            dtype=torch.float32,
        )
        y = torch.tensor([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]], dtype=torch.float32)

        state = Network.initialise(recipe, X_cont, y, seed=3)

        assert state.input_geometry.continuous_centroids.shape == (2, 2)
        assert state.input_geometry.feature_weights is not None
        assert state.input_geometry.feature_weights.dtype is torch.float32


class TestInitialisationDeterminism:
    def test_seeding_does_not_build_an_autograd_graph(self):
        X_cont, y = continuous_classification_data()
        X_cont.requires_grad_()
        saved_tensors = []

        def record_saved_tensor(tensor):
            saved_tensors.append(tensor)
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(record_saved_tensor, lambda tensor: tensor):
            Network.initialise(classification_recipe(K=2), X_cont, y, seed=4)

        assert saved_tensors == []

    def test_is_deterministic_under_its_seed(self):
        X_cont = torch.tensor(
            [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [3.0, 3.0], [4.0, 4.0]],
            dtype=torch.float64,
        )
        y = torch.tensor([0, 0, 1, 1, 0, 1])
        recipe = Recipe.chain(
            Input(K=3),
            Hidden(K=2),
            ClassificationHead(coupling=Coupling.M),
            coupling=Coupling.S,
        )

        first = Network.initialise(recipe, X_cont, y, seed=17)
        repeated = Network.initialise(recipe, X_cont, y, seed=17)
        different = Network.initialise(recipe, X_cont, y, seed=18)

        assert tuple(name for name, _ in first.connection_sub_seeds) == (
            "input_to_hidden_1",
            "hidden_1_to_output",
        )
        assert torch.equal(
            first.input_geometry.continuous_centroids, repeated.input_geometry.continuous_centroids
        )
        assert first.connection_sub_seeds == repeated.connection_sub_seeds
        assert not torch.equal(
            first.input_geometry.continuous_centroids, different.input_geometry.continuous_centroids
        )
        assert first.connection_sub_seeds != different.connection_sub_seeds

    def test_connection_sub_seeds_depend_only_on_the_root_seed_and_name(self):
        X_cont = torch.tensor(
            [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [3.0, 3.0], [4.0, 4.0]],
            dtype=torch.float64,
        )
        y = torch.tensor([0, 0, 1, 1, 0, 1])
        head = ClassificationHead(coupling=Coupling.M)
        narrow = Recipe.chain(Input(K=2), Hidden(K=2), head, coupling=Coupling.M)
        wide = Recipe.chain(Input(K=4), Hidden(K=2), head, coupling=Coupling.M)
        deeper = Recipe.chain(Input(K=2), Hidden(K=2), Hidden(K=2), head, coupling=Coupling.M)

        narrow_seeds = dict(Network.initialise(narrow, X_cont, y, seed=17).connection_sub_seeds)
        wide_seeds = dict(Network.initialise(wide, X_cont, y, seed=17).connection_sub_seeds)
        deeper_seeds = dict(Network.initialise(deeper, X_cont, y, seed=17).connection_sub_seeds)
        other_seeds = dict(Network.initialise(narrow, X_cont, y, seed=18).connection_sub_seeds)

        assert narrow_seeds == wide_seeds
        assert deeper_seeds["input_to_hidden_1"] == narrow_seeds["input_to_hidden_1"]
        assert tuple(deeper_seeds) == (
            "input_to_hidden_1",
            "hidden_1_to_hidden_2",
            "hidden_2_to_output",
        )
        assert len(set(deeper_seeds.values())) == 3
        assert all(0 <= sub_seed < 2**63 for sub_seed in deeper_seeds.values())
        assert other_seeds["input_to_hidden_1"] != narrow_seeds["input_to_hidden_1"]

    def test_falls_back_to_uniform_selection_among_coincident_rows(self):
        X_cont = torch.tensor([[0.0], [0.0], [0.0], [5.0]], dtype=torch.float64)
        y = torch.tensor([0, 1, 0, 1])

        state = Network.initialise(classification_recipe(K=3), X_cont, y, seed=9)

        assert state.input_geometry.continuous_centroids.shape == (3, 1)
        assert state.input_geometry.continuous_centroids.flatten().sort().values.tolist() == [
            0.0,
            0.0,
            5.0,
        ]

    def test_returns_detached_tensors_without_caller_storage_aliases(self):
        X_cont, y = continuous_classification_data()
        X_cont.requires_grad_()

        state = Network.initialise(classification_recipe(K=2), X_cont, y, seed=4)
        original = state.input_geometry.continuous_centroids.clone()
        with torch.no_grad():
            X_cont.add_(10.0)

        assert not state.input_geometry.continuous_centroids.requires_grad
        assert state.input_geometry.feature_weights is not None
        assert not state.input_geometry.feature_weights.requires_grad
        assert state.input_geometry.continuous_centroids.untyped_storage().data_ptr() != (
            X_cont.untyped_storage().data_ptr()
        )
        assert torch.equal(state.input_geometry.continuous_centroids, original)
