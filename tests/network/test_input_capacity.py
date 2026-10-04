"""Requested input widths larger than the eligible seed-row pool."""

import warnings
from dataclasses import replace

import pytest
import torch
from conftest import DEVICE, DTYPE

from entlearn import (
    ClassificationHead,
    Coupling,
    Hidden,
    Input,
    ManifoldInput,
    Network,
    Recipe,
    RegressionHead,
)


class TestInputCapacity:
    @pytest.mark.parametrize("classification", [False, True])
    def test_generated_width_warns_and_caps_without_changing_the_recipe(self, classification):
        X = torch.tensor([[0.0], [0.25], [0.75], [1.0]], dtype=DTYPE, device=DEVICE)
        y = torch.tensor([0, 0, 1, 1], device=DEVICE) if classification else X[:, 0]
        recipe = Recipe.chain(
            Input(K=7),
            ClassificationHead(Coupling.M) if classification else RegressionHead(),
        )
        with pytest.warns(UserWarning, match=r"K=7.*4.*reducing"):
            state = Network.initialise(recipe, X, y, seed=3)
        assert state.input_geometry.K_active == 4
        assert state.input_geometry.input == recipe.blocks[0]
        assert recipe.blocks[0].K == 7
        torch.testing.assert_close(state.input_geometry.continuous_centroids.sort(dim=0).values, X)
        with pytest.warns(UserWarning, match=r"K=7.*4.*reducing"):
            fitted = Network.fit(recipe, X, y, seed=3)
        assert fitted.recipe == recipe
        assert dict(fitted.schema.K_active)["input"] == 4
        with pytest.warns(UserWarning, match="holds 4 centroids for the requested K=7"):
            replay = Network.fit(recipe, X, y, initial_state=fitted.initial_state)
        torch.testing.assert_close(fitted.predict(X), replay.predict(X))

    def test_a_width_equal_to_the_row_count_is_not_reduced(self):
        X = torch.tensor([[0.0], [0.25], [0.75], [1.0]], dtype=DTYPE, device=DEVICE)
        recipe = Recipe.chain(Input(K=4), RegressionHead())
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=".*reducing the initial input")
            state = Network.initialise(recipe, X, X[:, 0], seed=3)
        assert state.input_geometry.K_active == 4

    @pytest.mark.parametrize("kind", ["continuous", "mixed", "categorical", "manifold"])
    @pytest.mark.parametrize(
        "strategy,candidates",
        [("kmeans++", None), ("greedy-kmeans++", None), ("greedy-kmeans++", 3)],
    )
    def test_capped_geometry_uses_every_row_and_materialises_hidden_layers(
        self, kind, strategy, candidates
    ):
        X = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=DTYPE, device=DEVICE)
        y = X[:, 0].clone()
        if kind == "categorical":
            X = X[:, :0]
        categories = (
            [torch.tensor([0, 3], device=DEVICE)] if kind in ("mixed", "categorical") else None
        )
        weights = torch.tensor([1.0, 2.0], dtype=DTYPE, device=DEVICE)
        input_type = ManifoldInput if kind == "manifold" else Input
        source = input_type(K=8, centroid_strategy=strategy, greedy_candidates=candidates)
        recipe = Recipe.chain(source, Hidden(K=2), RegressionHead(), coupling=Coupling.M)
        with pytest.warns(UserWarning, match=r"K=8.*2.*reducing"):
            state = Network.initialise(recipe, X, y, X_cat=categories, sample_weights=weights)
        assert state.input_geometry.K_active == 2
        if X.shape[1]:
            torch.testing.assert_close(
                state.input_geometry.continuous_centroids.sort(dim=0).values, X
            )
        if categories is not None:
            assert set(state.input_geometry.categorical_centroids[0].argmax(dim=1).tolist()) == {
                0,
                3,
            }
        if kind == "manifold":
            assert state.input_geometry.manifold_projectors.shape == (2, 2, 1)
        with pytest.warns(UserWarning, match="holds 2 centroids for the requested K=8"):
            fitted = Network.fit(
                recipe, X, y, X_cat=categories, sample_weights=weights, initial_state=state
            )
        assert dict(fitted.schema.K_active)["input"] == 2
        assert torch.isfinite(fitted.predict(X, X_cat=categories)).all()

    @pytest.mark.parametrize("strategy", ["kmeans++", "greedy-kmeans++"])
    def test_balanced_capacity_excludes_unlabelled_rows(self, strategy):
        X = torch.linspace(0, 1, 4, dtype=DTYPE, device=DEVICE)[:, None]
        y = torch.tensor([0, -1, 1, -1], device=DEVICE)
        recipe = Recipe.chain(
            Input(K=4, balanced=True, centroid_strategy=strategy),
            ClassificationHead(Coupling.M),
        )
        with pytest.warns(UserWarning, match=r"K=4.*2.*reducing"):
            fitted = Network.fit(recipe, X, y)
        assert fitted.initial_state.input_geometry.K_active == 2
        torch.testing.assert_close(
            fitted.initial_state.input_geometry.continuous_centroids.sort(dim=0).values, X[[0, 2]]
        )
        assert torch.isfinite(fitted.predict(X)).all()

    @pytest.mark.timeout(10, method="signal")
    def test_spare_balanced_seeds_follow_eligible_class_mass_and_capacity(self):
        """Extra seeds go to the heaviest class with room, counting eligible rows only."""
        # Class 2 is heaviest but holds one row; class 1 is next and holds two.
        # The heavy unlabelled row must not enter either count.
        X = torch.tensor([[0.0], [0.0], [0.4], [0.45], [0.9], [0.6]], dtype=DTYPE, device=DEVICE)
        y = torch.tensor([0, 0, 1, 1, 2, -1], device=DEVICE)
        weights = torch.tensor([0.1, 0.1, 0.5, 0.5, 4.0, 10.0], dtype=DTYPE, device=DEVICE)
        recipe = Recipe.chain(
            Input(K=4, balanced=True),
            ClassificationHead(Coupling.M),
        )
        state = Network.initialise(recipe, X, y, sample_weights=weights)
        assert state.input_geometry.K_active == 4
        torch.testing.assert_close(
            state.input_geometry.continuous_centroids.sort(dim=0).values, X[[0, 2, 3, 4]]
        )

    @pytest.mark.parametrize("backend", ["threads", "processes"])
    def test_validation_selection_retains_capped_original_states_across_refits(self, backend):
        X = torch.linspace(0, 1, 6, dtype=DTYPE, device=DEVICE)[:, None]
        y = X[:, 0]
        recipe = Recipe.chain(Input(K=8), RegressionHead())
        pairs = [(torch.tensor([0, 2, 4], device=DEVICE), torch.tensor([1, 3, 5], device=DEVICE))]
        common_rows = torch.tensor([0, 2], device=DEVICE)

        def loss(prediction, target, **kwargs):
            return (prediction - target).square().mean()

        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter("always")
            fitted = Network.fit(
                recipe,
                X,
                y,
                init_rows=common_rows,
                validation_pairs=pairs,
                selection_loss=loss,
                n_inits=2,
                n_jobs=2,
                parallel_backend=backend,
                retain="members",
                seed=3,
            )
        if backend == "threads":
            assert (
                sum(
                    "reducing the initial input cluster count" in str(item.message)
                    for item in observed
                )
                == 2
            )
        assert len(fitted.members) == 2
        for state, member in zip(fitted.initial_states, fitted.members, strict=True):
            assert state is member.initial_state
            assert state.input_geometry.K_active == 2
            assert state.input_geometry.input.K == 8
            torch.testing.assert_close(
                state.input_geometry.continuous_centroids.sort(dim=0).values, X[common_rows]
            )
            with pytest.warns(UserWarning, match="holds 2 centroids for the requested K=8"):
                replay = Network.fit(recipe, X, y, initial_state=state)
            torch.testing.assert_close(member.predict(X), replay.predict(X))

    def test_supplied_geometry_is_not_truncated_to_row_capacity(self):
        X = torch.tensor([[0.0], [1.0]], dtype=DTYPE, device=DEVICE)
        recipe = Recipe.chain(Input(K=4), RegressionHead())
        centroids = torch.linspace(0, 1, 4, dtype=DTYPE, device=DEVICE)[:, None]
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter("always")
            state = Network.initialise(recipe, X, X[:, 0], continuous_centroids=centroids)
        assert not observed
        assert state.input_geometry.K_active == 4
        torch.testing.assert_close(state.input_geometry.continuous_centroids, centroids)

    @pytest.mark.parametrize("input_type", [Input, ManifoldInput])
    @pytest.mark.parametrize("width", [0, 4])
    def test_invalid_state_width_still_raises(self, input_type, width):
        X = torch.tensor([[0.0], [1.0]], dtype=DTYPE, device=DEVICE)
        recipe = Recipe.chain(input_type(K=2), RegressionHead())
        state = Network.initialise(recipe, X, X[:, 0])
        invalid = replace(
            state,
            input_geometry=replace(
                state.input_geometry,
                continuous_centroids=torch.zeros(width, 1, dtype=DTYPE, device=DEVICE),
            ),
        )
        with pytest.raises(ValueError, match="centroid count"):
            Network.fit(recipe, X, X[:, 0], initial_state=invalid)

    def test_empty_balanced_initialisation_pool_still_raises(self):
        X = torch.linspace(0, 1, 4, dtype=DTYPE, device=DEVICE)[:, None]
        y = torch.tensor([0, 1, -1, -1], device=DEVICE)
        recipe = Recipe.chain(Input(K=3, balanced=True), ClassificationHead(Coupling.M))
        with pytest.raises(ValueError, match=r"at least one.*eligible row"):
            Network.fit(recipe, X, y, init_rows=torch.tensor([2, 3], device=DEVICE))

    def test_capping_input_does_not_silently_change_hidden_profile_capacity(self):
        X = torch.tensor([[0.0], [1.0]], dtype=DTYPE, device=DEVICE)
        recipe = Recipe.chain(Input(K=8), Hidden(K=4), RegressionHead(), coupling=Coupling.M)
        with (
            pytest.warns(UserWarning, match="reducing"),
            pytest.raises(ValueError, match=r"4.*2.*rows"),
        ):
            Network.fit(recipe, X, X[:, 0])
