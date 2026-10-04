import pickle
from dataclasses import replace

import pytest
import torch
from network._fixtures import resume_case

from entlearn import Coupling, Hidden, Network, Recipe


class TestCapture:
    @pytest.mark.parametrize("kind", ["standard", "categorical", "manifold"])
    def test_capture_is_detached_active_input_geometry(self, kind):
        recipe, X, y, cats = resume_case(kind=kind)
        source = Network.fit(recipe, X, y, X_cat=cats, max_iter=3, seed=5)
        state = source.capture_current_state(blocks=("input",))
        name = recipe.blocks[0].name
        assert state.input_geometry.captured and not state.input_geometry.prefix_reusable
        assert state.input_geometry.input == replace(
            recipe.blocks[0], K=dict(source.schema.K_active)[name]
        )
        assert state.connection_sub_seeds == source.initial_state.connection_sub_seeds
        torch.testing.assert_close(
            state.input_geometry.continuous_centroids,
            source.inspect("continuous_centroids")[name],
            rtol=0,
            atol=0,
        )
        if kind == "manifold":
            torch.testing.assert_close(
                state.input_geometry.manifold_projectors,
                source.inspect("manifold_projectors")[name],
                rtol=0,
                atol=0,
            )
            assert state.input_geometry.feature_weights is None
        else:
            torch.testing.assert_close(
                state.input_geometry.feature_weights,
                source.inspect("feature_weights")[name],
                rtol=0,
                atol=0,
            )
            if cats:
                for captured, fitted in zip(
                    state.input_geometry.categorical_centroids,
                    source.inspect("categorical_centroids")[name],
                    strict=True,
                ):
                    torch.testing.assert_close(captured, fitted, rtol=0, atol=0)
        before = source.predict(X, X_cat=cats)
        with torch.inference_mode():
            state.input_geometry.continuous_centroids.add_(10)
        torch.testing.assert_close(source.predict(X, X_cat=cats), before, rtol=0, atol=0)
        assert source.capture_current_state() is not source.initial_state

    @pytest.mark.parametrize("kind", ["standard", "manifold"])
    @pytest.mark.parametrize("offset", [-1, 1])
    def test_incompatible_captured_width_initialises_normally(self, kind, offset):
        recipe, X, y, _ = resume_case(kind=kind)
        source = Network.fit(recipe, X, y, max_iter=1)
        state = source.capture_current_state(blocks=("input",))
        new = replace(
            recipe,
            blocks=(
                replace(state.input_geometry.input, K=state.input_geometry.K_active + offset),
                *recipe.blocks[1:],
            ),
        )
        with pytest.warns(UserWarning, match="incompatible"):
            result = Network.fit(new, X, y, initial_state=state, max_iter=1)
        assert not result.initial_state.input_geometry.captured
        assert result.initial_state.input_geometry.K_active == new.blocks[0].K


class TestDeepen:
    def test_transported_capture_reuses_new_names_in_a_second_growth(self):
        recipe, X, y, _ = resume_case("regression")
        source = Network.fit(recipe, X, y, max_iter=2, seed=9)
        captured = source.capture_current_state(blocks=("input",))
        first_recipe = Recipe.chain(
            captured.input_geometry.input,
            Hidden(name="first", K=2),
            recipe.blocks[-1],
            coupling=Coupling.M,
        )
        first = Network.fit(first_recipe, X, y, initial_state=captured, seed=17, max_iter=2)
        first = pickle.loads(pickle.dumps(first))
        second_state = first.capture_current_state()
        second_recipe = Recipe.chain(
            second_state.input_geometry.input,
            first_recipe.blocks[1],
            Hidden(name="second", K=2),
            first_recipe.blocks[-1],
            coupling=Coupling.M,
        )
        second = Network.fit(second_recipe, X, y, initial_state=second_state, seed=29, max_iter=2)
        recorded = dict(second_state.connection_sub_seeds)
        assert "input_to_first" in recorded
        assert (
            dict(second.capture_current_state().connection_sub_seeds)["input_to_first"]
            == recorded["input_to_first"]
        )
        assert dict(second.initial_state.connection_sub_seeds) == dict(
            second.capture_current_state().connection_sub_seeds
        )
        assert second.can_resume

    @pytest.mark.parametrize("task", ["classification", "regression"])
    @pytest.mark.parametrize("kind", ["standard", "categorical", "manifold"])
    def test_deeper_fit_reuses_geometry_and_named_seeds(self, task, kind):
        recipe, X, y, cats = resume_case(task, kind)
        source = Network.fit(recipe, X, y, X_cat=cats, max_iter=3, seed=5)
        state = source.capture_current_state(blocks=("input",))
        deeper = Recipe.chain(
            state.input_geometry.input,
            recipe.blocks[1],
            Hidden(name="deeper", K=2),
            recipe.blocks[-1],
            coupling=Coupling.M,
        )
        result = Network.fit(deeper, X, y, X_cat=cats, initial_state=state, max_iter=2, seed=7)
        repeat = Network.fit(deeper, X, y, X_cat=cats, initial_state=state, max_iter=2, seed=7)
        other = Network.fit(deeper, X, y, X_cat=cats, initial_state=state, max_iter=2, seed=8)
        assert result.can_resume
        torch.testing.assert_close(
            result.initial_state.input_geometry.continuous_centroids,
            state.input_geometry.continuous_centroids,
            rtol=0,
            atol=0,
        )
        seeds = dict(result.capture_current_state().connection_sub_seeds)
        recorded = dict(state.connection_sub_seeds)
        assert set(seeds) == {connection.name for connection in deeper.connections}
        assert seeds == dict(repeat.capture_current_state().connection_sub_seeds)
        changed = dict(other.capture_current_state().connection_sub_seeds)
        for name, seed in seeds.items():
            if name in recorded:
                assert seed == recorded[name] == changed[name]
            else:
                assert seed != changed[name]
        torch.testing.assert_close(
            result.predict(X, X_cat=cats), repeat.predict(X, X_cat=cats), rtol=0, atol=0
        )

    def test_growth_from_pruned_input_uses_active_not_requested_width(self):
        recipe, X, y, _ = resume_case()
        recipe = replace(
            recipe, blocks=(replace(recipe.blocks[0], epsilon=0, K=8), *recipe.blocks[1:])
        )
        source = Network.fit(recipe, X, y, max_iter=3)
        state = source.capture_current_state(blocks=("input",))
        assert state.input_geometry.K_active < recipe.blocks[0].K
        deeper = Recipe.chain(
            state.input_geometry.input, Hidden(K=2), recipe.blocks[-1], coupling=Coupling.M
        )
        result = Network.fit(deeper, X, y, initial_state=state, max_iter=2)
        assert result.initial_state.input_geometry.K_active == state.input_geometry.K_active
        assert result.can_resume
