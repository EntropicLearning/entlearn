"""Fresh fits from compatible original input geometry."""

import warnings
from dataclasses import replace

import pytest
import torch
from conftest import DEVICE, DTYPE
from network._fixtures import resume_case, squared_error, validation_pairs

from entlearn import (
    ClassificationHead,
    Coupling,
    Hidden,
    InitialState,
    Input,
    ManifoldInput,
    Network,
    PredictConfig,
    Recipe,
)


def provided_messages(caught):
    """Return the caught warnings about a provided state, in order."""
    return [str(item.message) for item in caught if str(item.message).startswith("InitialState")]


def small_problem():
    """Return eight ordered rows in two classes and a one-input classification Recipe."""
    X = torch.linspace(0, 1, 16, dtype=DTYPE, device=DEVICE).reshape(8, 2)
    y = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], device=DEVICE)
    return X, y, Recipe.chain(Input(K=3), ClassificationHead(Coupling.M))


class TestInitialStateReuse:
    @pytest.mark.parametrize(
        ("change", "message"),
        [({"W_std": 0.2}, r"W_std.*0\.0.*0\.2"), ({"K": 6}, r"K=6.*4 centroids")],
        ids=["initialisation-control", "wider"],
    )
    def test_incompatible_request_warns_with_the_reason_and_regenerates(self, change, message):
        X = torch.linspace(0, 1, 12, dtype=DTYPE, device=DEVICE).reshape(6, 2)
        y = torch.tensor([0, 0, 0, 1, 1, 1], device=DEVICE)
        recipe = Recipe.chain(Input(K=4), ClassificationHead(Coupling.M))
        state = Network.initialise(recipe, X, y)
        changed = recipe.replace_block("input", **change)
        with pytest.warns(UserWarning, match=message):
            fitted = Network.fit(changed, X, y, initial_state=state, max_iter=1)
        assert fitted.initial_state.input_geometry.input == changed.blocks[0]

    def test_geometry_of_another_input_kind_warns_and_regenerates(self):
        X, y, recipe = small_problem()
        state = Network.initialise(recipe, X, y)
        manifold = Recipe.chain(ManifoldInput(K=3, subspace_dimension=1), recipe.blocks[-1])
        with pytest.warns(UserWarning, match="reuse requires the same input kind"):
            fitted = Network.fit(manifold, X, y, initial_state=state, max_iter=1)
        assert fitted.initial_state.input_geometry.input == manifold.blocks[0]

    @pytest.mark.parametrize("kind", ["standard", "manifold"])
    def test_captured_geometry_trimmed_below_its_recorded_width_raises(self, kind):
        recipe, X, y, _ = resume_case(kind=kind)
        source = Network.fit(recipe, X, y, max_iter=1)
        geometry = source.capture_current_state(blocks=("input",)).input_geometry
        keep = geometry.K_active - 1
        projectors = geometry.manifold_projectors
        trimmed = replace(
            geometry,
            continuous_centroids=geometry.continuous_centroids[:keep],
            categorical_centroids=tuple(c[:keep] for c in geometry.categorical_centroids),
            manifold_projectors=None if projectors is None else projectors[:keep],
        )
        target = replace(recipe, blocks=(geometry.input, *recipe.blocks[1:]))
        with pytest.raises(ValueError, match="centroid count"):
            Network.fit(target, X, y, initial_state=InitialState(input_geometry=trimmed))

    @pytest.mark.parametrize(
        ("damage", "message"),
        [
            (lambda g: {"input": Hidden(K=3)}, "invalid input geometry kind"),
            (lambda g: {"captured": 0}, "captured must be a bool"),
            (lambda g: {"categorical_centroids": list(g.categorical_centroids)}, "must be a tuple"),
            (
                lambda g: {
                    "manifold_projectors": torch.ones_like(g.continuous_centroids)[..., None]
                },
                "standard input cannot have manifold_projectors",
            ),
            (
                lambda g: {"categorical_centroids": (g.categorical_centroids[0][:-1],)},
                "invalid categorical centroid shape",
            ),
            (
                lambda g: {"categorical_centroids": (g.categorical_centroids[0][:, :1],)},
                "invalid categorical centroid shape",
            ),
            (
                lambda g: {"feature_weights": g.feature_weights[1:] / g.feature_weights[1:].sum()},
                "feature_weights has invalid shape",
            ),
        ],
        ids=[
            "kind",
            "captured",
            "categorical-list",
            "projectors",
            "categorical-rows",
            "levels",
            "feature-weight-count",
        ],
    )
    def test_malformed_standard_geometry_raises(self, damage, message):
        X, y, recipe = small_problem()
        X_cat = [y.flip(0)]
        geometry = Network.initialise(recipe, X, y, X_cat=X_cat).input_geometry
        state = InitialState(input_geometry=replace(geometry, **damage(geometry)))
        with pytest.raises(ValueError, match=message):
            Network.fit(recipe, X, y, X_cat=X_cat, initial_state=state, max_iter=1)

    @pytest.mark.parametrize(
        ("first", "change"),
        [
            (Input(K=3), {"epsilon_D": 0.5}),
            (Input(K=3), {"epsilon_T": 0.5}),
            (Input(K=3), {"delta_cat": 0.5}),
            (ManifoldInput(K=3, subspace_dimension=1), {"alpha": 0.5}),
        ],
        ids=["epsilon_D", "epsilon_T", "delta_cat", "alpha"],
    )
    @pytest.mark.filterwarnings("ignore::entlearn.ConvergenceWarning")
    def test_generated_geometry_is_reused_across_a_fitting_setting(self, first, change):
        X, y, _ = small_problem()
        recipe = Recipe.chain(first, ClassificationHead(Coupling.M))
        state = Network.initialise(recipe, X, y)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fitted = Network.fit(
                recipe.replace_block("input", **change), X, y, initial_state=state, max_iter=1
            )
        assert provided_messages(caught) == []
        assert fitted.initial_state.input_geometry.input == recipe.blocks[0]

    @pytest.mark.parametrize(
        ("controls", "change"),
        [
            ({}, {"epsilon": 0.2}),
            ({}, {"centroid_strategy": "greedy-kmeans++"}),
            ({"centroid_strategy": "greedy-kmeans++"}, {"greedy_candidates": 3}),
            ({}, {"balanced": True}),
            ({}, {"W_std": 0.2}),
        ],
        ids=["epsilon", "centroid_strategy", "greedy_candidates", "balanced", "W_std"],
    )
    @pytest.mark.filterwarnings("ignore::entlearn.ConvergenceWarning")
    def test_captured_geometry_is_reused_across_an_initialisation_setting(self, controls, change):
        X, y, _ = small_problem()
        recipe = Recipe.chain(Input(K=2, **controls), ClassificationHead(Coupling.M))
        state = Network.fit(recipe, X, y, max_iter=1).capture_current_state(blocks=("input",))
        assert state.input_geometry.K_active == 2
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fitted = Network.fit(
                recipe.replace_block("input", **change), X, y, initial_state=state, max_iter=1
            )
        assert provided_messages(caught) == []
        assert fitted.initial_state.input_geometry.input == recipe.blocks[0]

    @pytest.mark.parametrize(
        "controls", [{}, {"centroid_strategy": "greedy-kmeans++", "greedy_candidates": 3}]
    )
    def test_generated_standard_geometry_has_independent_width_prefixes(self, controls):
        X = torch.rand(30, 3, generator=torch.Generator().manual_seed(31), dtype=DTYPE).to(DEVICE)
        y = (X[:, 0] > 0.5).long()
        recipe = Recipe.chain(Input(K=7, W_std=0.2, **controls), ClassificationHead(Coupling.M))
        weights = torch.linspace(0.1, 2.0, len(X), dtype=DTYPE, device=DEVICE)
        wide = Network.initialise(recipe, X, y, sample_weights=weights, seed=13)
        narrow = Network.initialise(
            recipe.replace_block("input", K=3), X, y, sample_weights=weights, seed=13
        )
        torch.testing.assert_close(
            narrow.input_geometry.continuous_centroids,
            wide.input_geometry.continuous_centroids[:3],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            narrow.input_geometry.feature_weights,
            wide.input_geometry.feature_weights,
            rtol=0,
            atol=0,
        )

    def test_initialise_subset_keeps_full_categorical_vocabulary_and_caps_width(self):
        X = torch.arange(6, dtype=DTYPE, device=DEVICE).reshape(-1, 1)
        y = torch.tensor([0, 1, 0, 1, 0, 1], device=DEVICE)
        codes = torch.tensor([0, 1, 0, 1, 2, 2], device=DEVICE)
        recipe = Recipe.chain(Input(K=5), ClassificationHead(coupling=Coupling.M))
        with pytest.warns(UserWarning, match="K"):
            state = Network.initialise(
                recipe,
                X,
                y,
                X_cat=(codes,),
                init_rows=torch.tensor([0, 1, 2], device=DEVICE),
            )
        assert state.input_geometry.K_active == 3
        assert state.input_geometry.categorical_centroids[0].shape[1] == 3
        assert set(state.input_geometry.continuous_centroids[:, 0].tolist()) == {0, 1, 2}
        with pytest.warns(UserWarning, match="holds 3 centroids for the requested K=5"):
            model = Network.fit(recipe, X, y, X_cat=(codes,), initial_state=state, max_iter=2)
        assert model.predict(X, X_cat=(codes,)).shape == (6, 2)

    def test_prefix_and_changed_objective_match_explicit_geometry(self):
        X = torch.rand(30, 2, generator=torch.Generator().manual_seed(13), dtype=DTYPE).to(DEVICE)
        y = (X[:, 0] > 0.5).long()
        recipe = Recipe.chain(Input(K=5, epsilon=0.1), ClassificationHead(coupling=Coupling.M))
        original = Network.initialise(recipe, X, y, seed=11)
        snapshot = original.input_geometry.continuous_centroids.clone()
        changed = replace(
            recipe, blocks=(replace(recipe.blocks[0], K=3, epsilon=0.2), *recipe.blocks[1:])
        )
        explicit = replace(
            original,
            input_geometry=replace(
                original.input_geometry,
                input=changed.blocks[0],
                continuous_centroids=original.input_geometry.continuous_centroids[:3].clone(),
            ),
        )
        kwargs = dict(max_iter=3, predict_config=PredictConfig(epsilon_P=1.0))
        fitted = Network.fit(changed, X, y, initial_state=original, **kwargs)
        replay = Network.fit(changed, X, y, initial_state=explicit, **kwargs)
        torch.testing.assert_close(fitted.predict(X), replay.predict(X))
        assert fitted.recipe == changed
        assert fitted.initial_state.input_geometry.input == original.input_geometry.input
        torch.testing.assert_close(
            fitted.initial_state.input_geometry.continuous_centroids, snapshot
        )
        torch.testing.assert_close(original.input_geometry.continuous_centroids, snapshot)

    @pytest.mark.parametrize(
        "controls",
        [
            {"balanced": True},
            {"centroid_strategy": "greedy-kmeans++"},
        ],
    )
    def test_non_prefix_strategy_warns_and_falls_back(self, controls):
        X = torch.linspace(0, 1, 20, dtype=DTYPE, device=DEVICE).reshape(10, 2)
        y = torch.tensor([0] * 5 + [1] * 5, device=DEVICE)
        recipe = Recipe.chain(Input(K=4, **controls), ClassificationHead(coupling=Coupling.M))
        state = Network.initialise(recipe, X, y)
        changed = replace(recipe, blocks=(replace(recipe.blocks[0], K=3), *recipe.blocks[1:]))
        with pytest.warns(UserWarning, match=r"incompatible.*first 3.*centroids"):
            fitted = Network.fit(changed, X, y, initial_state=state)
        assert fitted.initial_state.input_geometry.K_active == 3


@pytest.mark.filterwarnings("ignore::entlearn.ConvergenceWarning")
class TestUnusedProvidedState:
    def test_a_renamed_input_block_warns_and_regenerates(self):
        X, y, recipe = small_problem()
        old = Recipe.chain(replace(recipe.blocks[0], name="old"), recipe.blocks[1])
        state = Network.initialise(old, X, y)
        with pytest.warns(UserWarning, match="'old' matches no Recipe block") as caught:
            fitted = Network.fit(recipe, X, y, initial_state=state, max_iter=1)
        unmatched = "InitialState block 'old' matches no Recipe block and is ignored; "
        assert provided_messages(caught)[0] == unmatched + "initialising normally"
        assert provided_messages(caught)[0] in fitted.diagnostics.warnings
        assert fitted.initial_state.input_geometry.input == recipe.blocks[0]

    @pytest.mark.parametrize("seeds", [True, False], ids=("with-seeds", "sole-group"))
    def test_a_group_for_a_block_the_recipe_lacks_warns_once_per_part(self, seeds):
        recipe, X, y, _ = resume_case()
        state = Network.fit(recipe, X, y, max_iter=1).capture_current_state(blocks=("hidden_1",))
        if not seeds:
            state = replace(state, connection_sub_seeds=())
        renamed = Recipe.chain(
            recipe.blocks[0],
            replace(recipe.blocks[1], name="renamed"),
            recipe.blocks[2],
            coupling=Coupling.M,
        )
        with pytest.warns(UserWarning, match="'hidden_1' matches no Recipe block") as caught:
            fitted = Network.fit(renamed, X, y, initial_state=state, max_iter=1)
        connections = ("input_to_hidden_1", "hidden_1_to_output") if seeds else ()
        assert provided_messages(caught) == [
            (
                "InitialState block 'hidden_1' matches no Recipe block and is ignored; "
                "initialising normally"
            ),
            *(
                f"InitialState connection seed {name!r} matches no Recipe connection and is "
                "ignored; initialising normally"
                for name in connections
            ),
        ]
        assert fitted.initial_state.parameters == ()

    @pytest.mark.parametrize("geometry", [True, False], ids=("with-geometry", "sole-seed"))
    def test_a_connection_seed_the_recipe_lacks_warns_and_is_ignored(self, geometry):
        X, y, recipe = small_problem()
        state = Network.initialise(recipe, X, y) if geometry else InitialState()
        extra = (*state.connection_sub_seeds, ("input_to_gone", 5))
        with pytest.warns(
            UserWarning, match="'input_to_gone' matches no Recipe connection"
        ) as caught:
            fitted = Network.fit(
                recipe, X, y, initial_state=replace(state, connection_sub_seeds=extra), max_iter=1
            )
        message = (
            "InitialState connection seed 'input_to_gone' matches no Recipe connection and is "
            "ignored; initialising normally"
        )
        assert provided_messages(caught) == [message]
        assert message in fitted.diagnostics.warnings
        assert [name for name, _ in fitted.initial_state.connection_sub_seeds] == [
            "input_to_output"
        ]

    def test_an_incompatible_sole_group_warns_only_its_rejection(self):
        X, y, recipe = small_problem()
        lone = InitialState(input_geometry=Network.initialise(recipe, X, y).input_geometry)
        wider = recipe.replace_block("input", K=4)
        with pytest.warns(UserWarning, match="is incompatible") as caught:
            Network.fit(wider, X, y, initial_state=lone, max_iter=1)
        assert provided_messages(caught) == [
            (
                "InitialState block 'input' is incompatible (requested K=4 exceeds the original "
                "request for 3 centroids; generate a new InitialState for a larger model); "
                "initialising normally"
            )
        ]

    def test_supplied_geometry_with_fewer_centroids_than_K_warns_with_both_widths(self):
        X, y, recipe = small_problem()
        rows = torch.tensor([0, 4], device=DEVICE)
        with pytest.warns(UserWarning, match="reducing the initial input cluster count to 2"):
            state = Network.initialise(recipe, X, y, init_rows=rows)
        with pytest.warns(UserWarning, match="holds 2 centroids") as caught:
            fitted = Network.fit(recipe, X, y, initial_state=state, max_iter=1)
        assert provided_messages(caught) == [
            (
                "InitialState block 'input' holds 2 centroids for the requested K=3; "
                "fitting 2 input clusters"
            )
        ]
        assert dict(fitted.schema.K_active)["input"] <= 2

    def test_geometry_capped_inside_the_fit_warns_once_per_candidate(self):
        X, y, recipe = small_problem()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Network.fit(
                recipe,
                X,
                y,
                n_inits=2,
                init_rows=torch.tensor([0, 4], device=DEVICE),
                validation_pairs=validation_pairs(),
                selection_loss=squared_error,
                max_iter=1,
            )
        capped = [item for item in caught if "reducing the initial" in str(item.message)]
        assert len(capped) == 2
        assert provided_messages(caught) == []

    def test_an_empty_state_warns_once_that_it_is_not_used(self):
        X, y, recipe = small_problem()
        with pytest.warns(UserWarning, match="holds no group") as caught:
            Network.fit(recipe, X, y, initial_state=InitialState(), max_iter=1)
        assert provided_messages(caught) == [
            "InitialState holds no group or connection seed of this Recipe; initialising normally"
        ]

    def test_a_state_of_used_connection_seeds_alone_raises_no_warning(self):
        recipe, X, y, _ = resume_case()
        state = Network.fit(recipe, X, y, max_iter=1).capture_current_state(blocks=())
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fitted = Network.fit(recipe, X, y, initial_state=state, max_iter=1)
        assert provided_messages(caught) == []
        assert fitted.initial_state.connection_sub_seeds == state.connection_sub_seeds

    def test_a_fully_matching_captured_state_raises_no_warning(self):
        recipe, X, y, _ = resume_case()
        state = Network.fit(recipe, X, y, max_iter=1).capture_current_state()
        assert state.input_geometry.K_active == recipe.blocks[0].K
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fitted = Network.fit(recipe, X, y, initial_state=state, max_iter=1)
        assert provided_messages(caught) == []
        assert fitted.initial_state.block_names == ("input", "hidden_1", "output")

    def test_an_unmatched_input_is_listed_before_unmatched_groups(self):
        recipe, X, y, _ = resume_case()
        state = Network.fit(recipe, X, y, max_iter=1).capture_current_state()
        renamed = Recipe.chain(
            replace(recipe.blocks[0], name="renamed_input"),
            replace(recipe.blocks[1], name="renamed_hidden"),
            recipe.blocks[2],
            coupling=Coupling.M,
        )
        with pytest.warns(UserWarning, match="matches no Recipe block"):
            fitted = Network.fit(renamed, X, y, initial_state=state, max_iter=1)
        assert [m for m in fitted.diagnostics.warnings if "matches no Recipe block" in m] == [
            f"InitialState block {name!r} matches no Recipe block and is ignored; "
            "initialising normally"
            for name in ("input", "hidden_1")
        ]

    @pytest.mark.parametrize("seeds", [True, False], ids=("with-seeds", "without-seeds"))
    def test_a_used_downstream_only_state_raises_no_warning(self, seeds):
        recipe, X, y, _ = resume_case("regression")
        state = Network.fit(recipe, X, y, max_iter=2).capture_current_state(blocks=("output",))
        if not seeds:
            state = replace(state, connection_sub_seeds=())
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fitted = Network.fit(recipe, X, y, initial_state=state, max_iter=2)
        assert provided_messages(caught) == []
        assert fitted.initial_state.block_names == ("input", "output")
