"""Immutable name-based Recipe tuning through the estimator."""

from dataclasses import fields
from itertools import product

import numpy as np
import pytest
from conftest import DEVICE, DTYPE, FIXED_BUDGET
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV

from entlearn import ClassificationHead, Connection, Coupling, Hidden, Input, Recipe
from entlearn.scikit_adapter import EONClassifier

from ._fixtures import classifier_recipe, parameter_variants, tabular_data


class TestNestedParameters:
    def test_named_parameter_changes_rebuild_without_changing_the_original(self):
        recipe = classifier_recipe(hidden=True)
        estimator = EONClassifier(recipe)
        path = "recipe__blocks__hidden_b__K"
        assert estimator.get_params()[path] == 2

        assert estimator.set_params(**{path: 4}) is estimator

        assert estimator.get_params()[path] == 4
        assert estimator.recipe == recipe.replace_block("hidden_b", K=4)
        assert estimator.recipe is not recipe
        assert recipe.blocks[1].K == 2

    def test_every_description_field_is_discovered_and_changed(self):
        discovered = set()
        changed = set()
        for old, new in parameter_variants():
            estimator = EONClassifier(old)
            params = estimator.get_params()
            updates = {}
            for group in ("blocks", "connections"):
                for before, after in zip(getattr(old, group), getattr(new, group), strict=True):
                    for field in fields(before):
                        path = f"recipe__{group}__{before.name}__{field.name}"
                        assert params[path] == getattr(before, field.name)
                        updates[path] = getattr(after, field.name)
                        key = (type(before), field.name)
                        discovered.add(key)
                        if updates[path] != params[path]:
                            changed.add(key)
            estimator.set_params(**updates)
            assert estimator.recipe == new
        # The variants change every field, so no field's path goes untested.
        assert changed == discovered

    def test_whole_recipe_precedes_nested_paths_regardless_of_keyword_order(self):
        estimator = EONClassifier(classifier_recipe())
        replacement = classifier_recipe(hidden=True)
        estimator.set_params(recipe__blocks__hidden_b__K=4, recipe=replacement)
        assert estimator.recipe == replacement.replace_block("hidden_b", K=4)

    @pytest.mark.parametrize(
        ("updates", "message"),
        [
            (
                {"recipe__blocks__missing__K": 4},
                r"Recipe parameter\(s\): \['recipe__blocks__missing__K",
            ),
            ({"recipe__blocks__hidden_b__K": 0}, "K must be a positive integer"),
            (
                {"recipe__blocks__hidden_b__epsilon_ratio": 1.5},
                "epsilon_ratio cannot be combined with an explicit epsilon",
            ),
            (
                {
                    "recipe__blocks__hidden_b__epsilon": None,
                    "recipe__blocks__hidden_b__epsilon_ratio": -1.5,
                },
                r"epsilon_ratio.*non-negative",
            ),
            (
                {"recipe__blocks__features__epsilon_ratio": 1.5},
                r"Invalid Recipe parameter\(s\): \['recipe",
            ),
            (
                {"recipe__connections__hidden_b_to_labels__delta_ratio": 3.0},
                "delta_ratio cannot be combined with an explicit delta",
            ),
            (
                {
                    "recipe__connections__features_to_hidden_b__delta": None,
                    "recipe__connections__features_to_hidden_b__delta_ratio": 3.0,
                },
                "delta_ratio requires a previous connection",
            ),
            (
                {"recipe__connections__features_to_hidden_b__source": "missing"},
                "endpoints must name Recipe blocks",
            ),
            ({"unknown": 2}, r"Invalid parameter\(s\) \['unknown'\] for EONClassifier"),
        ],
    )
    def test_invalid_changes_leave_every_parameter_unchanged(self, updates, message):
        estimator = EONClassifier(classifier_recipe(hidden=True), max_iter=10)
        before = estimator.get_params()
        with pytest.raises(ValueError, match=message):
            estimator.set_params(max_iter=20, **updates)
        assert estimator.get_params() == before

    def test_coupled_description_fields_are_replaced_together(self):
        estimator = EONClassifier(classifier_recipe())
        estimator.set_params(
            recipe__blocks__features__greedy_candidates=4,
            recipe__blocks__features__centroid_strategy="greedy-kmeans++",
        )
        assert estimator.recipe.blocks[0].greedy_candidates == 4
        assert estimator.recipe.blocks[0].centroid_strategy == "greedy-kmeans++"

    def test_named_paths_survive_unrelated_dag_insertions(self):
        original = classifier_recipe(hidden=True)
        expanded = Recipe(
            blocks=(Hidden(name="other"), *original.blocks),
            connections=(
                Connection(
                    "features_to_other", "features", "other", coupling=Coupling.M, theta_alpha=1
                ),
                Connection("other_to_labels", "other", "labels"),
                Connection("skip_a_to_b", "features", "labels"),
                *original.connections,
            ),
        )
        estimator = EONClassifier(expanded)
        estimator.set_params(
            recipe__blocks__hidden_b__K=5, recipe__connections__skip_a_to_b__delta=0.4
        )
        assert estimator.recipe == expanded.replace_block("hidden_b", K=5).replace_connection(
            "skip_a_to_b", delta=0.4
        )

    def test_a_grid_search_fits_its_best_candidate_with_the_chosen_nested_K(self):
        X, y = tabular_data()
        # The default K=3 is outside the grid, so an ignored nested path shows.
        search = GridSearchCV(
            EONClassifier(
                classifier_recipe(), random_state=8, max_iter=100, dtype=DTYPE, device=DEVICE
            ),
            {"recipe__blocks__features__K": [4, 5]},
            cv=2,
            scoring="neg_log_loss",
            error_score="raise",
        ).fit(X, y)
        K = search.best_params_["recipe__blocks__features__K"]
        fitted = search.best_estimator_.network_
        assert fitted.recipe.blocks[0].K == K
        assert fitted.initial_state.input_geometry.continuous_centroids.shape[0] == K

    def test_default_couplings_are_exposed_as_named_parameters(self):
        recipe = Recipe.chain(
            Input(K=3, name="features", epsilon=0.05),
            Hidden(K=2, name="hidden_b", epsilon=0.04),
            ClassificationHead(name="labels"),
        )
        params = EONClassifier(recipe).get_params()
        assert params["recipe__connections__features_to_hidden_b__coupling"] == Coupling.M
        assert params["recipe__blocks__labels__coupling"] == Coupling.M
        assert params["recipe__connections__hidden_b_to_labels__coupling"] is None

    def test_a_default_theta_alpha_is_a_named_parameter_that_none_restores(self):
        recipe = Recipe(
            blocks=(
                Input(K=3, name="features", epsilon=0.05),
                Hidden(K=2, name="hidden_b", epsilon=0.04),
                ClassificationHead(name="labels"),
            ),
            connections=(
                Connection("features_to_hidden_b", "features", "hidden_b"),
                Connection("hidden_b_to_labels", "hidden_b", "labels"),
            ),
        )
        estimator = EONClassifier(recipe)
        path = "recipe__connections__features_to_hidden_b__theta_alpha"
        params = estimator.get_params()
        assert params[path] == 1.0
        assert params["recipe__connections__hidden_b_to_labels__theta_alpha"] is None
        estimator.set_params(**{path: 2.0}).set_params(**{path: None})
        assert estimator.get_params()[path] == 1.0

    def test_a_hidden_temperature_switches_forms_and_clones_without_resolving(self):
        estimator = EONClassifier(classifier_recipe(hidden=True))
        epsilon = "recipe__blocks__hidden_b__epsilon"
        ratio = "recipe__blocks__hidden_b__epsilon_ratio"

        estimator.set_params(**{epsilon: None, ratio: 1.5})
        assert estimator.get_params()[epsilon] is None
        assert estimator.get_params()[ratio] == 1.5
        cloned = clone(estimator)
        assert cloned.get_params() == estimator.get_params()

        cloned.set_params(**{epsilon: 0.0, ratio: None})
        assert cloned.get_params()[epsilon] == 0.0
        assert cloned.get_params()[ratio] is None
        assert estimator.get_params()[epsilon] is None

    def test_a_grid_searches_both_ratios_with_the_first_delta(self):
        X, y = tabular_data()
        recipe = Recipe.chain(
            Input(K=4, name="features", epsilon=0.05),
            Hidden(K=3, name="hidden_b", epsilon_ratio=1.5),
            ClassificationHead(name="labels"),
            delta_ratio=3,
        )
        estimator = EONClassifier(recipe, random_state=8, max_iter=20, dtype=DTYPE, device=DEVICE)
        ratio = "recipe__blocks__hidden_b__epsilon_ratio"
        delta = "recipe__connections__features_to_hidden_b__delta"
        delta_ratio = "recipe__connections__hidden_b_to_labels__delta_ratio"
        params = estimator.get_params()
        assert params[ratio] == 1.5
        assert params["recipe__blocks__hidden_b__epsilon"] is None
        assert params[delta_ratio] == 3
        assert params["recipe__connections__hidden_b_to_labels__delta"] is None

        grid = {ratio: [0.5, 1.5], delta: [0.1, 1.0], delta_ratio: [1.0, 3.0]}
        search = GridSearchCV(estimator, grid, cv=2, error_score="raise").fit(X, y)

        assert {(p[ratio], p[delta], p[delta_ratio]) for p in search.cv_results_["params"]} == set(
            product(*grid.values())
        )
        assert np.isfinite(search.cv_results_["mean_test_score"]).all()
        selected = search.best_estimator_
        best = search.best_params_
        explicit = selected.recipe.resolve_ratios()
        assert selected.network_.recipe == selected.recipe
        assert selected.recipe.blocks[1].epsilon_ratio == best[ratio]
        assert selected.recipe.connections[1].delta_ratio == best[delta_ratio]
        assert explicit.blocks[1] == Hidden(K=3, name="hidden_b", epsilon=best[ratio] * best[delta])
        assert explicit.connections[1].delta == best[delta_ratio] * best[delta]
        refitted = clone(selected).set_params(recipe=explicit).fit(X, y)
        np.testing.assert_array_equal(refitted.predict_proba(X), selected.predict_proba(X))

    @FIXED_BUDGET
    def test_a_ratio_estimator_resumes_and_fine_tunes_a_changed_delta(self):
        X, y = tabular_data()
        recipe = Recipe.chain(
            Input(K=4, epsilon=0.05),
            Hidden(K=3, epsilon_ratio=1.5),
            ClassificationHead(),
            delta_ratio=1,
        )
        estimator = EONClassifier(
            recipe, random_state=8, max_iter=3, tol=0, dtype=DTYPE, device=DEVICE
        ).fit(X, y)

        estimator.set_params(warm_start="resume", max_iter=6).fit(X, y)
        assert estimator.network_.recipe == recipe
        assert estimator.n_iter_ == 6

        estimator.set_params(
            warm_start="fine_tune", recipe__connections__input_to_hidden_1__delta=0.2
        ).fit(X, y)
        assert estimator.network_.recipe == recipe.replace_connection(
            "input_to_hidden_1", delta=0.2
        )
        assert estimator.network_.recipe.resolve_ratios().connections[1].delta == 0.2
