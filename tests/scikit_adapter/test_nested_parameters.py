"""Immutable name-based Recipe tuning through the estimator."""

from dataclasses import fields

import numpy as np
import pytest
from conftest import DEVICE, DTYPE
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV

from entlearn import ClassificationHead, Connection, Coupling, Hidden, Input, Recipe
from entlearn.scikit_adapter import EONClassifier

from ._fixtures import classifier_recipe, parameter_variants, tabular_data


class TestNestedParameters:
    def test_named_parameter_changes_rebuild_without_changing_the_original(self):
        from entlearn.scikit_adapter import EONClassifier

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
            assert estimator.get_params(deep=False)["recipe"] == new
        assert changed == discovered

    def test_whole_recipe_precedes_nested_paths_regardless_of_keyword_order(self):
        estimator = EONClassifier(classifier_recipe())
        replacement = classifier_recipe(hidden=True)
        estimator.set_params(recipe__blocks__hidden_b__K=4, recipe=replacement)
        assert estimator.recipe == replacement.replace_block("hidden_b", K=4)

    @pytest.mark.parametrize(
        ("updates", "message"),
        [
            ({"recipe__blocks__missing__K": 4}, r"Invalid Recipe parameter\(s\): \['recipe"),
            ({"recipe__blocks__hidden_b__width": 4}, r"Invalid Recipe parameter\(s\): \['recipe"),
            (
                {"recipe__blocks__hidden_b__K__value": 4},
                r"Invalid Recipe parameter\(s\): \['recipe",
            ),
            ({"recipe__blocks__hidden_b__K": 0}, "K must be a positive integer"),
            (
                {"recipe__connections__features_to_hidden_b__source": "missing"},
                "connection endpoints must name Recipe blocks",
            ),
            (
                {"predict_config__epsilon_P": 0.4},
                r"Invalid Recipe parameter\(s\): \['predict_config",
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

    def test_clone_is_parameter_only_and_grid_search_tunes_a_chain(self):
        X, y = tabular_data()
        estimator = EONClassifier(
            classifier_recipe(), random_state=8, max_iter=100, dtype=DTYPE, device=DEVICE
        ).fit(X, y)
        cloned = clone(estimator)
        assert cloned.get_params() == estimator.get_params()
        assert not hasattr(cloned, "network_")
        search = GridSearchCV(
            cloned,
            {"recipe__blocks__features__K": [3, 4]},
            cv=2,
            scoring="neg_log_loss",
            error_score="raise",
        ).fit(X, y)
        assert search.best_estimator_.score(X, y) > 0.9
        assert np.isfinite(search.cv_results_["mean_test_score"]).all()
        assert search.best_estimator_.recipe.blocks[0].K in (3, 4)

    def test_default_couplings_are_named_parameters_a_grid_can_tune(self):
        X, y = tabular_data()
        recipe = Recipe.chain(
            Input(K=3, name="features", epsilon=0.05),
            Hidden(K=2, name="hidden_b", epsilon=0.04),
            ClassificationHead(name="labels"),
        )
        estimator = EONClassifier(recipe, random_state=8, max_iter=20, dtype=DTYPE, device=DEVICE)
        connection = "recipe__connections__features_to_hidden_b__coupling"
        head = "recipe__blocks__labels__coupling"
        params = estimator.get_params()
        assert params[connection] is Coupling.M
        assert params[head] is Coupling.M
        assert params["recipe__connections__hidden_b_to_labels__coupling"] is None

        grid = {connection: [Coupling.M, Coupling.S], head: [Coupling.M, Coupling.S]}
        search = GridSearchCV(estimator, grid, cv=2, error_score="raise").fit(X, y)

        assert {(p[connection], p[head]) for p in search.cv_results_["params"]} == {
            (Coupling.M, Coupling.M),
            (Coupling.M, Coupling.S),
            (Coupling.S, Coupling.M),
            (Coupling.S, Coupling.S),
        }
        assert np.isfinite(search.cv_results_["mean_test_score"]).all()

    def test_a_default_theta_alpha_is_a_named_parameter_a_grid_can_tune(self):
        X, y = tabular_data()
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
        estimator = EONClassifier(recipe, random_state=8, max_iter=20, dtype=DTYPE, device=DEVICE)
        path = "recipe__connections__features_to_hidden_b__theta_alpha"
        params = estimator.get_params()
        assert params[path] == 1.0
        assert params["recipe__connections__hidden_b_to_labels__theta_alpha"] is None

        search = GridSearchCV(estimator, {path: [1.0, 2.0]}, cv=2, error_score="raise").fit(X, y)

        assert [p[path] for p in search.cv_results_["params"]] == [1.0, 2.0]
        assert search.best_estimator_.recipe.connections[0].theta_alpha in (1.0, 2.0)
        assert np.isfinite(search.cv_results_["mean_test_score"]).all()
        estimator.set_params(**{path: 2.0}).set_params(**{path: None})
        assert estimator.get_params()[path] == 1.0
