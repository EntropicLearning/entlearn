"""The public import surface of the package."""

import entlearn

# The lifecycle objects, then the vocabulary a Recipe is written in. Extend this list
# deliberately: an export that appears without a line here is an accident.
EXPECTED = {
    # lifecycle
    "Network",
    "DataSchema",
    "FitDiagnostics",
    "InitialState",
    "InputGeometry",
    "LossIncreaseWarning",
    "ConvergenceWarning",
    "PredictConfig",
    "PredictionResult",
    "ReconstructionResult",
    # Recipe and its description vocabulary
    "Recipe",
    "Connection",
    "Coupling",
    "Input",
    "ManifoldInput",
    "Hidden",
    "ClassificationHead",
    "RegressionHead",
    # role aliases shared by typed reporting and Recipe descriptions
    "InputBlock",
    "Head",
}


class TestPublicSurface:
    def test_all_lists_exactly_the_intended_names(self):
        assert set(entlearn.__all__) == EXPECTED

    def test_all_is_sorted_and_free_of_duplicates(self):
        assert entlearn.__all__ == sorted(set(entlearn.__all__))

    def test_every_exported_name_resolves(self):
        for name in entlearn.__all__:
            assert getattr(entlearn, name) is not None

    def test_role_aliases_keep_the_precise_description_types(self):
        assert entlearn.InputBlock == entlearn.Input | entlearn.ManifoldInput
        assert entlearn.Head == entlearn.ClassificationHead | entlearn.RegressionHead

    def test_helper_namespace_lists_exactly_the_intended_names(self):
        from entlearn import helpers

        expected = {"feature_weights", "reporting"}
        assert set(helpers.__all__) == expected
        assert helpers.__all__ == sorted(expected)
        for name in helpers.__all__:
            assert getattr(helpers, name) is not None

    def test_no_private_name_leaks_into_the_namespace(self):
        public = {name for name in vars(entlearn) if not name.startswith("_")}
        assert (
            public - {"helpers", "network", "recipe", "primitives", "scikit_adapter", "plotting"}
            == EXPECTED
        )

    def test_plotting_namespace_exports_only_figure_functions(self):
        from entlearn import plotting

        expected = {
            "plot_affiliations",
            "plot_centroids",
            "plot_decision",
            "plot_feature_importance",
            "plot_loss",
            "plot_manifold",
            "plot_parallel",
            "plot_theta",
        }
        assert set(plotting.__all__) == expected
        assert plotting.__all__ == sorted(expected)
        for name in plotting.__all__:
            assert getattr(plotting, name) is not None

    def test_scikit_adapter_exports_estimators_queries_and_search_helpers(self):
        from entlearn import scikit_adapter

        assert set(scikit_adapter.__all__) == {
            "EONClassifier",
            "EONRegressor",
            "FeatureLayout",
            "TabularReconstruction",
            "common_train_rows",
        }
        assert scikit_adapter.EONClassifier is not None
        assert scikit_adapter.EONRegressor is not None
        assert scikit_adapter.TabularReconstruction is not None
