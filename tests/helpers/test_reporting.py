"""Reporting preserves fitted parameter and effective-dimension semantics."""

import pytest
import torch
from conftest import DEVICE, DTYPE
from network._fixtures import prediction_model

from entlearn import Coupling, Network, Recipe
from entlearn.helpers.reporting import (
    active_features,
    count_parameters,
    effective_dimensions,
    feature_importances,
    target_weights,
)
from entlearn.primitives import effective_dimension, entropy
from helpers._fixtures import fitted_manifold_report, reporting_chain, reporting_model


class TestParameterCounting:
    def test_default_active_threshold_is_the_uniform_share(self):
        # Scaled weights are 1.4, 0.2, 0.2 and 2.2, so only a threshold of one
        # selects the first feature; any larger default would drop it.
        model = reporting_model(weights=(0.35, 0.05, 0.05, 0.55))
        torch.testing.assert_close(
            active_features(model), torch.tensor([0, 3], dtype=torch.int64, device=DEVICE)
        )
        torch.testing.assert_close(active_features(model), active_features(model, tol=1.0))

    def test_categorical_activity_is_read_at_its_own_feature_position(self):
        # Scaled weights are 0.5, 0.5, 2.5, 0.5 and 1.0: only the third
        # continuous feature is active, and neither categorical feature is.
        model = reporting_model(weights=(0.1, 0.1, 0.5, 0.1, 0.2), categorical_features=2)
        torch.testing.assert_close(
            active_features(model, tol=1), torch.tensor([2], dtype=torch.int64, device=DEVICE)
        )
        # One active continuous centroid entry and two output centroids.
        assert count_parameters(model, active_tol=1) == 3

    @pytest.mark.parametrize("raw", [False, True])
    def test_default_excludes_pinned_weights_and_respects_simplex_constraints(self, raw):
        model = reporting_model()
        # Three continuous entries, a two-category simplex and two output centroids.
        assert count_parameters(model, raw=raw) == (7 if raw else 6)
        # Each of the two rows has a one-cluster affiliation. Pinned Wt is not learned.
        assert count_parameters(model, raw=raw, include_affiliations=True) == (9 if raw else 6)

    @pytest.mark.parametrize(
        ("weights", "indices", "free", "raw"),
        [
            ((0.6, 0.05, 0.3, 0.05), [0, 2], 4, 4),
            ((0.1, 0.1, 0.1, 0.7), [3], 3, 4),
            ((0.25, 0.05, 0.1, 0.6), [3], 3, 4),
        ],
    )
    def test_active_counts_drop_negligible_continuous_and_categorical_dimensions(
        self, weights, indices, free, raw
    ):
        model = reporting_model(weights=weights)
        torch.testing.assert_close(
            active_features(model, tol=1),
            torch.tensor(indices, dtype=torch.int64, device=DEVICE),
        )
        assert count_parameters(model, active_tol=1) == free
        assert count_parameters(model, active_tol=1, raw=True) == raw

    @pytest.mark.parametrize("operation", [active_features, count_parameters, effective_dimensions])
    def test_unfitted_values_are_rejected(self, operation):
        with pytest.raises(ValueError, match="fitted Network"):
            operation(object())

    @pytest.mark.parametrize("tol", [-1, float("nan"), float("inf"), 4, True])
    def test_invalid_active_threshold_is_rejected(self, tol):
        with pytest.raises(ValueError, match="tol"):
            active_features(reporting_model(), tol=tol)

    def test_uniform_weights_and_manifold_have_no_active_feature_ranking(self):
        uniform = reporting_model(weights=(0.25, 0.25, 0.25, 0.25))
        with pytest.raises(ValueError, match="uniform"):
            active_features(uniform, tol=1)
        manifold, _, _ = prediction_model(input_kind="manifold")
        with pytest.raises(ValueError, match="feature_weights"):
            active_features(manifold, tol=1)

    def test_active_learned_weight_subset_has_no_simplex_deduction(self):
        model = reporting_model(learn_features=True)
        torch.testing.assert_close(
            active_features(model, tol=1), torch.tensor([2], dtype=torch.int64, device=DEVICE)
        )
        assert count_parameters(model) == 9
        # One continuous centroid, its learned feature weight, two output centroids.
        assert count_parameters(model, active_tol=1) == 4
        assert count_parameters(model, active_tol=1, raw=True) == 4

    @pytest.mark.parametrize(
        ("epsilon_M", "W_M", "extra"),
        [(float("inf"), None, 0), (float("inf"), (0.2, 0.8), 0), (0.2, None, 1)],
    )
    def test_only_learned_output_weights_count(self, epsilon_M, W_M, extra):
        model = reporting_model(epsilon_M=epsilon_M, W_M=W_M)
        assert count_parameters(model) == 6 + extra
        assert count_parameters(model, raw=True) == 7 + 2 * extra


class TestEffectiveDimensions:
    def test_zero_mass_and_normalisation_remain_tensor_native(self):
        probabilities = torch.tensor(
            [[1.0, 0.0], [0.5, 0.5], [0.0, 0.0]], dtype=DTYPE, device=DEVICE
        )
        torch.testing.assert_close(
            entropy(probabilities),
            torch.tensor([0.0, 0.6931471805599453, 0.0], dtype=DTYPE, device=DEVICE),
        )
        expected = torch.tensor([1.0, 2.0, 1.0], dtype=DTYPE, device=DEVICE)
        torch.testing.assert_close(effective_dimension(probabilities, normalise=False), expected)
        torch.testing.assert_close(effective_dimension(probabilities), expected / 2)

    def test_report_reads_fitted_distributions(self):
        model = reporting_model()
        report = effective_dimensions(model, normalise=False)
        assert set(report) == {model.recipe.blocks[0].name}
        values = report[model.recipe.blocks[0].name]
        assert set(values) == {"affiliations", "feature_weights", "instance_weights"}
        torch.testing.assert_close(
            values["affiliations"], torch.ones((), dtype=DTYPE, device=DEVICE)
        )
        torch.testing.assert_close(
            values["instance_weights"], torch.full((), 2.0, dtype=DTYPE, device=DEVICE)
        )

    def test_normalisation_divides_each_field_by_its_distribution_width(self):
        recipe, X, y, categories, state = reporting_chain(
            task="regression", kind="mixed", coupling=Coupling.M, head_coupling=Coupling.M
        )
        model = Network.fit(recipe, X, y, X_cat=categories, initial_state=state, max_iter=1)
        spreads = effective_dimensions(model, normalise=False)
        shares = effective_dimensions(model)
        assert set(spreads) == {"features", "wide", "narrow", "response"}
        assert set(spreads["features"]) == {"affiliations", "feature_weights", "instance_weights"}
        # A represented output simplex replaces the head's affiliation entry.
        assert set(spreads["response"]) == {"output_weights"}
        widths = {"instance_weights": X.shape[0], "feature_weights": 5, "output_weights": 2}
        for name, fields in spreads.items():
            assert shares[name].keys() == fields.keys()
            for field, spread in fields.items():
                width = (
                    dict(model.schema.K_active)[name] if field == "affiliations" else widths[field]
                )
                assert width > 1
                torch.testing.assert_close(shares[name][field] * width, spread)


class TestReorderedReporting:
    @pytest.mark.parametrize("task", ["classification", "regression"])
    @pytest.mark.parametrize("kind", ["mixed", "manifold"])
    @pytest.mark.parametrize("order", [(3, 2, 1, 0), (2, 0, 3, 1)])
    @pytest.mark.parametrize("operation", ["counts", "dimensions"])
    def test_named_graph_identity_not_declaration_order_controls_reporting(
        self, task, kind, order, operation
    ):
        recipe, X, y, categories, state = reporting_chain(
            task=task, kind=kind, coupling=Coupling.S, head_coupling=Coupling.M
        )
        canonical = Network.fit(recipe, X, y, X_cat=categories, initial_state=state, max_iter=1)
        reordered = Network.fit(
            Recipe(tuple(recipe.blocks[index] for index in order), recipe.connections),
            X,
            y,
            X_cat=categories,
            initial_state=state,
            max_iter=1,
        )
        torch.testing.assert_close(
            reordered.predict(X, X_cat=categories), canonical.predict(X, X_cat=categories)
        )
        if operation == "counts":
            for raw in (False, True):
                for include_affiliations in (False, True):
                    assert count_parameters(
                        reordered, raw=raw, include_affiliations=include_affiliations
                    ) == count_parameters(
                        canonical, raw=raw, include_affiliations=include_affiliations
                    )
        else:
            for normalise in (False, True):
                expected = effective_dimensions(canonical, normalise=normalise)
                actual = effective_dimensions(reordered, normalise=normalise)
                assert actual.keys() == expected.keys()
                for name, fields in expected.items():
                    assert actual[name].keys() == fields.keys()
                    for field, value in fields.items():
                        torch.testing.assert_close(actual[name][field], value)


class TestIndependentParameterCounts:
    @pytest.mark.parametrize("kind", ["mixed", "manifold"])
    @pytest.mark.parametrize(
        ("coupling", "hidden_free"),
        [
            ((Coupling.M, Coupling.M), 11),
            ((Coupling.M, Coupling.S), 12),
            ((Coupling.S, Coupling.M), 12),
            ((Coupling.S, Coupling.S), 13),
        ],
    )
    @pytest.mark.parametrize(
        ("task", "head_coupling", "head_free"),
        [
            ("classification", Coupling.M, 4),
            ("classification", Coupling.S, 3),
            ("regression", Coupling.M, 5),
        ],
    )
    def test_asymmetric_chain_counts_match_independent_formulas(
        self, kind, coupling, hidden_free, task, head_coupling, head_free
    ):
        recipe, X, y, categories, state = reporting_chain(
            task=task, kind=kind, coupling=coupling, head_coupling=head_coupling
        )
        model = Network.fit(recipe, X, y, X_cat=categories, initial_state=state, max_iter=1)
        assert dict(model.schema.K_active) == {
            "features": 4,
            "wide": 3,
            "narrow": 2,
            "response": 3 if task == "classification" else 2,
        }
        # Geometry: 4x3 continuous; mixed adds 4x2 + 4x3 categorical and 5 weights.
        # Manifold stores 4x3x2 basis entries but has 4 * [3 + 2*(3-2)] freedoms.
        geometry_raw, geometry_free = (37, 28) if kind == "mixed" else (36, 20)
        # Transitions: 3x4 and 2x3; M subtracts 4 + 3, S subtracts 3 + 2.
        # Heads: 3x2 classification, or 2x2 regression plus a 2-element weight vector.
        assert count_parameters(model, raw=True) == geometry_raw + 18 + 6
        assert count_parameters(model) == geometry_free + hidden_free + head_free
        # Twelve rows carry three affiliation vectors of widths 4, 3 and 2,
        # plus one learned instance-weight vector: raw 120, free 120 - 36 - 1.
        assert count_parameters(model, raw=True, include_affiliations=True) == (
            geometry_raw + 18 + 6 + 120
        )
        assert count_parameters(model, include_affiliations=True) == (
            geometry_free + hidden_free + head_free + 83
        )


class TestInferenceMode:
    """Every reporting helper returns inference tensors, whichever path computed them."""

    def test_active_features_result_is_an_inference_tensor(self):
        assert active_features(reporting_model()).is_inference()

    def test_effective_dimensions_results_are_inference_tensors(self):
        report = effective_dimensions(reporting_model())
        for fields in report.values():
            for value in fields.values():
                assert value.is_inference()

    def test_target_weights_result_is_an_inference_tensor_pinned_and_uniform(self):
        assert target_weights(reporting_model(W_M=(0.2, 0.8))).is_inference()
        # No W_M and infinite epsilon_M falls back to a freshly built uniform simplex.
        assert target_weights(reporting_model()).is_inference()

    def test_feature_importances_result_is_an_inference_tensor_standard_and_manifold(self):
        assert feature_importances(reporting_model()).is_inference()
        assert feature_importances(fitted_manifold_report()).is_inference()
