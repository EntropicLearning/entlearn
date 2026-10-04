"""Public prediction and requested details."""

import pytest
import torch
from conftest import DEVICE, DTYPE
from network._fixtures import (
    blobs,
    classification_recipe,
    deep_classification_recipe,
    prediction_model,
)

from entlearn import Input, Network, PredictConfig, PredictionResult, ReconstructionResult


class TestResultRecords:
    @pytest.mark.parametrize(
        "build",
        [
            lambda: PredictionResult(prediction=torch.ones(3, dtype=DTYPE, device=DEVICE)),
            lambda: ReconstructionResult(
                continuous=torch.ones(3, 2, dtype=DTYPE, device=DEVICE),
                categorical=(torch.full((3, 2), 0.5, dtype=DTYPE, device=DEVICE),),
            ),
        ],
        ids=["prediction", "reconstruction"],
    )
    def test_records_compare_and_hash_by_identity(self, build):
        record, copy = build(), build()
        assert record == record  # noqa: PLR0124
        assert record != copy
        assert len({record, copy}) == 2


class TestPredictionDetails:
    @pytest.mark.parametrize("mode", ["single", "iterative"])
    def test_diagnostics_are_requested_together(self, mode):
        model, X, _ = prediction_model()
        config = PredictConfig(predict_mode=mode, max_iter=3, tol=0)
        result = model.predict_with_details(X, predict_config=config, details=("diagnostics",))
        assert result.n_iter == len(result.loss_history)
        assert isinstance(result.converged, bool)
        assert result.affiliations is None
        assert result.instance_weights is None
        minimal = model.predict_with_details(X, predict_config=config)
        assert minimal.n_iter is None
        assert minimal.converged is None
        assert minimal.loss_history is None
        torch.testing.assert_close(result.prediction, minimal.prediction, rtol=0, atol=0)

    @pytest.mark.parametrize(("max_iter", "tol", "expected"), [(1, 0.0, False), (8, 1.0, True)])
    def test_convergence_flag_reports_the_tolerance_not_the_iteration_ceiling(
        self, max_iter, tol, expected
    ):
        model, X, _ = prediction_model()
        config = PredictConfig(predict_mode="iterative", max_iter=max_iter, tol=tol)
        result = model.predict_with_details(X, predict_config=config, details=("diagnostics",))
        assert result.converged is expected
        assert (result.n_iter < max_iter) is expected

    @pytest.mark.parametrize("name", ["n_iter", "converged", "loss_history"])
    def test_individual_diagnostic_requests_are_rejected(self, name):
        model, _, _ = prediction_model()
        with pytest.raises(ValueError, match="detail"):
            model.predict_with_details(None, details=(name,))

    @pytest.mark.parametrize(
        "details", [None, 42, "affiliations", ("unknown",), (["affiliations"],)]
    )
    def test_bad_detail_request_is_rejected_before_staging(self, details):
        model, _, _ = prediction_model()
        with pytest.raises(ValueError, match="detail"):
            model.predict_with_details(None, details=details)

    def test_single_pass_details_agree_with_primary_prediction(self):
        X, y = blobs(3, 0.15)
        network = Network.fit(deep_classification_recipe((4, 3)), X, y, max_iter=20)
        result = network.predict_with_details(X, details=("affiliations", "diagnostics"))
        torch.testing.assert_close(result.prediction, network.predict(X), rtol=0, atol=0)
        assert result.n_iter == 0
        assert result.converged is False
        assert result.loss_history == ()
        assert result.instance_weights is None
        assert result.affiliations is not None
        for gamma in result.affiliations.values():
            torch.testing.assert_close(gamma.sum(dim=1), torch.ones_like(X[:, 0]))
        minimal = network.predict_with_details(X)
        assert minimal.affiliations is None
        assert minimal.n_iter is None

    def test_invalid_detail_is_rejected_before_query_staging(self):
        X, y = blobs(2, 0.1)
        network = Network.fit(classification_recipe(K=3), X, y, max_iter=10)
        with pytest.raises(ValueError, match="detail"):
            network.predict_with_details(None, details=("unknown",))
        with pytest.raises(ValueError, match="instance_weights"):
            network.predict_with_details(None, details=("instance_weights",))

    def test_start_is_rejected_for_single_pass(self):
        X, y = blobs(2, 0.1)
        network = Network.fit(classification_recipe(K=3), X, y, max_iter=10)
        with pytest.raises(ValueError, match="iterative"):
            network.predict(X, predict_config=PredictConfig(), predict_init="uniform")

    def test_details_own_independent_detached_storage(self):
        model, X, _ = prediction_model()
        X.requires_grad_(True)
        policy = PredictConfig(predict_mode="iterative", max_iter=3)
        names = ("affiliations", "instance_weights")
        result = model.predict_with_details(X, predict_config=policy, details=names)
        expected = model.predict_with_details(X, predict_config=policy, details=names)
        assert not result.prediction.requires_grad
        with torch.inference_mode():
            result.prediction.zero_()
            result.instance_weights.zero_()
            for gamma in result.affiliations.values():
                assert not gamma.requires_grad
                gamma.zero_()
        actual = model.predict_with_details(X, predict_config=policy, details=names)
        torch.testing.assert_close(actual.prediction, expected.prediction, rtol=0, atol=0)
        torch.testing.assert_close(
            actual.instance_weights, expected.instance_weights, rtol=0, atol=0
        )
        for name, gamma in actual.affiliations.items():
            torch.testing.assert_close(gamma, expected.affiliations[name], rtol=0, atol=0)


class TestRecoveredWeights:
    def test_recovery_uses_the_training_normaliser_not_the_query_batch(self):
        X, y = blobs(2, 0.0, rows_per_blob=2, n_features=1)
        recipe = classification_recipe(Input(K=1, epsilon_T=1.0))
        state = Network.initialise(recipe, X, y, continuous_centroids=X.mean(0, keepdim=True))
        network = Network.fit(recipe, X, y, initial_state=state, max_iter=30)
        query = torch.tensor([[0.5], [0.0], [1.0]], dtype=X.dtype, device=X.device)
        result = network.predict_with_details(query, details=("instance_weights",))
        # Symmetry keeps the single centroid at 0.5 and training weights uniform.
        normaliser = torch.exp(-(X[:, 0] - 0.5).square()).sum()
        expected = torch.exp(-(query[:, 0] - 0.5).square()) / normaliser
        torch.testing.assert_close(result.instance_weights, expected)
        part = network.predict_with_details(query[:1], details=("instance_weights",))
        torch.testing.assert_close(part.instance_weights, expected[:1])
        assert not torch.isclose(
            result.instance_weights.sum(), result.instance_weights.new_tensor(1)
        )


class TestMemberDetails:
    @pytest.mark.parametrize("task", ["classification", "regression"])
    def test_member_details_preserve_order_and_delegate_the_complete_policy(self, task):
        model, X, cats = prediction_model(task, members=True)
        policy = PredictConfig(
            predict_mode="iterative",
            epsilon_P=0.4 if task == "classification" else None,
            max_iter=3,
            tol=0,
        )
        results = model.predict_all_with_details(
            X,
            X_cat=cats,
            predict_config=policy,
            predict_init=7,
            details=("affiliations", "instance_weights", "diagnostics"),
        )
        predictions = model.predict_all(X, X_cat=cats, predict_config=policy, predict_init=7)
        for member, result, prediction in zip(model.members, results, predictions, strict=True):
            expected = member.predict_with_details(
                X,
                X_cat=cats,
                predict_config=policy,
                predict_init=7,
                details=("affiliations", "instance_weights", "diagnostics"),
            )
            torch.testing.assert_close(result.prediction, prediction, rtol=0, atol=0)
            torch.testing.assert_close(result.prediction, expected.prediction, rtol=0, atol=0)
            torch.testing.assert_close(result.instance_weights, expected.instance_weights)
            for name, gamma in result.affiliations.items():
                torch.testing.assert_close(gamma, expected.affiliations[name])
            assert result.n_iter == expected.n_iter
            assert result.converged == expected.converged
            assert result.loss_history == expected.loss_history

    def test_member_details_require_retained_members(self):
        model, X, _ = prediction_model()
        with pytest.raises(ValueError, match='retain="members"'):
            model.predict_all_with_details(X)
