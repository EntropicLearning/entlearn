"""Starting coordinates affect refinement without changing fitted state."""

from dataclasses import replace

import pytest
import torch
from network._fixtures import prediction_model

from entlearn import Coupling, PredictConfig


class TestPredictionStarts:
    @pytest.mark.parametrize("task", ["classification", "regression"])
    def test_supplied_tensor_is_the_first_iterations_prediction_coordinate(self, task):
        model, X, categories = prediction_model(task)
        config = replace(model.predict_config, predict_mode="iterative", max_iter=1)
        start = model.predict(X, X_cat=categories).flip(0)
        original = start.clone()
        result = model.predict_with_details(
            X,
            X_cat=categories,
            predict_config=config,
            predict_init=start,
            details=("affiliations", "diagnostics"),
        )
        torch.testing.assert_close(result.prediction, start, rtol=0, atol=0)
        torch.testing.assert_close(start, original, rtol=0, atol=0)
        assert torch.isfinite(
            torch.tensor(result.loss_history, dtype=X.dtype, device=X.device)
        ).all()
        default = model.predict_with_details(
            X, X_cat=categories, predict_config=config, details=("affiliations",)
        )
        assert any(
            not torch.equal(value, default.affiliations[name])
            for name, value in result.affiliations.items()
        )

    @pytest.mark.parametrize("task,word", [("classification", "uniform"), ("regression", "mean")])
    def test_symmetric_start_and_reproducible_seed(self, task, word):
        model, X, _ = prediction_model(task)
        config = replace(model.predict_config, predict_mode="iterative", max_iter=1)
        symmetric = model.predict(X, predict_config=config, predict_init=word)
        torch.testing.assert_close(symmetric, symmetric[:1].expand_as(symmetric))
        if task == "classification":
            torch.testing.assert_close(
                symmetric, torch.full_like(symmetric, 1 / symmetric.shape[1])
            )
        else:
            torch.testing.assert_close(symmetric[0], model._graph.head.C_y.mean(dim=1))
        seeded = model.predict(X, predict_config=config, predict_init=41)
        again = model.predict(X, predict_config=config, predict_init=41)
        other = model.predict(X, predict_config=config, predict_init=42)
        torch.testing.assert_close(seeded, again, rtol=0, atol=0)
        assert not torch.equal(seeded, other)

    def test_arithmetic_start_uses_the_arithmetic_single_pass(self):
        model, X, _ = prediction_model()
        policy = replace(model.predict_config, predict_mode="iterative", max_iter=1)
        expected = model.predict(X, predict_config=PredictConfig(output_mode="arithmetic"))
        result = model.predict(X, predict_config=policy, predict_init="arithmetic")
        torch.testing.assert_close(result, expected, rtol=0, atol=0)

    @pytest.mark.parametrize("bad", ["negative", "sum", "device"])
    def test_classification_tensor_checks_simplex_and_device(self, bad):
        model, X, _ = prediction_model()
        start = model.predict(X).clone()
        if bad == "negative":
            start[0] = start.new_tensor([-1, 1, 1])
        elif bad == "sum":
            start[0] *= 0.5
        else:
            start = torch.empty(start.shape, dtype=start.dtype, device="meta")
        with pytest.raises(ValueError, match="predict_init"):
            model.predict(
                X, predict_config=PredictConfig(predict_mode="iterative"), predict_init=start
            )

    @pytest.mark.parametrize("task", ["classification", "regression"])
    @pytest.mark.parametrize("bad", ["wrong", "geometric", True, 2.5, -1, 2**63])
    def test_bad_symbolic_start_fails_before_staging(self, task, bad):
        model, _, _ = prediction_model(task)
        with pytest.raises(ValueError, match=r"predict_init|seed"):
            model.predict(
                None, predict_config=PredictConfig(predict_mode="iterative"), predict_init=bad
            )

    @pytest.mark.parametrize("task", ["classification", "regression"])
    @pytest.mark.parametrize("bad", ["shape", "dtype", "nan", "inf"])
    def test_invalid_tensor_is_rejected(self, task, bad):
        model, X, _ = prediction_model(task)
        start = model.predict(X)
        if bad == "shape":
            start = start[:, 0]
        elif bad == "dtype":
            start = start.to(torch.float32 if X.dtype == torch.float64 else torch.float64)
        else:
            start = start.clone()
            start[0, 0] = float(bad)
        with pytest.raises(ValueError, match="predict_init"):
            model.predict(
                X, predict_config=PredictConfig(predict_mode="iterative"), predict_init=start
            )

    def test_arithmetic_start_on_s_head_is_rejected_before_staging(self):
        model, _, _ = prediction_model(head_coupling=Coupling.S)
        with pytest.raises(ValueError, match="arithmetic"):
            model.predict(
                None,
                predict_config=PredictConfig(predict_mode="iterative"),
                predict_init="arithmetic",
            )
