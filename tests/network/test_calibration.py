"""Classification calibration through the public fitted-model boundary."""

import math

import pytest
import torch
from _alloc import assert_no_float64
from conftest import DEVICE, DTYPE
from network._fixtures import blobs, classification_recipe
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax

from entlearn import ClassificationHead, Coupling, Input, Network, PredictConfig, Recipe


class TestCalibration:
    def test_large_delta_prices_the_probabilities_actually_served(self):
        # delta is representable in float32, but delta * log(theta) need not be.
        X = torch.zeros(8, 1, dtype=torch.float32, device=DEVICE)
        codes = torch.tensor([0] * 7 + [1], dtype=torch.int64, device=DEVICE)
        model = Network.fit(
            Recipe.chain(Input(K=2), ClassificationHead(coupling=Coupling.M), delta=1e38),
            X,
            codes,
            max_iter=2,
            seed=0,
        )
        probabilities = model.predict(X)
        expected = torch.tensor([7 / 8, 1 / 8], dtype=X.dtype, device=DEVICE).expand(8, -1)
        torch.testing.assert_close(probabilities, expected, rtol=1e-3, atol=1e-4)
        served_loss = -probabilities.gather(1, codes[:, None]).log().mean()
        assert torch.isfinite(served_loss)
        assert float(served_loss) == pytest.approx(
            -(7 / 8 * math.log(7 / 8) + 1 / 8 * math.log(1 / 8)), abs=1e-5
        )
        # Complete overrides exercise the same numerical owner, including the hard gate.
        for beta in (0.5, 1.0, 2.0):
            override = PredictConfig(epsilon_P=model.predict_config.epsilon_P / beta)
            oracle = (probabilities.log() * beta).softmax(dim=1)
            torch.testing.assert_close(model.predict(X, predict_config=override), oracle)
        hard = model.predict(X, predict_config=PredictConfig(epsilon_P=0))
        torch.testing.assert_close(hard, torch.nn.functional.one_hot(hard.argmax(1), 2).to(X))

    @pytest.mark.parametrize("coupling", [Coupling.M, Coupling.S])
    @pytest.mark.parametrize("masked", [False, True])
    @pytest.mark.parametrize("distribution_target", [False, True])
    def test_weighted_log_loss_matches_scipy(self, distribution_target, masked, coupling):
        X, codes = blobs(3, 0.28, rows_per_blob=16, seed=4)
        target = torch.nn.functional.one_hot(codes, 3).to(DTYPE)
        if distribution_target:
            target = 0.8 * target + 0.2 / 3
        weights = torch.linspace(0.2, 2, X.shape[0], dtype=DTYPE, device=DEVICE)
        classes = torch.tensor([0.5, 1.0, 1.5], dtype=DTYPE, device=DEVICE)
        supplied_target = target if distribution_target else codes
        if masked:
            classes[0] = 0
            supplied_target = supplied_target.clone()
            supplied_target[-1] = 0 if distribution_target else -1
            target[-1] = 0
        model = Network.fit(
            Recipe.chain(Input(K=4, epsilon=0.15), ClassificationHead(coupling=coupling)),
            X,
            supplied_target,
            sample_weights=weights,
            class_weights=classes,
            max_iter=5,
            tol=0,
            seed=7,
        )
        epsilon = model.predict_config.epsilon_P
        assert epsilon is not None
        assert 1e-6 <= 1 / epsilon <= 1e6
        uncalibrated = model.predict(X, predict_config=PredictConfig())
        logits = uncalibrated.log().cpu().numpy()
        components = (weights[:, None] * target * classes).cpu().numpy()
        components /= components.sum()

        def objective(log_beta):
            return -(components * log_softmax(math.exp(log_beta) * logits, axis=1)).sum()

        oracle = minimize_scalar(
            objective,
            bounds=(math.log(1e-6), math.log(1e6)),
            method="bounded",
            options={"xatol": torch.finfo(DTYPE).eps},
        )
        achieved = objective(math.log(1 / epsilon))
        tolerance = 10 * torch.finfo(DTYPE).eps
        assert achieved <= objective(0)
        oracle_loss = min(oracle.fun, objective(math.log(1e-6)), objective(math.log(1e6)))
        assert achieved <= oracle_loss + tolerance
        if coupling is Coupling.M and not masked:
            # These fits have identifiable interior optima; the other cases include
            # flat S heads and boundary optima, where loss is the relevant comparison.
            assert math.log(1 / epsilon) == pytest.approx(
                oracle.x, abs=10 * math.sqrt(torch.finfo(DTYPE).eps)
            )
        torch.testing.assert_close(
            model.predict(X), model.predict(X, predict_config=model.predict_config)
        )
        target_weights = weights[:, None] * target * classes
        selected = target_weights > 0
        served_loss = (
            -(target_weights[selected] * model.predict(X)[selected].log()).sum()
            / target_weights.sum()
        )
        baseline_loss = (
            -(target_weights[selected] * uncalibrated[selected].log()).sum() / target_weights.sum()
        )
        assert float(served_loss) <= float(baseline_loss) + tolerance

    @pytest.mark.parametrize("hard_target", [False, True])
    @pytest.mark.parametrize("delta_scale", [0.5, 2.0])
    def test_calibration_considers_the_output_gate(self, hard_target, delta_scale):
        X = torch.tensor([[0.0], [1.0]], dtype=DTYPE, device=DEVICE)
        codes = torch.tensor([0, 1], dtype=torch.int64, device=DEVICE)
        target = torch.tensor(
            [[256 / 257, 1 / 257], [1 / 257, 256 / 257]], dtype=DTYPE, device=DEVICE
        )
        floor = torch.finfo(DTYPE).eps
        delta = delta_scale * floor
        model = Network.fit(
            Recipe.chain(
                Input(K=2, epsilon=0.5), ClassificationHead(coupling=Coupling.M), delta=delta
            ),
            X,
            codes if hard_target else target,
            max_iter=1,
        )
        epsilon = model.predict_config.epsilon_P
        assert epsilon is not None
        if hard_target:
            assert epsilon <= floor
            torch.testing.assert_close(model.predict(X), torch.eye(2, dtype=DTYPE, device=DEVICE))
        else:
            # The unconstrained smooth optimum lies in the hard-output interval.
            # Its soft limit is attainable just above the gate; a hard answer has infinite loss.
            assert epsilon > floor
            assert delta / epsilon == pytest.approx(delta_scale)
            log_p = model.predict(X).log()
            assert bool(torch.isfinite(log_p).all())
            baseline = model.predict(X, predict_config=PredictConfig()).log()
            assert float(-(target * log_p).sum()) <= float(-(target * baseline).sum())
            z = model.predict(X, predict_config=PredictConfig(epsilon_P=2 * floor)).log()
            z = z.cpu().numpy() / (delta / (2 * floor))
            truth = target.cpu().numpy()
            oracle = minimize_scalar(
                lambda x: -(truth * log_softmax(math.exp(x) * z, axis=1)).sum(),
                bounds=(math.log(1e-6), math.log(delta_scale)),
                method="bounded",
                options={"xatol": floor},
            )
            assert float(-(target * log_p).sum()) == pytest.approx(
                oracle.fun, abs=10 * math.sqrt(floor)
            )

    def test_flat_objective_keeps_the_uncalibrated_temperature(self):
        X, codes = blobs(2, 0.28)
        model = Network.fit(
            Recipe.chain(Input(K=1), ClassificationHead(coupling=Coupling.S), delta=0.7),
            X,
            codes,
            max_iter=2,
        )
        assert model.predict_config.epsilon_P == 0.7
        torch.testing.assert_close(model.predict(X), torch.full_like(X[:, :2], 0.5))

    def test_no_finite_candidate_prevents_publication(self):
        X = torch.tensor([[0.0], [1.0]], dtype=DTYPE, device=DEVICE)
        target = torch.full((2, 2), 0.5, dtype=DTYPE, device=DEVICE)
        recipe = Recipe.chain(
            Input(K=2),
            ClassificationHead(coupling=Coupling.M),
            delta=torch.finfo(DTYPE).eps * 1e-7,
        )
        with pytest.raises(RuntimeError, match="no finite-loss read-out"):
            Network.fit(recipe, X, target, max_iter=1)

    def test_underflow_cannot_make_returned_probabilities_worse_than_uncalibrated(self):
        X = torch.tensor(
            [[0.0], [0.05], [0.495], [0.505], [0.95], [1.0]], dtype=DTYPE, device=DEVICE
        )
        codes = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int64, device=DEVICE)
        target = torch.nn.functional.one_hot(codes, 2).to(DTYPE) * (1 - 1e-6) + 1e-6 / 2
        model = Network.fit(
            classification_recipe(Input(K=2, epsilon=0.03)), X, target, max_iter=4, seed=7
        )
        probabilities = model.predict(X)
        baseline = model.predict(X, predict_config=PredictConfig())
        assert bool((probabilities > 0).all())
        assert float(-(target * probabilities.log()).sum()) <= float(
            -(target * baseline.log()).sum()
        )

    @pytest.mark.parametrize("epsilon", [0, 1])
    def test_supplied_temperature_bypasses_calibration(self, monkeypatch, epsilon):
        X, codes = blobs(2, 0.28)

        def unavailable(*args, **kwargs):
            raise RuntimeError("log-softmax unavailable")

        monkeypatch.setattr(torch, "log_softmax", unavailable)
        model = Network.fit(
            classification_recipe(),
            X,
            codes,
            predict_config=PredictConfig(epsilon_P=epsilon),
            max_iter=2,
        )
        assert model.predict_config.epsilon_P == epsilon
        assert bool(torch.isfinite(model.predict(X)).all())

    @pytest.mark.parametrize("failure", ["exception", "nonfinite"])
    def test_failed_calibration_does_not_replace_a_fitted_network(self, monkeypatch, failure):
        X, codes = blobs(2, 0.28)
        model = Network.fit(classification_recipe(), X, codes, max_iter=2)
        before = model.predict(X).clone()
        config = model.predict_config
        diagnostics = model.diagnostics

        def unavailable(values, **kwargs):
            if failure == "exception":
                raise RuntimeError("log-softmax unavailable")
            return torch.full_like(values, torch.nan)

        monkeypatch.setattr(torch, "log_softmax", unavailable)
        with pytest.raises(RuntimeError):
            model = Network.fit(classification_recipe(), X, codes, max_iter=2)
        assert model.predict_config == config
        assert model.diagnostics == diagnostics
        torch.testing.assert_close(model.predict(X), before)

    def test_float32_calibration_allocates_no_float64_tensors(self):
        # The allocation dtype is the subject of this test.
        X, codes = blobs(2, 0.28)
        model = assert_no_float64(
            Network.fit,
            classification_recipe(Input(K=3, epsilon=0.2)),
            X.to(torch.float32),
            codes,
            max_iter=2,
        )
        result = assert_no_float64(model.predict, X.to(torch.float32))
        assert result.dtype == torch.float32
