"""Classification calibration through the public fitted-model boundary."""

import math

import pytest
import torch
from conftest import DEVICE, DTYPE, FIXED_BUDGET
from network._fixtures import blobs, classification_recipe, resume_case
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax

from entlearn import ClassificationHead, Coupling, Input, Network, PredictConfig, Recipe
from entlearn.network import calibration
from entlearn.primitives.normalise import _tiny

pytestmark = FIXED_BUDGET


class TestCalibration:
    def test_a_subnormal_coupling_strength_still_calibrates_a_soft_temperature(self):
        # Every positive temperature is soft, so delta / beta needs no floor in any dtype.
        X = torch.tensor([[0.0], [1.0]], dtype=DTYPE, device=DEVICE)
        target = torch.tensor([[0.99, 0.01], [0.01, 0.99]], dtype=DTYPE, device=DEVICE)
        delta = _tiny(DTYPE) / 2
        model = Network.fit(
            Recipe.chain(
                Input(K=2, epsilon=0.5), ClassificationHead(coupling=Coupling.M), delta=delta
            ),
            X,
            target,
            max_iter=1,
        )
        epsilon_P = model.predict_config.epsilon_P
        assert epsilon_P is not None
        assert 0 < epsilon_P < _tiny(DTYPE)
        assert torch.isfinite(model.predict(X).log()).all()

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
        # The root finder stops within sqrt(eps) in log beta, which moves each probability by
        # at most p0 * p1 * log(7) < sqrt(eps). At the minimised loss that error is second
        # order, below eps, beside about 2 eps from rounding the float32 logs.
        eps = torch.finfo(torch.float32).eps
        torch.testing.assert_close(probabilities, expected, rtol=0, atol=math.sqrt(eps))
        served_loss = -probabilities.gather(1, codes[:, None]).log().mean()
        assert torch.isfinite(served_loss)
        assert float(served_loss) == pytest.approx(
            -(7 / 8 * math.log(7 / 8) + 1 / 8 * math.log(1 / 8)), abs=4 * eps
        )
        # Complete overrides exercise the same numerical owner, including the hard gate.
        for beta in (0.5, 1.0, 2.0):
            override = PredictConfig(epsilon_P=model.predict_config.epsilon_P / beta)
            oracle = (probabilities.log() * beta).softmax(dim=1)
            torch.testing.assert_close(model.predict(X, predict_config=override), oracle)
        hard = model.predict(X, predict_config=PredictConfig(epsilon_P=0))
        majority = torch.tensor([1.0, 0.0], dtype=X.dtype, device=DEVICE).expand(8, -1)
        torch.testing.assert_close(hard, majority, rtol=0, atol=0)

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
        # delta = 1, so epsilon_P = 1 is the uncalibrated beta = 1.
        uncalibrated = model.predict(X, predict_config=PredictConfig(epsilon_P=1.0))
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
            # Each solver places the minimum only to about sqrt(eps) in log beta: ours stops
            # at that bracket width, and the loss is flat to round-off within it for scipy.
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
    def test_calibration_respects_positive_temperatures_below_machine_precision(
        self, hard_target, delta_scale
    ):
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
            torch.testing.assert_close(model.predict(X), torch.eye(2, dtype=DTYPE, device=DEVICE))
        else:
            assert 0 < epsilon < floor
            log_p = model.predict(X).log()
            assert bool(torch.isfinite(log_p).all())
            baseline = model.predict(X, predict_config=PredictConfig()).log()
            assert float(-(target * log_p).sum()) <= float(-(target * baseline).sum())
            z = model.predict(X, predict_config=PredictConfig(epsilon_P=delta)).log().cpu().numpy()
            truth = target.cpu().numpy()
            oracle = minimize_scalar(
                lambda x: -(truth * log_softmax(math.exp(x) * z, axis=1)).sum(),
                bounds=(math.log(1e-6), math.log(1e6)),
                method="bounded",
                options={"xatol": floor},
            )
            assert float(-(target * log_p).sum()) <= oracle.fun + 10 * floor

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

    def test_small_positive_temperature_is_not_forced_to_an_infeasible_hard_readout(self):
        X = torch.tensor([[0.0], [1.0]], dtype=DTYPE, device=DEVICE)
        target = torch.full((2, 2), 0.5, dtype=DTYPE, device=DEVICE)
        recipe = Recipe.chain(
            Input(K=2),
            ClassificationHead(coupling=Coupling.M),
            delta=torch.finfo(DTYPE).eps * 1e-7,
        )
        model = Network.fit(recipe, X, target, max_iter=1)
        assert model.predict_config.epsilon_P > 0
        torch.testing.assert_close(model.predict(X), target)

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
    def test_a_supplied_temperature_is_kept(self, epsilon):
        X, codes = blobs(2, 0.28)
        model = Network.fit(
            classification_recipe(),
            X,
            codes,
            predict_config=PredictConfig(epsilon_P=epsilon),
            max_iter=2,
        )
        assert model.predict_config.epsilon_P == epsilon
        torch.testing.assert_close(
            model.predict(X), model.predict(X, predict_config=PredictConfig(epsilon_P=epsilon))
        )

    def test_a_failed_calibration_leaves_the_source_network_unchanged(self, monkeypatch):
        recipe, X, codes, _ = resume_case()
        model = Network.fit(recipe, X, codes, max_iter=2, tol=0)
        before = model.predict(X)
        config, diagnostics = model.predict_config, model.diagnostics

        def non_finite(out, logits, numerator, denominator, scratch):
            out.fill_(torch.nan)

        # Resuming continues the source's unconverged trajectory, so it could alter the source.
        with monkeypatch.context() as patch:
            patch.setattr(calibration, "scale_logits_", non_finite)
            with pytest.raises(RuntimeError, match="non-finite derivative"):
                model.resume(X, codes, max_iter=4)
        assert model.predict_config == config
        assert model.diagnostics == diagnostics
        torch.testing.assert_close(model.predict(X), before, rtol=0, atol=0)
