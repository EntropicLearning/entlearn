"""The stopping rule: the loss change against the loss plus its offset."""

import math

import pytest
import torch
from conftest import DEVICE, DTYPE, FIXED_BUDGET
from network._fixtures import materialise, resume_case, skip_fit_iterations

from entlearn import Hidden, Input, Network, PredictConfig, Recipe, RegressionHead
from entlearn.network import fit as fit_ops
from entlearn.primitives.normalise import _tiny

pytestmark = FIXED_BUDGET

SEED = 5
# Enough iterations for every fit below to pass the tolerances the tests pick.
ITERATIONS = 30


def documented_offset(recipe, widths: dict[str, int], rows: int, features: int, outputs: int):
    """C from the documented formula: c log n for every entropy penalty the Recipe holds."""
    offset = 0.0
    for block in recipe.blocks:
        if isinstance(block, (Input, Hidden)):
            offset += block.epsilon * math.log(widths[block.name])
        if isinstance(block, Input):
            if math.isfinite(block.epsilon_T):
                offset += block.epsilon_T * math.log(rows)
            if math.isfinite(block.epsilon_D):
                offset += block.epsilon_D * math.log(features)
        if isinstance(block, RegressionHead) and math.isfinite(block.epsilon_M):
            offset += block.epsilon_M * math.log(outputs)
    return offset


def first_stop(losses, offsets, tol):
    """The first iteration n with ``|L_n - L_{n-1}| <= tol · |L_{n-1} + C_n|``."""
    for n in range(1, len(losses)):
        if abs(losses[n] - losses[n - 1]) <= tol * abs(losses[n - 1] + offsets[n]):
            return n
    return None


def discriminating_case(monkeypatch):
    """The first case and tolerance at which the rule stops where the rule without C would not."""
    for case in ("learned weights", "large instance-weight temperature"):
        recipe, X, y, _ = resume_case("regression")
        if case == "large instance-weight temperature":
            recipe = recipe.replace_block("input", epsilon_T=3.0)
        with monkeypatch.context() as patch:
            skip_fit_iterations(patch)
            start = Network.fit(recipe, X, y, seed=SEED, max_iter=1, tol=0)
        # At tol = 0 a fit runs to max_iter or to the first iteration that leaves the loss
        # unchanged, where every positive tolerance stops too.
        losses = Network.fit(recipe, X, y, seed=SEED, max_iter=ITERATIONS, tol=0)
        losses = losses.diagnostics.loss_history
        widths = [dict(start.schema.K_active)] + [
            dict(Network.fit(recipe, X, y, seed=SEED, max_iter=n, tol=0).schema.K_active)
            for n in range(1, len(losses))
        ]
        offsets = [
            documented_offset(recipe, w, rows=len(X), features=X.shape[1], outputs=y.shape[1])
            for w in widths
        ]
        for tol in (10 ** (-k / 4) for k in range(8, 40)):
            stop = first_stop(losses, offsets, tol)
            if stop is not None and first_stop(losses, [0.0] * len(losses), tol) != stop:
                return recipe, X, y, tol, stop
    pytest.fail("no case tells the rule from the rule without the offset")


class TestLossOffset:
    @pytest.mark.parametrize(
        "temperature",
        [0.0, _tiny(DTYPE) / 2, _tiny(DTYPE), 2 * _tiny(DTYPE), torch.finfo(DTYPE).eps / 16],
    )
    def test_feature_weight_loss_and_offset_share_the_zero_temperature_gate(self, temperature):
        X = torch.zeros(2, 2, dtype=DTYPE, device=DEVICE)
        y = torch.zeros(2, 1, dtype=DTYPE, device=DEVICE)
        recipe = Recipe.chain(Input(K=2, epsilon_D=temperature), RegressionHead())
        session = materialise(recipe, X, y, seed=SEED)

        loss = fit_ops._loss(session)

        # With zero reconstruction error and two uniform feature weights, only this
        # reward contributes. Scaling by its temperature makes a missing tiny term visible.
        if temperature == 0:
            assert loss == 0.0
            assert session.offset == 0.0
            return
        tolerance = 8 * torch.finfo(DTYPE).eps
        expected = math.log(2)
        assert loss / temperature == pytest.approx(-expected, rel=tolerance, abs=tolerance)
        assert session.offset / temperature == pytest.approx(expected, rel=tolerance, abs=tolerance)

    @pytest.mark.parametrize("task", ["classification", "regression"])
    def test_the_offset_sums_every_entropy_reward_at_its_largest_value(self, task):
        recipe, X, y, _ = resume_case(task)
        session = materialise(recipe, X, y, seed=SEED, warmup=1)
        fit_ops._loss(session)
        graph = session.graph
        assert {name: graph.affiliations(name).shape[1] for name in graph.order[:-1]} == {
            "input": 4,
            "hidden_1": 3,
        }
        # 30 rows and 3 features: epsilon_T, epsilon_D, the input's and the hidden block's
        # epsilon, and for regression epsilon_M over the 2 outputs.
        expected = 0.3 * math.log(30) + 0.2 * math.log(3) + 0.1 * math.log(4) + 0.15 * math.log(3)
        if task == "regression":
            expected += 0.2 * math.log(2)
        assert session.offset == pytest.approx(expected, rel=1e-12)

    def test_the_offset_follows_a_prune(self):
        recipe, X, y, _ = resume_case()
        session = materialise(recipe, X, y, seed=SEED)
        block = session.graph.input
        # Two clusters with no mass, which the prune below removes.
        block.gamma[:, 1:3] = 0
        block.gamma[:, 0] = 1 - block.gamma[:, 3]
        session.graph.prune_(block, session.workspace)
        fit_ops._loss(session)
        graph = session.graph
        widths = {name: graph.affiliations(name).shape[1] for name in graph.order[:-1]}
        assert widths["input"] == 2
        expected = documented_offset(recipe, widths, rows=30, features=3, outputs=0)
        assert session.offset == pytest.approx(expected, rel=1e-12)


class TestStoppingRule:
    def test_a_fit_stops_at_the_first_iteration_that_meets_the_loss_test(self, monkeypatch):
        recipe, X, y, tol, stop = discriminating_case(monkeypatch)
        network = Network.fit(recipe, X, y, seed=SEED, max_iter=ITERATIONS, tol=tol)
        assert network.diagnostics.converged
        assert network.diagnostics.n_iter == stop


class TestIterativePredictionStoppingRule:
    @pytest.mark.parametrize("task", ["classification", "regression"])
    def test_iterative_prediction_stops_against_its_loss_plus_its_offset(self, task):
        recipe, X, y, _ = resume_case(task)
        model = Network.fit(recipe, X, y, seed=SEED, max_iter=ITERATIONS, tol=0)
        # Rows away from the training rows take more iterations to settle.
        noise = torch.randn(X.shape, generator=torch.Generator().manual_seed(1), dtype=X.dtype)
        rows = X + 0.1 * noise.to(X)
        # A large output temperature makes epsilon_P log M dominate the classification C.
        epsilon_P = 2.0 if task == "classification" else None

        def predict(tol):
            config = PredictConfig(
                predict_mode="iterative", epsilon_P=epsilon_P, max_iter=ITERATIONS, tol=tol
            )
            return model.predict_with_details(rows, predict_config=config, details=("diagnostics",))

        widths = dict(model.schema.K_active)
        # C of the query loss: epsilon log K for each soft clustering, and epsilon_P log M.
        offset = 0.1 * math.log(widths["input"]) + 0.15 * math.log(widths["hidden_1"])
        if epsilon_P is not None:
            offset += epsilon_P * math.log(widths["output"])
        # The first iteration has no previous loss to compare with.
        losses = [math.nan, *predict(0.0).loss_history]
        offsets = [offset] * len(losses)
        # Loose tolerances distinguish the rules even when the query settles in two updates.
        for tol in (10 ** (-k / 8) for k in range(128)):
            stop = first_stop(losses, offsets, tol)
            if stop is not None and first_stop(losses, [0.0] * len(losses), tol) != stop:
                break
        else:
            pytest.fail("no tolerance tells the rule from the rule without the offset")
        result = predict(tol)
        assert result.converged
        assert result.n_iter == stop
