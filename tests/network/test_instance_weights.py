"""Learned instance weights are relative to the sample weights."""

import math

import pytest
import torch
from conftest import DEVICE, DTYPE, FIXED_BUDGET, RANK_DEFICIENT
from network._fixtures import blobs, materialise

from entlearn import ClassificationHead, Input, ManifoldInput, Network, Recipe, RegressionHead
from entlearn.network.blocks import input_common as inp

# Mirror pairs about 0.5 with equal sample weights keep a single centroid at 0.5, so each
# row's discretisation error is (x - 0.5)² at every iteration (scaled by alpha on a
# one-dimensional manifold, whose plane is the whole line).
_X = ((0.0,), (1.0,), (0.25,), (0.75,))
_CLASSES = (0, 1, 0, 1)
_SAMPLE_WEIGHTS = (1.0, 1.0, 3.0, 3.0)
_ALPHA = 0.5
_EPSILON_T = 0.1
_ERROR_SCALES = {"standard": 1.0, "manifold": _ALPHA}


def _symmetric_fit(kind: str) -> tuple[Network, torch.Tensor, torch.Tensor, float]:
    """Fit one hard cluster to the mirror pairs; return the network, data, w and error scale."""
    X = torch.tensor(_X, dtype=DTYPE, device=DEVICE)
    y = torch.tensor(_CLASSES, device=DEVICE)
    w = torch.tensor(_SAMPLE_WEIGHTS, dtype=DTYPE, device=DEVICE)
    first = (
        ManifoldInput(K=1, subspace_dimension=1, alpha=_ALPHA, epsilon=0.0, epsilon_T=_EPSILON_T)
        if kind == "manifold"
        else Input(K=1, epsilon=0.0, epsilon_D=math.inf, epsilon_T=_EPSILON_T)
    )
    recipe = Recipe.chain(first, ClassificationHead(), delta=1.0)
    state = Network.initialise(
        recipe, X, y, sample_weights=w, continuous_centroids=X.new_tensor([[0.5]])
    )
    network = Network.fit(recipe, X, y, sample_weights=w, initial_state=state, max_iter=30)
    return network, X, w / w.sum(), _ERROR_SCALES[kind]


def _error(network: Network, X: torch.Tensor, scale: float) -> torch.Tensor:
    """Return each row's discretisation error from the fitted single centroid."""
    centroid = network.inspect("continuous_centroids")["input"]
    return scale * (X - centroid).square().sum(1)


class TestRelativeInstanceWeightStep:
    @RANK_DEFICIENT
    @pytest.mark.parametrize("kind", ["standard", "manifold"])
    def test_learned_weights_tilt_the_sample_weights_by_the_error(self, kind):
        network, X, w, scale = _symmetric_fit(kind)
        b = _error(network, X, scale)
        expected = torch.softmax(w.log() - b / _EPSILON_T, 0)
        assert not torch.allclose(expected, torch.softmax(-b / _EPSILON_T, 0), rtol=0.1, atol=0)
        actual = network.inspect("training_instance_weights")["input"]
        torch.testing.assert_close(actual, expected)

    @RANK_DEFICIENT
    @pytest.mark.parametrize("kind", ["standard", "manifold"])
    def test_reported_loss_holds_the_relative_entropy_from_the_sample_weights(self, kind):
        network, X, w, scale = _symmetric_fit(kind)
        b = _error(network, X, scale)
        Wt = network.inspect("training_instance_weights")["input"]
        # Each class holds half the sample weight, so the head term at delta = 1 is log 2.
        relative_entropy = (Wt * (Wt / w).log()).sum()
        expected = (Wt * b).sum() + math.log(2) + _EPSILON_T * (relative_entropy - math.log(4))
        assert network.diagnostics.loss_history[-1] == pytest.approx(
            expected.item(), rel=16 * torch.finfo(DTYPE).eps, abs=0
        )

    # Both heads share the step: each head shapes the affiliations, and so the errors.
    @FIXED_BUDGET
    @RANK_DEFICIENT
    @pytest.mark.parametrize("task", ["classification", "regression"])
    @pytest.mark.parametrize("kind", ["standard", "manifold"])
    def test_a_large_temperature_tends_to_the_sample_weights(self, kind, task):
        X, codes = blobs(3, 0.15, n_features=2, seed=5)
        y = codes if task == "classification" else X.square()
        first = (
            ManifoldInput(K=3, subspace_dimension=1, epsilon=0.1, epsilon_T=1e4)
            if kind == "manifold"
            else Input(K=3, epsilon=0.1, epsilon_D=math.inf, epsilon_T=1e4)
        )
        head = ClassificationHead() if task == "classification" else RegressionHead()
        w = torch.linspace(1.0, 3.0, X.shape[0], dtype=DTYPE, device=DEVICE)
        network = Network.fit(Recipe.chain(first, head), X, y, sample_weights=w, max_iter=5)
        Wt = network.inspect("training_instance_weights")["input"]
        # On features in [0, 1] each error is at most a few units, so at epsilon_T = 1e4
        # every ratio Wt / w lies within 1e-3 of one; uniform weights miss by up to 50 %.
        torch.testing.assert_close(Wt, w / w.sum(), rtol=1e-3, atol=0)


class TestRecoveryUnderSampleWeights:
    @RANK_DEFICIENT
    @pytest.mark.parametrize("kind", ["standard", "manifold"])
    def test_recovery_divides_by_the_sample_weighted_training_normaliser(self, kind):
        network, X, w, scale = _symmetric_fit(kind)
        b = _error(network, X, scale)
        log_Z = torch.logsumexp(-b / _EPSILON_T + (X.shape[0] * w).log(), 0)
        unweighted = torch.logsumexp(-b / _EPSILON_T, 0)
        assert not torch.isclose(log_Z, unweighted, rtol=0.1, atol=0)
        query = X.new_tensor([[0.5], [0.0], [0.3], [1.0]])
        expected = (-_error(network, query, scale) / _EPSILON_T - log_Z).exp().clamp(max=1.0)
        result = network.predict_with_details(query, details=("instance_weights",))
        torch.testing.assert_close(result.instance_weights, expected)


class TestUniformSampleWeights:
    @torch.inference_mode()
    def test_the_unweighted_step_and_loss_term_are_unchanged_bit_for_bit(self):
        # No public oracle reaches the last bits, so this checks the step itself, against
        # the unweighted softmax and entropy evaluated in the same order. At T = 107 rows,
        # T times the staged weight 1/T is not exactly one in either dtype, so any use of
        # log(T w) on this path would show.
        generator = torch.Generator().manual_seed(7)
        X = torch.rand(107, 2, dtype=DTYPE, generator=generator).to(DEVICE)
        y = (X[:, 0] > 0.5).long()
        epsilon_T = 0.3
        recipe = Recipe.chain(
            Input(K=3, epsilon=0.2, epsilon_D=math.inf, epsilon_T=epsilon_T), ClassificationHead()
        )
        session = materialise(recipe, X, y, seed=3, warmup=1)
        block = session.graph.input
        inp.update_affiliations_(block, session)
        b = (block.gamma * block.live_cache.disc_cost).sum(1)

        inp.update_instance_weights_(block, session)
        Wt = block.instance_weights
        session.workspace.loss.zero_()
        inp.instance_weight_loss_(block, session)

        assert torch.equal(Wt, torch.softmax((b.min() - b) * (1.0 / epsilon_T), 0))
        entropy = torch.special.entr(Wt).sum()
        assert torch.equal(session.workspace.loss, -epsilon_T * entropy)
