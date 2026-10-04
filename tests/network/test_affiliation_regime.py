"""The affiliation regime depends on the temperature and the dtype, not on the row count."""

import math

import pytest
import torch
from conftest import DEVICE

from entlearn import ClassificationHead, Input, Network, Recipe

# Pinned in every lane: float32 is where epsilon / T reaches machine precision at
# realistic row counts, which is what the old row-count gate got wrong.
_DTYPE = torch.float32
_ROWS = 200
_REPEATS = 500
_EPSILON = 1e-2
_SEED = 7


def _small_data() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(_SEED)
    X = torch.rand(_ROWS, 2, generator=generator, dtype=_DTYPE).to(DEVICE)
    y = (X[:, 0] > 0.5).to(torch.int64)
    return X, y


def _affiliations(network: Network) -> torch.Tensor:
    return network.inspect("training_affiliations")["input"]


class TestTrainingRowCount:
    @pytest.mark.filterwarnings("ignore::entlearn.ConvergenceWarning")
    @pytest.mark.parametrize("weighted", (False, True), ids=("uniform", "weighted"))
    def test_tiled_rows_keep_the_soft_affiliations_of_the_original_rows(
        self, weighted: bool
    ) -> None:
        # epsilon / T crosses float32 machine precision between the two row counts,
        # while the assignment exponent -d_tk / epsilon is the same for both.
        eps = torch.finfo(_DTYPE).eps
        assert _EPSILON / (_ROWS * _REPEATS) <= eps < _EPSILON / _ROWS
        X, y = _small_data()
        recipe = Recipe.chain(Input(K=3, epsilon=_EPSILON), ClassificationHead())
        # Fixed centroids keep the start device-independent. Two of them share class 0,
        # so the rows between them stay soft whatever the head learns.
        centroids = torch.tensor(
            [[0.25, 0.25], [0.25, 0.75], [0.75, 0.5]], dtype=_DTYPE, device=DEVICE
        )
        state = Network.initialise(recipe, X, y, continuous_centroids=centroids)
        small = Network.fit(recipe, X, y, initial_state=state, max_iter=5, tol=0)
        # Every original row carries the same total weight, split 1:3 between its copies
        # on the weighted path, so both tiled fits minimise the small fit's loss.
        sample_weights = None
        if weighted:
            copy_weights = torch.tensor([1.0, 3.0], dtype=_DTYPE, device=DEVICE)
            sample_weights = copy_weights.repeat(_REPEATS // 2).repeat_interleave(_ROWS)
        tiled = Network.fit(
            recipe,
            X.repeat(_REPEATS, 1),
            y.repeat(_REPEATS),
            sample_weights=sample_weights,
            initial_state=state,
            max_iter=5,
            tol=0,
        )

        expected = _affiliations(small)
        actual = _affiliations(tiled).view(_REPEATS, _ROWS, -1)
        for gamma in (expected, actual):
            assert bool((gamma.amax(dim=-1) < 0.9).any())
        # The centroids are sums over the rows, whose float32 rounding grows like
        # sqrt(T) eps, and the exponent -d_tk / epsilon passes an error in d_tk to the
        # affiliations divided by epsilon. Measured gaps stay below a tenth of it.
        tolerance = 2 * math.sqrt(_ROWS * _REPEATS) * eps / _EPSILON
        torch.testing.assert_close(actual, expected.expand_as(actual), rtol=0, atol=tolerance)
