"""The delta_cat starting value prices uninformative features of both kinds alike."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from conftest import DEVICE, DTYPE

from entlearn import ClassificationHead, Input, Network, Recipe
from entlearn.helpers import delta_cat_guess

# These inputs are float64 NumPy arrays, so the helper computes in float64 on every lane;
# this allows for the rounding of sums over a few thousand rows.
FLOAT64_REL = 1e4 * np.finfo(np.float64).eps


def balanced_codes(rng: np.random.Generator, rows: int, levels: int = 3) -> np.ndarray:
    """Return codes whose levels are exactly equally frequent, in random order."""
    return rng.permutation(np.arange(rows) % levels)


class TestDeltaCatGuess:
    def test_uniform_noise_with_three_equal_levels_gives_one_eighth(self) -> None:
        rng = np.random.default_rng(0)
        rows = 12_000
        X_cont = rng.uniform(0.0, 1.0, (rows, 5))
        codes = [balanced_codes(rng, rows) for _ in range(4)]

        # Var = 1/12 against (1 - 1/3) = 2/3. The sample variance of uniform noise has a
        # relative standard error of sqrt(0.8 / rows), about 0.008 here.
        assert delta_cat_guess(X_cont, codes) == pytest.approx(0.125, rel=0.03)

    def test_standardised_columns_with_three_equal_levels_give_three_halves(self) -> None:
        rng = np.random.default_rng(1)
        rows = 3_000
        X_cont = rng.normal(0.0, 1.0, (rows, 4))
        X_cont = (X_cont - X_cont.mean(axis=0)) / X_cont.std(axis=0)
        codes = [balanced_codes(rng, rows) for _ in range(2)]

        assert delta_cat_guess(X_cont, codes) == pytest.approx(1.5, rel=FLOAT64_REL)

    def test_parity_value_keeps_uninformative_feature_weights_equal(self) -> None:
        rng = np.random.default_rng(2)
        rows = 300
        grid = np.linspace(0.0, 1.0, rows)
        X_cont = np.column_stack([rng.permutation(grid) for _ in range(3)])
        codes = [balanced_codes(rng, rows) for _ in range(2)]
        target = rng.integers(0, 2, rows)

        # One cluster knows nothing about any feature, so each costs its uninformative
        # cost, and a small epsilon_D turns any mismatch into unequal feature weights.
        recipe = Recipe.chain(
            Input(K=1, epsilon_D=0.05, delta_cat=delta_cat_guess(X_cont, codes)),
            ClassificationHead(),
        )
        network = Network.fit(
            recipe,
            torch.as_tensor(X_cont, dtype=DTYPE, device=DEVICE),
            torch.as_tensor(target, device=DEVICE),
            X_cat=tuple(torch.as_tensor(code, device=DEVICE) for code in codes),
            max_iter=20,
        )

        weights = network.inspect("feature_weights")["input"]
        torch.testing.assert_close(weights, torch.full_like(weights, 1 / 5))

    def test_sample_weights_act_as_repeated_rows(self) -> None:
        rng = np.random.default_rng(3)
        rows = 400
        X_cont = rng.uniform(0.0, 1.0, (rows, 3))
        codes = [rng.integers(0, 3, rows), rng.integers(0, 4, rows)]
        counts = rng.integers(1, 5, rows)
        # Heavier rows sit at the edges of the first column and on level 0, so the
        # weights change both medians.
        counts[(X_cont[:, 0] < 0.2) | (codes[0] == 0)] += 6

        weighted = delta_cat_guess(X_cont, codes, sample_weights=counts.astype(np.float64))
        repeated = delta_cat_guess(
            np.repeat(X_cont, counts, axis=0), [np.repeat(code, counts) for code in codes]
        )

        assert weighted == pytest.approx(repeated, rel=FLOAT64_REL)
        assert weighted != pytest.approx(delta_cat_guess(X_cont, codes), rel=0.01)

    def test_levels_are_the_distinct_codes(self) -> None:
        rng = np.random.default_rng(4)
        rows = 200
        X_cont = rng.uniform(0.0, 1.0, (rows, 2))
        codes = rng.integers(0, 3, rows)

        expected = delta_cat_guess(X_cont, [codes])

        assert delta_cat_guess(X_cont, [np.array([0, 5, 9])[codes]]) == expected
        assert delta_cat_guess(X_cont, [np.array(["a", "b", "c"])[codes]]) == expected

    def test_constant_columns_are_left_out(self) -> None:
        rng = np.random.default_rng(5)
        rows = 200
        X_cont = rng.uniform(0.0, 1.0, (rows, 2))
        codes = [rng.integers(0, 3, rows), rng.integers(0, 2, rows)]
        padded_X = np.column_stack((X_cont, np.full(rows, 0.5), np.full(rows, 0.5)))
        padded_codes = [*codes, np.zeros(rows, dtype=np.int64), np.ones(rows, dtype=np.int64)]

        assert delta_cat_guess(padded_X, padded_codes) == delta_cat_guess(X_cont, codes)


def _uniform(rows: int = 6, columns: int = 2) -> np.ndarray:
    return np.random.default_rng(6).uniform(0.0, 1.0, (rows, columns))


_CODES = [np.array([0, 1, 2, 0, 1, 2])]


class TestDeltaCatGuessRejections:
    @pytest.mark.parametrize(
        ("X_cont", "codes", "kwargs", "message"),
        [
            pytest.param(
                np.empty((6, 0)), _CODES, {}, "varying continuous feature", id="no-continuous"
            ),
            pytest.param(_uniform(), [], {}, "at least two levels", id="no-categorical"),
            pytest.param(
                np.full((6, 2), 0.5),
                _CODES,
                {},
                "varying continuous feature",
                id="constant-continuous",
            ),
            pytest.param(
                _uniform(), [np.zeros(6, dtype=np.int64)], {}, "at least two levels", id="one-level"
            ),
            pytest.param(
                _uniform()[:, 0],
                _CODES,
                {},
                "X_cont must be a non-empty finite",
                id="one-dimensional",
            ),
            pytest.param(
                np.empty((0, 2)),
                [np.empty(0)],
                {},
                "X_cont must be a non-empty finite",
                id="no-rows",
            ),
            pytest.param(
                np.where(np.eye(6, 2) > 0, np.nan, _uniform()),
                _CODES,
                {},
                "X_cont contains NaN",
                id="nan",
            ),
            pytest.param(
                np.where(np.eye(6, 2) > 0, np.inf, _uniform()),
                _CODES,
                {},
                "X_cont contains NaN, infinity",
                id="inf",
            ),
            pytest.param(
                _uniform(),
                [np.array([0.0, 1.0, np.nan, 0.0, 1.0, 2.0])],
                {},
                r"categorical_codes\[0\] contains NaN",
                id="nan-code",
            ),
            pytest.param(
                _uniform(),
                [np.array(["a", "b", None, "a", "b", "c"], dtype=object)],
                {},
                r"categorical_codes\[0\] contains NaN",
                id="none-code",
            ),
            pytest.param(
                _uniform(),
                [np.arange(5)],
                {},
                r"categorical_codes\[0\].*one entry per",
                id="short-codes",
            ),
            pytest.param(
                _uniform(),
                _CODES,
                {"sample_weights": np.ones(5)},
                "sample_weights must be a length-6",
                id="short-weights",
            ),
            pytest.param(
                _uniform(),
                _CODES,
                {"sample_weights": np.array([1.0, 1.0, 0.0, 1.0, 1.0, 1.0])},
                "sample_weights must be positive",
                id="zero-weight",
            ),
            pytest.param(
                _uniform(),
                _CODES,
                {"sample_weights": np.array([1.0, 1.0, -1.0, 1.0, 1.0, 1.0])},
                "sample_weights must be positive",
                id="negative-weight",
            ),
            pytest.param(
                _uniform(),
                _CODES,
                {"sample_weights": np.array([1.0, 1.0, np.nan, 1.0, 1.0, 1.0])},
                "sample_weights must contain only finite",
                id="nan-weight",
            ),
        ],
    )
    def test_malformed_or_degenerate_input_is_rejected(
        self,
        X_cont: np.ndarray,
        codes: list[np.ndarray],
        kwargs: dict[str, np.ndarray],
        message: str,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            delta_cat_guess(X_cont, codes, **kwargs)
