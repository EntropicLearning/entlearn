"""Categorical classification through the public Network lifecycle."""

import pytest
import torch
from conftest import DEVICE, DTYPE, FIXED_BUDGET, by_depth
from network._fixtures import assert_simplex_rows, classification_recipe, materialise

from entlearn import ClassificationHead, Coupling, Input, Network, Recipe
from entlearn.primitives.normalise import _tiny


class TestCategoricalFit:
    def test_fits_and_predicts_hard_categorical_features(self) -> None:
        X_cont = torch.empty(6, 0, dtype=torch.float64)
        X_cat = (torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int64),)
        y = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int64)
        recipe = classification_recipe(Input(K=2, epsilon=0.05))

        network = Network.fit(recipe, X_cont, y, X_cat=X_cat, max_iter=20, seed=7)
        probabilities = network.predict(X_cont, X_cat=X_cat)

        assert_simplex_rows(probabilities)
        assert torch.equal(probabilities.argmax(dim=1), y)

    def test_distribution_categorical_features_match_hard_codes(self) -> None:
        X_cont = torch.tensor([[0.0], [0.1], [0.2], [0.8], [0.9], [1.0]], dtype=torch.float64)
        hard = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int64)
        distribution = torch.nn.functional.one_hot(hard).to(torch.float64)
        original = distribution.clone()
        recipe = classification_recipe(Input(K=2, epsilon=0.05))

        hard_network = Network.fit(recipe, X_cont, hard, X_cat=(hard,), max_iter=20, seed=7)
        distribution_network = Network.fit(
            recipe,
            X_cont,
            hard,
            X_cat=(distribution,),
            max_iter=20,
            seed=7,
        )

        torch.testing.assert_close(
            distribution_network.predict(X_cont, X_cat=(distribution,)),
            hard_network.predict(X_cont, X_cat=(hard,)),
        )
        assert torch.equal(distribution, original)

    def test_prediction_accepts_hard_codes_within_the_fitted_cardinality(self) -> None:
        X_cont = torch.empty(6, 0, dtype=torch.float64)
        X_cat = (torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int64),)
        y = torch.tensor([0, 0, 1, 1, 1, 1], dtype=torch.int64)
        recipe = classification_recipe(Input(K=2, epsilon=0.05))
        network = Network.fit(recipe, X_cont, y, X_cat=X_cat, max_iter=5, seed=3)
        # Code 2 is the largest fitted code.
        query = (torch.tensor([0, 1, 2], dtype=torch.int64),)

        probabilities = network.predict(
            torch.empty(3, 0, dtype=torch.float64),
            X_cat=query,
        )

        assert probabilities.shape == (3, 2)
        assert_simplex_rows(probabilities)

    @FIXED_BUDGET
    @pytest.mark.parametrize(
        "query,message",
        (
            pytest.param(
                torch.tensor([0, 3], dtype=torch.int64),
                r"X_cat\[0\] holds code 3",
                id="code-out-of-range",
            ),
            pytest.param(
                torch.ones(2, 2, dtype=torch.float64) / 2,
                r"\(T, M_i\) computation-dtype",
                id="wrong-distribution-width",
            ),
        ),
    )
    def test_prediction_rejects_categorical_features_outside_the_fitted_cardinality(
        self,
        query: torch.Tensor,
        message: str,
    ) -> None:
        X_cont = torch.empty(6, 0, dtype=torch.float64)
        X_cat = (torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int64),)
        y = torch.tensor([0, 0, 1, 1, 1, 1], dtype=torch.int64)
        network = Network.fit(classification_recipe(), X_cont, y, X_cat=X_cat, max_iter=1)

        with pytest.raises(ValueError, match=message):
            network.predict(torch.empty(2, 0, dtype=torch.float64), X_cat=(query,))

    @FIXED_BUDGET
    def test_initial_state_preserves_categorical_width_when_fit_rows_omit_a_level(self) -> None:
        recipe = classification_recipe()
        initial_state = Network.initialise(
            recipe,
            torch.empty(6, 0, dtype=torch.float64),
            torch.tensor([0, 0, 1, 1, 1, 1], dtype=torch.int64),
            X_cat=(torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int64),),
            seed=3,
        )
        X_cont = torch.empty(4, 0, dtype=torch.float64)
        X_cat = (torch.tensor([0, 0, 1, 1], dtype=torch.int64),)
        y = torch.tensor([0, 0, 1, 1], dtype=torch.int64)

        network = Network.fit(
            recipe,
            X_cont,
            y,
            X_cat=X_cat,
            initial_state=initial_state,
            max_iter=1,
        )

        assert network.schema.M_cat == (3,)
        assert network.predict(
            torch.empty(1, 0, dtype=torch.float64),
            X_cat=(torch.tensor([2], dtype=torch.int64),),
        ).shape == (1, 2)


def _noisy_categorical_data(
    noise_features: int,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], torch.Tensor]:
    """Return three clusters of 10, 20 and 10 rows, labelled 0, 1 and 0.

    Two continuous features and two three-level categorical features carry the cluster;
    ``noise_features`` further three-level categorical features are uniform noise.
    """
    # Draw on the CPU stream so every device lane fits the same data.
    generator = torch.Generator().manual_seed(0)
    ids = torch.repeat_interleave(torch.arange(3), torch.tensor([10, 20, 10]))
    centres = torch.tensor([[0.15, 0.1], [0.5, 0.5], [0.85, 0.9]], dtype=DTYPE)
    X_cont = centres[ids] + 0.05 * torch.randn(40, 2, dtype=DTYPE, generator=generator)
    noise = torch.randint(3, (noise_features, 40), generator=generator)
    X_cat = tuple(feature.to(DEVICE) for feature in (ids, ids.clone(), *noise))
    y = torch.tensor([0, 1, 0])[ids]
    return X_cont.clamp_(0.0, 1.0).to(DEVICE), X_cat, y.to(DEVICE)


# delta_cat = 1 prices the 0/1-scaled continuous features against the categorical ones as
# the fixture's informative-feature ranking assumes.
_LEARNED_WEIGHTS = Recipe.chain(
    Input(K=3, epsilon=0.015, epsilon_D=0.01, epsilon_T=0.15, delta_cat=1.0),
    ClassificationHead(Coupling.M),
)


class TestCategoricalStart:
    def test_a_fresh_fit_starts_from_the_level_mix_of_the_first_affiliations(self) -> None:
        X_cont, X_cat, y = _noisy_categorical_data(noise_features=2)
        state = Network.initialise(_LEARNED_WEIGHTS, X_cont, y, X_cat=X_cat, seed=0)

        block = materialise(
            _LEARNED_WEIGHTS, X_cont, y, X_cat=X_cat, initial_state=state
        ).graph.input

        weighted_gamma = block.gamma * block.instance_weights.unsqueeze(1)
        for feature, seeded, started in zip(
            X_cat,
            state.input_geometry.categorical_centroids,
            block.categorical_centroids,
            strict=True,
        ):
            levels = torch.nn.functional.one_hot(feature, 3).to(DTYPE)
            expected = weighted_gamma.T @ levels / weighted_gamma.sum(dim=0).unsqueeze(1)
            torch.testing.assert_close(started, expected)
            assert bool(((seeded == 0) | (seeded == 1)).all())
        assert not all(
            bool(((started == 0) | (started == 1)).all()) for started in block.categorical_centroids
        )

    def test_captured_geometry_keeps_its_categorical_centroids(self) -> None:
        X_cont, X_cat, y = _noisy_categorical_data(noise_features=2)
        fitted = Network.fit(_LEARNED_WEIGHTS, X_cont, y, X_cat=X_cat, seed=0)
        captured = fitted.capture_current_state()

        block = materialise(
            _LEARNED_WEIGHTS, X_cont, y, X_cat=X_cat, initial_state=captured
        ).graph.input

        for kept, stored in zip(
            block.categorical_centroids,
            captured.input_geometry.categorical_centroids,
            strict=True,
        ):
            assert torch.equal(kept, stored)

    @pytest.mark.parametrize("seed", by_depth((0,), (0, 1), (0, 1, 2, 3)))
    def test_noise_levels_do_not_collapse_the_learned_instance_weights(self, seed: int) -> None:
        X_cont, X_cat, y = _noisy_categorical_data(noise_features=14)

        network = Network.fit(_LEARNED_WEIGHTS, X_cont, y, X_cat=X_cat, seed=seed)

        # A one-hot categorical start would leave all the instance weight on the three
        # seed rows and the feature weights uniform.
        weights = network.inspect("training_instance_weights")["input"]
        effective_rows = torch.exp(-(weights * weights.clamp_min(_tiny(DTYPE)).log()).sum())
        assert float(effective_rows) > 20
        feature_weights = network.inspect("feature_weights")["input"]
        assert set(feature_weights.topk(4).indices.tolist()) == {0, 1, 2, 3}


def _delta_cat(network: Network) -> float | None:
    block = network.recipe.blocks[0]
    assert isinstance(block, Input)
    return block.delta_cat


@FIXED_BUDGET
class TestDeltaCatDefault:
    def test_an_omitted_value_is_the_helper_guess_on_the_weighted_training_rows(self) -> None:
        from entlearn.helpers import delta_cat_guess

        X_cont, X_cat, y = _noisy_categorical_data(noise_features=2)
        weights = torch.linspace(1.0, 3.0, 40, dtype=DTYPE, device=DEVICE)

        network = Network.fit(
            classification_recipe(Input(K=3)),
            X_cont,
            y,
            X_cat=X_cat,
            sample_weights=weights,
            max_iter=1,
        )

        expected = delta_cat_guess(
            X_cont.cpu().numpy(),
            [feature.cpu().numpy() for feature in X_cat],
            sample_weights=weights.cpu().numpy(),
        )
        # Both compute in the data dtype from the same values.
        assert _delta_cat(network) == pytest.approx(expected, rel=16 * torch.finfo(DTYPE).eps)
        unweighted = Network.fit(
            classification_recipe(Input(K=3)), X_cont, y, X_cat=X_cat, max_iter=1
        )
        assert _delta_cat(unweighted) != pytest.approx(expected, rel=1e-3)

    def test_level_distributions_give_the_guess_of_their_codes(self) -> None:
        X_cont, X_cat, y = _noisy_categorical_data(noise_features=2)
        distributions = tuple(torch.nn.functional.one_hot(feature).to(DTYPE) for feature in X_cat)
        recipe = classification_recipe(Input(K=3))

        hard = Network.fit(recipe, X_cont, y, X_cat=X_cat, max_iter=1)
        soft = Network.fit(recipe, X_cont, y, X_cat=distributions, max_iter=1)

        assert _delta_cat(soft) == pytest.approx(_delta_cat(hard), rel=16 * torch.finfo(DTYPE).eps)

    @pytest.mark.parametrize(
        ("delta_cat", "constant", "expected"),
        [
            pytest.param(0.5, False, 0.5, id="supplied"),
            pytest.param(None, True, 1.0, id="no-varying-continuous-feature"),
        ],
    )
    def test_a_supplied_or_undefined_value_is_not_guessed(
        self, delta_cat: float | None, constant: bool, expected: float
    ) -> None:
        X_cont, X_cat, y = _noisy_categorical_data(noise_features=0)
        if constant:
            X_cont = torch.full_like(X_cont, 0.5)

        network = Network.fit(
            classification_recipe(Input(K=3, delta_cat=delta_cat)),
            X_cont,
            y,
            X_cat=X_cat,
            max_iter=1,
        )

        assert _delta_cat(network) == expected

    def test_continuous_only_data_leave_it_omitted(self) -> None:
        X_cont, _, y = _noisy_categorical_data(noise_features=0)
        recipe = classification_recipe(Input(K=3))

        assert Network.fit(recipe, X_cont, y, max_iter=1).recipe == recipe

    def test_fine_tuning_to_an_omitted_value_guesses_it_from_the_new_rows(self) -> None:
        X_cont, X_cat, y = _noisy_categorical_data(noise_features=2)
        recipe = classification_recipe(Input(K=3, delta_cat=0.5))
        source = Network.fit(recipe, X_cont, y, X_cat=X_cat, max_iter=1)
        guessed = Network.fit(classification_recipe(Input(K=3)), X_cont, y, X_cat=X_cat, max_iter=1)

        tuned = source.fine_tune(
            X_cont,
            y,
            X_cat=X_cat,
            recipe=recipe.replace_block("input", delta_cat=None),
            max_iter=1,
        )

        assert _delta_cat(tuned) == _delta_cat(guessed) != 0.5
