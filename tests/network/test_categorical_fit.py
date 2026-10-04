"""Categorical classification through the public Network lifecycle."""

import pytest
import torch
from conftest import DEVICE, DTYPE, by_depth
from network._fixtures import assert_simplex_rows, classification_recipe, materialise

from entlearn import ClassificationHead, Coupling, Input, Network, Recipe


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
        query = (torch.tensor([0, 1, 0], dtype=torch.int64),)

        probabilities = network.predict(
            torch.empty(3, 0, dtype=torch.float64),
            X_cat=query,
        )

        assert probabilities.shape == (3, 2)
        assert_simplex_rows(probabilities)

    @pytest.mark.parametrize(
        "query",
        (
            pytest.param(torch.tensor([0, 3], dtype=torch.int64), id="code-out-of-range"),
            pytest.param(torch.ones(2, 2, dtype=torch.float64) / 2, id="wrong-distribution-width"),
        ),
    )
    def test_prediction_rejects_categorical_features_outside_the_fitted_cardinality(
        self,
        query: torch.Tensor,
    ) -> None:
        X_cont = torch.empty(6, 0, dtype=torch.float64)
        X_cat = (torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int64),)
        y = torch.tensor([0, 0, 1, 1, 1, 1], dtype=torch.int64)
        network = Network.fit(classification_recipe(), X_cont, y, X_cat=X_cat, max_iter=1)

        with pytest.raises(ValueError):
            network.predict(torch.empty(2, 0, dtype=torch.float64), X_cat=(query,))

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


_LEARNED_WEIGHTS = Recipe.chain(
    Input(K=3, epsilon=0.015, epsilon_D=0.01, epsilon_T=0.15),
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
        effective_rows = torch.exp(
            -(weights * weights.clamp_min(torch.finfo(DTYPE).tiny).log()).sum()
        )
        assert float(effective_rows) > 20
        feature_weights = network.inspect("feature_weights")["input"]
        assert set(feature_weights.topk(4).indices.tolist()) == {0, 1, 2, 3}
