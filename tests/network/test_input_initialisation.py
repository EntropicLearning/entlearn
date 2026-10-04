"""Behaviour tests for input initialisation."""

from collections import Counter

import pytest
import torch
from conftest import DEVICE, DTYPE
from network._fixtures import (
    classification_recipe,
    continuous_classification_data,
)

from entlearn import (
    InitialState,
    Input,
    Network,
    Recipe,
    RegressionHead,
)
from entlearn.network.initialisation.balanced import allocate_balanced_seeds
from entlearn.network.initialisation.centroids import (
    _greedy_candidate,
    _uniform_unchosen_pick,
    _weighted_choice,
)


def _classification_recipe(input_block: Input):
    return classification_recipe(input_block, n_classes=2)


def _initialise(input_block: Input, **options) -> InitialState:
    """Initialise a two-class Recipe with ``input_block`` on the square fixture data."""
    X_cont, y = continuous_classification_data()
    return Network.initialise(_classification_recipe(input_block), X_cont, y, **options)


def _seeding_potential(
    X_cont: torch.Tensor,
    centroids: torch.Tensor,
    sample_weights: torch.Tensor,
) -> torch.Tensor:
    nearest = torch.cdist(X_cont, centroids).square().amin(dim=1)
    return (sample_weights * nearest).sum()


def _selected_class_counts(state, X_cont, labels) -> Counter[int]:
    matches = (state.input_geometry.continuous_centroids[:, None] == X_cont[None]).all(dim=2)
    assert (matches.sum(dim=1) == 1).all()
    rows = matches.to(torch.int64).argmax(dim=1)
    return Counter(labels[rows].tolist())


class TestFeatureWeightInitialisation:
    def test_uniform_weights_are_the_default(self):
        state = _initialise(Input(K=2), seed=17)

        torch.testing.assert_close(
            state.input_geometry.feature_weights,
            torch.full((2,), 0.5, dtype=torch.float64),
            rtol=0.0,
            atol=0.0,
        )

    def test_logistic_normal_weights_are_deterministic_and_non_uniform(self):
        first = _initialise(Input(K=2, W_std=0.8), seed=17)
        repeated = _initialise(Input(K=2, W_std=0.8), seed=17)

        assert first.input_geometry.feature_weights is not None
        assert repeated.input_geometry.feature_weights is not None
        torch.testing.assert_close(
            first.input_geometry.feature_weights.sum(), torch.tensor(1.0, dtype=torch.float64)
        )
        assert torch.equal(
            first.input_geometry.feature_weights, repeated.input_geometry.feature_weights
        )
        assert not torch.equal(
            first.input_geometry.feature_weights,
            torch.full((2,), 0.5, dtype=torch.float64),
        )

    def test_supplied_weights_replace_the_draw_and_do_not_alias_the_caller(self):
        supplied = torch.tensor([3.0, 1.0], dtype=torch.float64)

        state = _initialise(Input(K=2), feature_weights=supplied, seed=17)
        supplied.fill_(0.5)

        assert state.input_geometry.feature_weights is not None
        torch.testing.assert_close(
            state.input_geometry.feature_weights,
            torch.tensor([0.75, 0.25], dtype=torch.float64),
            rtol=0.0,
            atol=0.0,
        )

    def test_supplied_weights_and_a_positive_draw_scale_are_mutually_exclusive(self):
        with pytest.raises(ValueError, match="feature_weights and a positive W_std"):
            _initialise(Input(K=2, W_std=0.8), feature_weights=torch.ones(2, dtype=torch.float64))

    @pytest.mark.parametrize(
        "weights",
        (
            pytest.param(torch.tensor([1.0]), id="wrong-width"),
            pytest.param(torch.tensor([0.0, 0.0]), id="zero-mass"),
            pytest.param(torch.tensor([-1.0, 2.0]), id="negative"),
            pytest.param(torch.tensor([float("nan"), 1.0]), id="non-finite"),
        ),
    )
    def test_supplied_weights_must_define_a_valid_feature_simplex(self, weights):
        with pytest.raises(ValueError, match="feature_weights"):
            _initialise(Input(K=2), feature_weights=weights.to(dtype=torch.float64))


class TestGlobalCentroidInitialisation:
    def test_greedy_selection_reduces_the_candidate_potential(self):
        origin = torch.zeros(20, 2, dtype=torch.float64)
        hub = torch.tensor([0.5, 0.0], dtype=torch.float64).expand(6, 2).clone()
        trap = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        X_cont = torch.cat((origin, hub, trap))
        y = torch.arange(X_cont.shape[0]) % 2
        sample_weights = torch.ones(X_cont.shape[0], dtype=torch.float64)

        standard = Network.initialise(
            _classification_recipe(Input(K=2)),
            X_cont,
            y,
            sample_weights=sample_weights,
            seed=4,
        )
        greedy = Network.initialise(
            _classification_recipe(
                Input(
                    K=2,
                    centroid_strategy="greedy-kmeans++",
                    greedy_candidates=2,
                )
            ),
            X_cont,
            y,
            sample_weights=sample_weights,
            seed=4,
        )

        assert _seeding_potential(
            X_cont, greedy.input_geometry.continuous_centroids, sample_weights
        ) < _seeding_potential(X_cont, standard.input_geometry.continuous_centroids, sample_weights)

    def test_fixed_greedy_candidates_reuse_a_wider_selection_prefix(self):
        generator = torch.Generator().manual_seed(8)
        X_cont = torch.rand(24, 3, generator=generator, dtype=torch.float64)
        y = torch.arange(24) % 2
        narrow = _classification_recipe(
            Input(
                K=3,
                centroid_strategy="greedy-kmeans++",
                greedy_candidates=3,
            )
        )
        wide = _classification_recipe(
            Input(
                K=6,
                centroid_strategy="greedy-kmeans++",
                greedy_candidates=3,
            )
        )

        narrow_state = Network.initialise(narrow, X_cont, y, seed=12)
        wide_state = Network.initialise(wide, X_cont, y, seed=12)

        assert narrow_state.input_geometry.prefix_reusable
        assert torch.equal(
            narrow_state.input_geometry.continuous_centroids,
            wide_state.input_geometry.continuous_centroids[: narrow_state.input_geometry.K_active],
        )

    def test_the_default_greedy_candidate_count_is_two_plus_floor_log_k(self):
        X_cont = torch.rand(40, 3, generator=torch.Generator().manual_seed(8), dtype=DTYPE)
        X_cont = X_cont.to(DEVICE)
        y = torch.arange(40, device=DEVICE) % 2

        def centroids(**control):
            recipe = _classification_recipe(
                Input(K=8, centroid_strategy="greedy-kmeans++", **control)
            )
            state = Network.initialise(recipe, X_cont, y, seed=12)
            return state.input_geometry.continuous_centroids

        # 2 + floor(log 8) = 4 candidates per seed.
        assert torch.equal(centroids(), centroids(greedy_candidates=4))

    def test_k_dependent_greedy_candidates_reject_prefix_reuse(self):
        state = _initialise(Input(K=2, centroid_strategy="greedy-kmeans++"), seed=5)

        assert not state.input_geometry.prefix_reusable

    def test_sample_weights_control_the_first_centroid(self):
        X_cont, y = continuous_classification_data()
        sample_weights = torch.tensor([1e-30, 1e-30, 1.0, 1e-30], dtype=torch.float64)

        state = Network.initialise(
            _classification_recipe(Input(K=1)),
            X_cont,
            y,
            sample_weights=sample_weights,
            seed=3,
        )

        assert torch.equal(state.input_geometry.continuous_centroids[0], X_cont[2])


class TestWeightedRowDraw:
    def test_zero_weight_rows_are_never_drawn_when_the_total_mass_underflows(self):
        # The k-means++ mass left after the chosen rows are zeroed can be subnormal, and
        # scaling a uniform draw by it then underflows to exactly zero. Every such draw
        # must still land on the one row that carries weight.
        smallest = torch.nextafter(torch.zeros((), dtype=DTYPE), torch.ones((), dtype=DTYPE))
        weights = torch.tensor([0.0, 1.0, 0.0], dtype=DTYPE, device=DEVICE) * smallest
        generator = torch.Generator(device=DEVICE).manual_seed(0)

        drawn = _weighted_choice(weights, generator, 64)

        assert torch.equal(drawn, torch.ones_like(drawn))


class TestDegenerateSeedFallback:
    def test_every_unchosen_row_is_reachable_and_only_the_generator_decides(self):
        # The fallback runs once the weighted draw has no mass left, so it must spread
        # over every unchosen row and still be reproducible from the supplied generator.
        chosen = torch.tensor([0], dtype=torch.int64, device=DEVICE)

        def picks(seed: int) -> list[int]:
            generator = torch.Generator(device=DEVICE).manual_seed(seed)
            return [_uniform_unchosen_pick(chosen, 1, 6, generator, DTYPE) for _ in range(64)]

        first = picks(11)
        assert set(first) == {1, 2, 3, 4, 5}
        torch.rand(1, device=DEVICE)
        assert picks(11) == first

    def test_the_first_greedy_candidate_wins_a_tie(self):
        # Identical rows give every candidate the same potential, so the draw order alone
        # decides and the documented first-wins rule must hold.
        X = torch.full((3, 1), 0.5, dtype=DTYPE, device=DEVICE)

        best = _greedy_candidate(
            X,
            [],
            [],
            torch.ones(1, dtype=DTYPE, device=DEVICE),
            torch.zeros(0, dtype=DTYPE, device=DEVICE),
            torch.ones(3, dtype=DTYPE, device=DEVICE),
            torch.zeros(3, dtype=DTYPE, device=DEVICE),
            torch.tensor([2, 0, 1], dtype=torch.int64, device=DEVICE),
        )

        assert best == 2


class TestBalancedCentroidInitialisation:
    @pytest.mark.parametrize(
        ("strategy", "greedy_candidates"),
        (
            pytest.param("kmeans++", None, id="standard"),
            pytest.param("greedy-kmeans++", 2, id="greedy"),
        ),
    )
    @pytest.mark.timeout(10, method="signal")
    def test_each_selector_uses_the_balanced_class_allocation(self, strategy, greedy_candidates):
        X_cont = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [0.0, 2.0],
                [1.0, 3.0],
                [1.0, 4.0],
                [1.0, 5.0],
            ],
            dtype=torch.float64,
        )
        y = torch.tensor([0, 0, 0, 1, 1, 1])
        X_cont.div_(8)
        recipe = _classification_recipe(
            Input(
                K=4,
                balanced=True,
                centroid_strategy=strategy,
                greedy_candidates=greedy_candidates,
            )
        )

        state = Network.initialise(recipe, X_cont, y, seed=4)

        assert _selected_class_counts(state, X_cont, y) == Counter({0: 2, 1: 2})

    @pytest.mark.timeout(10, method="signal")
    def test_round_robin_skips_an_exhausted_class(self):
        X_cont = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [1.0, 2.0],
                [1.0, 3.0],
                [1.0, 4.0],
                [2.0, 5.0],
                [2.0, 6.0],
                [2.0, 7.0],
                [2.0, 8.0],
            ],
            dtype=torch.float64,
        )
        y = X_cont[:, 0].to(dtype=torch.int64)
        X_cont.div_(8)
        sample_weights = torch.tensor(
            [100.0, 5.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0],
            dtype=torch.float64,
        )

        recipe = classification_recipe(Input(K=7, balanced=True), n_classes=3)
        state = Network.initialise(
            recipe,
            X_cont,
            y,
            sample_weights=sample_weights,
            seed=2,
        )

        assert _selected_class_counts(state, X_cont, y) == Counter({1: 3, 2: 3, 0: 1})

    @pytest.mark.timeout(10, method="signal")
    def test_a_full_class_is_skipped_without_ending_the_pass(self):
        # Masses 9 > 5 > 3 with capacities 3, 1, 3: the full middle class must not end the pass.
        labels = torch.tensor([0, 0, 0, 1, 2, 2, 2], device=DEVICE)
        weights = torch.tensor([3.0, 3.0, 3.0, 5.0, 1.0, 1.0, 1.0], dtype=DTYPE, device=DEVICE)
        eligible = torch.ones_like(labels, dtype=torch.bool)

        allocation = allocate_balanced_seeds(labels, eligible, weights, 5)

        assert allocation == ((0, 2), (1, 1), (2, 2))

    def test_rejects_fewer_centroids_than_active_classes(self):
        X_cont = torch.tensor([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], dtype=torch.float64)
        y = torch.tensor([0, 1, 2])
        recipe = classification_recipe(Input(K=2, balanced=True), n_classes=3)

        with pytest.raises(ValueError, match="active classes"):
            Network.initialise(recipe, X_cont, y)

    def test_distribution_ties_use_the_lowest_class_and_unlabelled_rows_are_excluded(self):
        X_cont = torch.arange(5, dtype=torch.float64)[:, None] / 4
        y = torch.tensor(
            [
                [0.5, 0.5],
                [0.0, 1.0],
                [0.0, 0.0],
                [0.5, 0.5],
                [0.0, 1.0],
            ],
            dtype=torch.float64,
        )

        state = Network.initialise(
            _classification_recipe(Input(K=2, balanced=True)),
            X_cont,
            y,
            seed=3,
        )

        selected = set(state.input_geometry.continuous_centroids[:, 0].tolist())
        assert 0.5 not in selected
        assert len(selected & {0.0, 0.75}) == 1
        assert len(selected & {0.25, 1.0}) == 1

    @pytest.mark.timeout(10, method="signal")
    def test_class_weights_do_not_change_balanced_selection(self):
        X_cont = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [0.0, 2.0],
                [0.0, 3.0],
                [1.0, 4.0],
                [1.0, 5.0],
                [1.0, 6.0],
                [1.0, 7.0],
            ],
            dtype=torch.float64,
        )
        y = X_cont[:, 0].to(dtype=torch.int64)
        X_cont.div_(8)
        sample_weights = torch.tensor([4.0, 4.0, 4.0, 4.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
        recipe = _classification_recipe(Input(K=5, balanced=True))

        ordinary = Network.initialise(recipe, X_cont, y, sample_weights=sample_weights, seed=9)
        reweighted = Network.initialise(
            recipe,
            X_cont,
            y,
            sample_weights=sample_weights,
            class_weights=torch.tensor([1.0, 1000.0], dtype=torch.float64),
            seed=9,
        )

        assert torch.equal(
            ordinary.input_geometry.continuous_centroids,
            reweighted.input_geometry.continuous_centroids,
        )
        assert _selected_class_counts(ordinary, X_cont, y) == Counter({0: 3, 1: 2})

    @pytest.mark.timeout(10, method="signal")
    def test_equal_class_mass_gives_the_extra_centroid_to_the_lower_class_code(self):
        X_cont = torch.tensor(
            [[0.0, 0.0], [0.0, 1.0], [1.0, 2.0], [1.0, 3.0]],
            dtype=torch.float64,
        )
        y = X_cont[:, 0].to(dtype=torch.int64)
        X_cont.div_(4)

        state = Network.initialise(
            _classification_recipe(Input(K=3, balanced=True)),
            X_cont,
            y,
            seed=6,
        )

        assert _selected_class_counts(state, X_cont, y) == Counter({0: 2, 1: 1})

    def test_regression_rejects_balanced_allocation(self):
        with pytest.raises(ValueError, match="balanced"):
            Recipe.chain(Input(K=2, balanced=True), RegressionHead())


class TestSuppliedCentroidInitialisation:
    def test_mixed_centroids_are_normalised_used_and_copied(self):
        X_cont = torch.tensor([[0.0], [0.25], [0.5], [0.75]], dtype=torch.float64)
        X_cat = [torch.tensor([0, 1, 0, 1])]
        y = torch.tensor([0, 0, 1, 1])
        continuous = torch.tensor([[0.125], [0.875]], dtype=torch.float64)
        categorical = [torch.tensor([[3.0, 1.0], [1.0, 3.0]], dtype=torch.float64)]

        with pytest.warns(UserWarning, match="categorical_centroids"):
            state = Network.initialise(
                _classification_recipe(Input(K=2)),
                X_cont,
                y,
                X_cat=X_cat,
                continuous_centroids=continuous,
                categorical_centroids=categorical,
                seed=8,
            )
        continuous.fill_(0.0)
        categorical[0].fill_(0.5)

        torch.testing.assert_close(
            state.input_geometry.continuous_centroids,
            torch.tensor([[0.125], [0.875]], dtype=torch.float64),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            state.input_geometry.categorical_centroids[0],
            torch.tensor([[0.75, 0.25], [0.25, 0.75]], dtype=torch.float64),
            rtol=0.0,
            atol=0.0,
        )
        assert state.input_geometry.prefix_reusable

    @pytest.mark.parametrize(
        "input_block",
        (
            pytest.param(
                Input(
                    K=2,
                    centroid_strategy="greedy-kmeans++",
                    greedy_candidates=2,
                ),
                id="greedy",
            ),
            pytest.param(Input(K=2, balanced=True), id="balanced"),
        ),
    )
    def test_supplied_centroids_and_selection_controls_are_mutually_exclusive(self, input_block):
        with pytest.raises(ValueError, match=r"supplied centroids.*mutually exclusive"):
            _initialise(input_block, continuous_centroids=torch.zeros(2, 2, dtype=torch.float64))

    def test_mixed_input_rejects_partial_supplied_geometry(self):
        X_cont = torch.tensor([[0.0], [0.25], [0.5], [0.75]], dtype=torch.float64)
        X_cat = [torch.tensor([0, 1, 0, 1])]
        y = torch.tensor([0, 0, 1, 1])

        with pytest.raises(ValueError, match="requires both"):
            Network.initialise(
                _classification_recipe(Input(K=2)),
                X_cont,
                y,
                X_cat=X_cat,
                continuous_centroids=torch.zeros(2, 1, dtype=torch.float64),
            )

    @pytest.mark.parametrize(
        "continuous",
        (
            pytest.param(torch.zeros(1, 2, dtype=torch.float64), id="wrong-size"),
            pytest.param(
                torch.tensor([[0.0, float("inf")], [1.0, 1.0]], dtype=torch.float64),
                id="non-finite",
            ),
        ),
    )
    def test_supplied_continuous_centroids_must_be_valid(self, continuous):
        with pytest.raises(ValueError, match="continuous_centroids"):
            _initialise(Input(K=2), continuous_centroids=continuous)

    def test_integer_continuous_centroids_are_rejected_even_with_a_computation_dtype(self):
        with pytest.raises(ValueError, match="continuous_centroids must be a floating tensor with"):
            _initialise(
                Input(K=2),
                continuous_centroids=torch.zeros(2, 2, dtype=torch.int64),
                computation_dtype=torch.float64,
            )

    @pytest.mark.parametrize(
        ("categorical", "message"),
        (
            pytest.param(
                [torch.tensor([[2.0, -1.0], [1.0, 1.0]], dtype=torch.float64)],
                "must be non-negative",
                id="negative",
            ),
            pytest.param(
                [torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)],
                "rows must have finite positive mass",
                id="zero-mass",
            ),
            pytest.param(
                [torch.tensor([[1.0, float("nan")], [1.0, 1.0]], dtype=torch.float64)],
                "must contain only finite values",
                id="non-finite",
            ),
        ),
    )
    def test_supplied_categorical_centroids_must_be_valid(self, categorical, message):
        X_cont = torch.empty(4, 0, dtype=torch.float64)
        X_cat = [torch.tensor([0, 1, 0, 1])]
        y = torch.tensor([0, 0, 1, 1])

        with pytest.raises(ValueError, match=rf"categorical_centroids\[0\] {message}"):
            Network.initialise(
                _classification_recipe(Input(K=2)),
                X_cont,
                y,
                X_cat=X_cat,
                categorical_centroids=categorical,
            )

    @pytest.mark.parametrize(
        ("continuous_input", "supplied", "message"),
        (
            pytest.param(
                False,
                lambda c: {"continuous_centroids": c},
                "continuous_centroids were supplied for an input without continuous data",
                id="no-continuous-features",
            ),
            pytest.param(
                True,
                lambda c: {"categorical_centroids": [c]},
                "categorical_centroids were supplied for an input without categorical data",
                id="no-categorical-features",
            ),
            pytest.param(
                False,
                lambda c: {"categorical_centroids": "ab"},
                "categorical_centroids must be a sequence of tensors",
                id="string",
            ),
            pytest.param(
                False,
                lambda c: {"categorical_centroids": c},
                "categorical_centroids must be a sequence of tensors",
                id="tensor",
            ),
            pytest.param(
                False,
                lambda c: {"categorical_centroids": [c, c]},
                r"categorical_centroids has 2 feature\(s\), but the input has 1",
                id="feature-count",
            ),
            pytest.param(
                True,
                lambda c: {"continuous_centroids": c[:, :1].tolist()},
                "continuous_centroids must be a floating tensor$",
                id="non-tensor",
            ),
        ),
    )
    def test_supplied_centroids_must_match_the_input_features(
        self, continuous_input, supplied, message
    ):
        X_cont = torch.tensor([[0.0], [0.25], [0.5], [0.75]], dtype=DTYPE, device=DEVICE)
        X_cat = [torch.tensor([0, 1, 0, 1], device=DEVICE)]
        if not continuous_input:
            X_cont = X_cont[:, :0]
        centroids = torch.full((2, 2), 0.5, dtype=DTYPE, device=DEVICE)

        with pytest.raises(ValueError, match=message):
            Network.initialise(
                _classification_recipe(Input(K=2)),
                X_cont,
                torch.tensor([0, 0, 1, 1], device=DEVICE),
                X_cat=None if continuous_input else X_cat,
                **supplied(centroids),
            )

    def test_one_hot_categorical_centroids_are_accepted(self):
        one_hot = torch.eye(2, dtype=DTYPE, device=DEVICE)
        state = Network.initialise(
            _classification_recipe(Input(K=2)),
            torch.empty(4, 0, dtype=DTYPE, device=DEVICE),
            torch.tensor([0, 0, 1, 1], device=DEVICE),
            X_cat=[torch.tensor([0, 1, 0, 1], device=DEVICE)],
            categorical_centroids=[one_hot],
        )
        torch.testing.assert_close(
            state.input_geometry.categorical_centroids[0], one_hot, rtol=0, atol=0
        )

    def test_categorical_centroids_alone_start_a_fit_with_empty_continuous_geometry(self):
        recipe = _classification_recipe(Input(K=2))
        X_cont = torch.empty(4, 0, dtype=DTYPE, device=DEVICE)
        y = torch.tensor([0, 0, 1, 1], device=DEVICE)
        X_cat = [torch.tensor([0, 1, 0, 1], device=DEVICE)]
        supplied = torch.tensor([[0.75, 0.25], [0.25, 0.75]], dtype=DTYPE, device=DEVICE)

        state = Network.initialise(recipe, X_cont, y, X_cat=X_cat, categorical_centroids=[supplied])
        empty = state.input_geometry.continuous_centroids

        assert empty.shape == (2, 0)
        assert empty.dtype is DTYPE
        Network.fit(recipe, X_cont, y, X_cat=X_cat, initial_state=state, max_iter=2)

    def test_continuous_centroid_dtype_conversion_requires_an_explicit_override(self):
        with pytest.raises(ValueError, match=r"continuous_centroids.*computation dtype"):
            _initialise(Input(K=2), continuous_centroids=torch.zeros(2, 2, dtype=torch.float32))

    def test_categorical_centroid_dtype_conversion_requires_an_explicit_override(self):
        X_cont = torch.empty(4, 0, dtype=torch.float64)
        X_cat = [torch.tensor([0, 1, 0, 1])]
        y = torch.tensor([0, 0, 1, 1])

        with pytest.raises(ValueError, match=r"categorical_centroids.*computation dtype"):
            Network.initialise(
                _classification_recipe(Input(K=2)),
                X_cont,
                y,
                X_cat=X_cat,
                categorical_centroids=[torch.ones(2, 2, dtype=torch.float32)],
            )

    def test_explicit_computation_dtype_converts_supplied_centroids_once(self):
        supplied = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)

        state = _initialise(
            Input(K=2), continuous_centroids=supplied, computation_dtype=torch.float32
        )

        assert state.input_geometry.continuous_centroids.dtype is torch.float32
        torch.testing.assert_close(
            state.input_geometry.continuous_centroids,
            supplied.to(dtype=torch.float32),
            rtol=0.0,
            atol=0.0,
        )

    def test_explicit_computation_dtype_converts_supplied_categorical_centroids(self):
        other = torch.float32 if DTYPE is torch.float64 else torch.float64
        supplied = torch.tensor([[0.25, 0.75], [0.5, 0.5]], dtype=other, device=DEVICE)

        state = Network.initialise(
            _classification_recipe(Input(K=2)),
            torch.empty(4, 0, dtype=DTYPE, device=DEVICE),
            torch.tensor([0, 0, 1, 1], device=DEVICE),
            X_cat=[torch.tensor([0, 1, 0, 1], device=DEVICE)],
            categorical_centroids=[supplied],
            computation_dtype=DTYPE,
        )

        (converted,) = state.input_geometry.categorical_centroids
        assert converted.dtype is DTYPE
        torch.testing.assert_close(converted, supplied.to(dtype=DTYPE), rtol=0.0, atol=0.0)
