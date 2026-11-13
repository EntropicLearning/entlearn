"""Behaviour tests for input initialisation."""

import math
from collections import Counter

import pytest
import torch
from conftest import DEVICE, DTYPE, FIXED_BUDGET
from network._fixtures import (
    blobs,
    classification_recipe,
    continuous_classification_data,
)

from entlearn import (
    InitialState,
    Input,
    Network,
)
from entlearn.network.initialisation import centroids
from entlearn.network.initialisation.balanced import allocate_balanced_seeds
from entlearn.network.initialisation.centroids import (
    _cumsum_rows,
    _greedy_candidate,
    _weighted_draws,
    select_kmeanspp_seeds_,
)
from entlearn.primitives.distance import dissimilarity_to_rows


@pytest.fixture
def nondeterministic():
    """Allow the CUDA kernels whose results vary from run to run, as a user's session does."""
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(False)
    yield
    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


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
        ("weights", "reason"),
        (
            pytest.param(torch.tensor([1.0]), "length-2", id="wrong-width"),
            pytest.param(torch.tensor([0.0, 0.0]), "positive total mass", id="zero-mass"),
            pytest.param(torch.tensor([-1.0, 2.0]), "non-negative", id="negative"),
            pytest.param(torch.tensor([float("nan"), 1.0]), "only finite", id="non-finite"),
        ),
    )
    def test_supplied_weights_must_define_a_valid_feature_simplex(self, weights, reason):
        with pytest.raises(ValueError, match=f"feature_weights .*{reason}"):
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


class TestGreedyCandidate:
    def test_the_winning_candidate_gives_each_row_its_nearest_seed_dissimilarity(self):
        X = torch.tensor([[0.0], [0.1], [1.0]], dtype=DTYPE, device=DEVICE)
        d2 = torch.tensor([0.0, 0.01, 1.0], dtype=DTYPE, device=DEVICE)  # seed row 0 chosen
        dissimilarity = dissimilarity_to_rows(
            X,
            [],
            torch.ones(1, dtype=DTYPE, device=DEVICE),
            torch.zeros(0, dtype=DTYPE, device=DEVICE),
        )

        # Row 1 leaves the potential at 0.81, row 2 at d2[1] = 0.01.
        best, updated, column = _greedy_candidate(
            dissimilarity,
            torch.ones(3, dtype=DTYPE, device=DEVICE),
            d2[None],
            torch.tensor([[1, 2]], device=DEVICE),
        )

        assert best.tolist() == [[2]]
        assert torch.equal(updated, torch.tensor([[0.0, 0.01, 0.0]], dtype=DTYPE, device=DEVICE))
        torch.testing.assert_close(
            column, torch.tensor([[1.0, 0.81, 0.0]], dtype=DTYPE, device=DEVICE)
        )


class TestCategoricalCodeSeeding:
    def test_codes_seed_exactly_as_their_one_hot_distributions(self):
        # A code row's cross-entropy to a seed is a lookup, so seeding from the codes must
        # choose the same rows as seeding from their one-hot rows, and start the categorical
        # centroids at those rows' one-hot codes.
        g = torch.Generator().manual_seed(5)
        X_cont = torch.rand(300, 4, generator=g, dtype=DTYPE).to(DEVICE)
        codes = [torch.randint(0, m, (300,), generator=g).to(DEVICE) for m in (2, 5)]
        one_hot = [
            torch.nn.functional.one_hot(c, m).to(DTYPE) for c, m in zip(codes, (2, 5), strict=True)
        ]
        y = torch.arange(300, device=DEVICE) % 2
        recipe = _classification_recipe(Input(K=12, centroid_strategy="greedy-kmeans++", W_std=0.5))

        from_codes = Network.initialise(recipe, X_cont, y, X_cat=codes, seed=7).input_geometry
        from_rows = Network.initialise(recipe, X_cont, y, X_cat=one_hot, seed=7).input_geometry

        assert torch.equal(from_codes.continuous_centroids, from_rows.continuous_centroids)
        rows = (from_codes.continuous_centroids[:, None] == X_cont).all(dim=2).nonzero()[:, 1]
        for coded, rows_one_hot in zip(from_codes.categorical_centroids, one_hot, strict=True):
            assert torch.equal(coded, rows_one_hot[rows])


class TestWeightedRowDraw:
    @pytest.mark.parametrize("rows", (1, 3))
    @pytest.mark.parametrize("n", (1, 2, 1_000, 1_000_003))
    def test_the_cumulative_sums_repeat(self, nondeterministic, rows, n):
        generator = torch.Generator().manual_seed(n)
        weights = torch.rand(rows, n, generator=generator, dtype=DTYPE).to(DEVICE)

        cumulative = _cumsum_rows(weights)

        assert all(torch.equal(cumulative, _cumsum_rows(weights)) for _ in range(8))
        torch.testing.assert_close(
            cumulative.double().cpu(),
            weights.double().cpu().cumsum(dim=1),
            rtol=8 * math.isqrt(n) * torch.finfo(DTYPE).eps,
            atol=0,
        )

    # Cover both direct row scans and the square-based scan for long rows.
    # Separately summed row offsets can cause decreases at square-row boundaries.
    # A single input row also exercises the duplication that avoids CUDA's 1-D scan.
    @pytest.mark.parametrize("rows", (1, 3))
    @pytest.mark.parametrize("n", (3_000, 3_000_000))
    def test_the_cumulative_sums_never_fall_and_are_flat_at_zero_weights(
        self, nondeterministic, rows, n
    ):
        generator = torch.Generator().manual_seed(0)
        weights = torch.rand(rows, n, generator=generator, dtype=DTYPE)
        weights[torch.rand(rows, n, generator=generator) < 0.3] = 0.0

        steps = _cumsum_rows(weights.to(DEVICE)).diff(dim=1).cpu()

        assert (steps >= 0).all()
        assert (steps[weights[:, 1:] == 0] == 0).all()

    def test_zero_weight_rows_are_never_drawn_when_the_total_mass_underflows(self):
        # The k-means++ mass left after the chosen rows are zeroed can be subnormal, and
        # scaling a uniform draw by it then underflows to exactly zero. Every such draw
        # must still land on the one row that carries weight.
        smallest = torch.nextafter(torch.zeros((), dtype=DTYPE), torch.ones((), dtype=DTYPE))
        weights = torch.tensor([0.0, 1.0, 0.0], dtype=DTYPE, device=DEVICE) * smallest
        generator = torch.Generator(device=DEVICE).manual_seed(0)

        drawn = _weighted_draws(weights[None], [generator], 64)

        assert torch.equal(drawn, torch.ones_like(drawn))


def _draw_batched_seeds(seeds: list[int], n_candidates: int, rows: int = 400) -> torch.Tensor:
    """Draw 12 centroid rows per random seed using continuous and categorical-code features."""
    g = torch.Generator().manual_seed(9)
    X = torch.rand(rows, 3, generator=g, dtype=DTYPE).to(DEVICE)
    codes = torch.randint(0, 4, (rows,), generator=g).to(DEVICE)
    generators = [torch.Generator(device=DEVICE).manual_seed(seed) for seed in seeds]
    Wd = torch.full((4,), 0.25, dtype=DTYPE, device=DEVICE)
    weights = torch.rand(rows, generator=g, dtype=DTYPE).to(DEVICE)
    return select_kmeanspp_seeds_(
        None, X, [codes], 12, generators, Wd[:3], Wd[3:], weights, n_candidates=n_candidates
    )


class TestBatchedSeeding:
    @pytest.mark.parametrize("n_candidates", (1, 3), ids=("standard", "greedy"))
    def test_each_generator_draws_its_seeds_whatever_the_others_draw(self, n_candidates):
        # Generators seeded together share every product, so changing one generator's seed
        # must change its own seeds and leave every other generator's seeds as they were.
        first = _draw_batched_seeds([1, 2, 3], n_candidates)
        assert all(len(set(row.tolist())) == 12 for row in first)
        for changed in range(3):
            seeds = [1, 2, 3]
            seeds[changed] = 20
            other = _draw_batched_seeds(seeds, n_candidates)
            kept = [i for i in range(3) if i != changed]
            assert torch.equal(other[kept], first[kept])
            assert not torch.equal(other[changed], first[changed])

    @pytest.mark.parametrize("n_candidates", (1, 3), ids=("standard", "greedy"))
    def test_the_seeds_repeat_from_run_to_run(self, nondeterministic, n_candidates):
        # On CUDA without deterministic algorithms, a draw built on an operation that varies
        # from run to run (an atomic sum, say) moves seeds at this many rows.
        rows = 100_000 if str(DEVICE).startswith("cuda") else 400

        first = _draw_batched_seeds(list(range(16)), n_candidates, rows)
        repeated = [_draw_batched_seeds(list(range(16)), n_candidates, rows) for _ in range(4)]

        assert all(torch.equal(seeds, first) for seeds in repeated)

    def test_each_plain_seed_is_drawn_against_its_nearest_chosen_seed(self):
        # Three tight clusters on a line: once two hold a seed, only the third carries mass.
        # Drawing against the first seed alone would often put two seeds in one cluster.
        centres = torch.tensor([0.0, 0.5, 1.0], dtype=DTYPE).repeat_interleave(20)
        noise = 1e-3 * torch.rand(60, generator=torch.Generator().manual_seed(0), dtype=DTYPE)
        X = (centres + noise).to(DEVICE)[:, None]
        weights = torch.ones(60, dtype=DTYPE, device=DEVICE)
        generators = [torch.Generator(device=DEVICE).manual_seed(s) for s in range(64)]

        seeds = select_kmeanspp_seeds_(
            None, X, [], 3, generators, weights[:1], weights[:0], weights
        )

        assert all(sorted(row) == [0, 1, 2] for row in (seeds // 20).tolist())

    @FIXED_BUDGET
    def test_each_candidate_draws_with_its_own_feature_weights(self):
        # Random feature weights differ between candidates, so pricing a candidate's draws
        # with another candidate's weights would move its centroids away from its seed's.
        X, y = blobs(2, 0.1, rows_per_blob=50)
        recipe = _classification_recipe(Input(K=6, W_std=0.5))

        network = Network.fit(recipe, X, y, n_inits=3, retain="states", max_iter=1)

        outcomes = network.diagnostics.initialisation_outcomes
        assert outcomes is not None
        for state, outcome in zip(network.initial_states, outcomes, strict=True):
            expected = Network.initialise(recipe, X, y, seed=outcome.seed).input_geometry
            centroids = state.input_geometry.continuous_centroids
            assert torch.equal(centroids, expected.continuous_centroids)

    @FIXED_BUDGET
    def test_a_candidates_state_does_not_depend_on_the_number_of_initialisations(self, monkeypatch):
        # Kernels chosen by a product's width can round a candidate's dissimilarities
        # differently beside more candidates. An offset that grows with the width stands in
        # for that rounding, so the states agree only if every pass has the same width.
        priced = centroids.dissimilarity_to_rows

        def width_dependent(*args):
            dissimilarity = priced(*args)
            return lambda seeds: dissimilarity(seeds) + 0.05 * seeds.numel()

        monkeypatch.setattr(centroids, "dissimilarity_to_rows", width_dependent)
        X, y = blobs(2, 0.1, rows_per_blob=60)
        recipe = _classification_recipe(Input(K=6))

        def states(n_inits, seed=0):
            network = Network.fit(
                recipe, X, y, n_inits=n_inits, retain="states", max_iter=1, seed=seed
            )
            outcomes = network.diagnostics.initialisation_outcomes
            return [
                (outcome.seed, state.input_geometry.continuous_centroids)
                for outcome, state in zip(outcomes, network.initial_states, strict=True)
            ]

        two, ten = states(2), states(10)
        for (seed, drawn), (seed_ten, drawn_ten) in zip(two, ten[:2], strict=True):
            assert seed == seed_ten and torch.equal(drawn, drawn_ten)
        for seed, drawn in ten[:3]:  # replaying a candidate's seed alone draws its state
            ((replayed_seed, replayed),) = states(1, seed=seed)
            alone = Network.initialise(recipe, X, y, seed=seed).input_geometry
            assert replayed_seed == seed and torch.equal(replayed, drawn)
            assert torch.equal(alone.continuous_centroids, drawn)


class TestDegenerateSeedFallback:
    def test_every_unchosen_row_is_reachable_and_only_the_generators_decide(self):
        # Identical rows leave no weighted mass after the first seed, so the fallback must
        # spread each later seed over the rows not yet chosen and repeat from the same generators.
        X = torch.full((6, 1), 0.5, dtype=DTYPE, device=DEVICE)
        weights = torch.ones(6, dtype=DTYPE, device=DEVICE)

        def picks() -> torch.Tensor:
            generators = [torch.Generator(device=DEVICE).manual_seed(s) for s in range(64)]
            return select_kmeanspp_seeds_(
                None, X, [], 6, generators, weights[:1], weights[:0], weights
            )

        first = picks()
        assert all(sorted(row) == list(range(6)) for row in first.tolist())
        assert set(first[:, 1].tolist()) == set(range(6))
        torch.rand(1, device=DEVICE)
        assert torch.equal(picks(), first)

    def test_the_first_greedy_candidate_wins_a_tie(self):
        # Identical rows give every candidate the same potential, so the draw order alone
        # decides and the documented first-wins rule must hold.
        X = torch.full((3, 1), 0.5, dtype=DTYPE, device=DEVICE)
        dissimilarity = dissimilarity_to_rows(
            X,
            [],
            torch.ones(1, dtype=DTYPE, device=DEVICE),
            torch.zeros(0, dtype=DTYPE, device=DEVICE),
        )

        best, d2, column = _greedy_candidate(
            dissimilarity,
            torch.ones(3, dtype=DTYPE, device=DEVICE),
            torch.zeros(1, 3, dtype=DTYPE, device=DEVICE),
            torch.tensor([[2, 0, 1]], dtype=torch.int64, device=DEVICE),
        )

        assert best.tolist() == [[2]]
        assert torch.equal(d2, torch.zeros(1, 3, dtype=DTYPE, device=DEVICE))
        assert torch.equal(column, d2)


class TestSeedColumns:
    @pytest.mark.parametrize("n_candidates", [1, 3])
    def test_each_column_is_every_rows_dissimilarity_to_its_seed(self, n_candidates):
        # Connection seeding assigns rows by these columns, so each must be its seed's
        # dissimilarity, and asking for them must leave the draw alone.
        generator = torch.Generator().manual_seed(0)
        X = torch.rand(40, 3, generator=generator, dtype=torch.float64).to(DEVICE, DTYPE)
        levels = 0.1 + torch.rand(40, 4, generator=generator, dtype=torch.float64)
        X_cat = [(levels / levels.sum(dim=1, keepdim=True)).to(DEVICE, DTYPE)]
        Wd_cont = torch.tensor([0.1, 0.2, 0.3], dtype=DTYPE, device=DEVICE)
        Wd_cat = torch.tensor([0.4], dtype=DTYPE, device=DEVICE)

        def draw(columns=None):
            return select_kmeanspp_seeds_(
                columns,
                X,
                X_cat,
                6,
                [torch.Generator(device=DEVICE).manual_seed(seed) for seed in (5, 7, 11)],
                Wd_cont,
                Wd_cat,
                torch.ones(40, dtype=DTYPE, device=DEVICE),
                n_candidates=n_candidates,
            )

        columns = torch.empty(3, 40, 6, dtype=DTYPE, device=DEVICE)
        seeds = draw(columns)

        assert torch.equal(seeds, draw())
        for batch, selected in enumerate(seeds):
            continuous = (Wd_cont * (X.unsqueeze(1) - X[selected]) ** 2).sum(dim=2)
            cross_entropy = -X_cat[0] @ X_cat[0][selected].log().T
            torch.testing.assert_close(columns[batch], continuous + Wd_cat * cross_entropy)

    def test_greedy_columns_reuse_the_scored_candidates(self, monkeypatch):
        batches = []
        original = centroids.dissimilarity_to_rows

        def track_dissimilarity(*args):
            dissimilarity = original(*args)

            def tracked(seeds):
                batches.append(seeds.numel())
                return dissimilarity(seeds)

            return tracked

        monkeypatch.setattr(centroids, "dissimilarity_to_rows", track_dissimilarity)
        X = torch.arange(8, dtype=DTYPE, device=DEVICE).unsqueeze(1) / 8
        columns = X.new_empty(3, 8, 4)

        select_kmeanspp_seeds_(
            columns,
            X,
            [],
            4,
            [torch.Generator(device=DEVICE).manual_seed(seed) for seed in (5, 7, 11)],
            X.new_ones(1),
            X.new_empty(0),
            X.new_ones(8),
            n_candidates=3,
        )

        assert batches == [3, 9, 9, 9]

    @pytest.mark.parametrize("n_candidates", [1, 3])
    def test_degenerate_fallback_writes_every_seed_column(self, n_candidates):
        X = torch.full((4, 1), 0.5, dtype=DTYPE, device=DEVICE)
        columns = X.new_full((3, 4, 4), torch.nan)

        seeds = select_kmeanspp_seeds_(
            columns,
            X,
            [],
            4,
            [torch.Generator(device=DEVICE).manual_seed(seed) for seed in (5, 7, 11)],
            X.new_ones(1),
            X.new_empty(0),
            X.new_ones(4),
            n_candidates=n_candidates,
        )

        assert torch.equal(seeds.sort(dim=1).values, torch.arange(4, device=DEVICE).expand(3, -1))
        assert torch.equal(columns, torch.zeros_like(columns))


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

        # The [0, 1] rows carry almost no weight, so a tie rule that sent the [0.5, 0.5] rows
        # to class 1 would seed both of them; the lowest-class rule seeds one of each pair.
        sample_weights = torch.tensor([1.0, 1e-30, 1.0, 1.0, 1e-30], dtype=torch.float64)
        state = Network.initialise(
            _classification_recipe(Input(K=2, balanced=True)),
            X_cont,
            y,
            sample_weights=sample_weights,
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
        ("continuous", "reason"),
        (
            pytest.param(
                torch.zeros(1, 2, dtype=torch.float64), r"shape \(2, 2\)", id="wrong-size"
            ),
            pytest.param(
                torch.tensor([[0.0, float("inf")], [1.0, 1.0]], dtype=torch.float64),
                "only finite",
                id="non-finite",
            ),
        ),
    )
    def test_supplied_continuous_centroids_must_be_valid(self, continuous, reason):
        with pytest.raises(ValueError, match=f"continuous_centroids .*{reason}"):
            _initialise(Input(K=2), continuous_centroids=continuous)

    def test_integer_continuous_centroids_are_rejected_even_with_a_computation_dtype(self):
        with pytest.raises(ValueError, match="continuous_centroids must be a floating tensor"):
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
                "non-negative",
                id="negative",
            ),
            pytest.param(
                [torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)],
                "finite positive mass",
                id="zero-mass",
            ),
            pytest.param(
                [torch.tensor([[1.0, float("nan")], [1.0, 1.0]], dtype=torch.float64)],
                "only finite values",
                id="non-finite",
            ),
        ),
    )
    def test_supplied_categorical_centroids_must_be_valid(self, categorical, message):
        X_cont = torch.empty(4, 0, dtype=torch.float64)
        X_cat = [torch.tensor([0, 1, 0, 1])]
        y = torch.tensor([0, 0, 1, 1])

        with pytest.raises(ValueError, match=rf"categorical_centroids\[0\] .*{message}"):
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
                r"continuous_centroids .*without continuous data",
                id="no-continuous-features",
            ),
            pytest.param(
                True,
                lambda c: {"categorical_centroids": [c]},
                r"categorical_centroids .*without categorical data",
                id="no-categorical-features",
            ),
            pytest.param(
                False,
                lambda c: {"categorical_centroids": "ab"},
                r"categorical_centroids .*sequence of tensors",
                id="string",
            ),
            pytest.param(
                False,
                lambda c: {"categorical_centroids": c},
                r"categorical_centroids .*sequence of tensors",
                id="tensor",
            ),
            pytest.param(
                False,
                lambda c: {"categorical_centroids": [c, c]},
                r"has 2 feature\(s\), but the input has 1",
                id="feature-count",
            ),
            pytest.param(
                True,
                lambda c: {"continuous_centroids": c[:, :1].tolist()},
                r"continuous_centroids .*floating tensor",
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
