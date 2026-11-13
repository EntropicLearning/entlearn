"""Connection seeding preserves categorical geometry and selected profiles."""

from __future__ import annotations

import math
from contextlib import nullcontext
from functools import partial

import pytest
import torch
from conftest import DEVICE, DTYPE, FIXED_BUDGET, by_depth
from network._fixtures import (
    banded_rows,
    blobs,
    connection_initialisation_data,
    materialise,
    outer_intervals_share_a_label,
    skip_fit_iterations,
    squared_error,
)
from primitives._fixtures import SEEDS

from entlearn import ClassificationHead, Coupling, Hidden, Input, Network, Recipe, RegressionHead
from entlearn.network.blocks.hidden import _HiddenBlock
from entlearn.network.initialisation import connections as initialisation
from entlearn.primitives.normalise import _tiny


class TestConnectionInitialisation:
    @FIXED_BUDGET
    # Without the continuous column an M chain has two distinct profiles for three clusters.
    @pytest.mark.filterwarnings("ignore:hidden block 'hidden_1' K=3 exceeds:UserWarning")
    @pytest.mark.parametrize("task", ["classification", "regression"])
    @pytest.mark.parametrize("continuous", [False, True])
    @pytest.mark.parametrize(
        ("coupling", "widths"),
        [(Coupling.M, (2, 3)), (Coupling.S, (3, 2)), (Coupling.S, (2, 2)), (Coupling.S, (2, 3))],
    )
    def test_profile_seeding_accepts_hard_categorical_rows(
        self, task: str, continuous: bool, coupling: Coupling, widths: tuple[int, int]
    ) -> None:
        codes = torch.tensor([0, 0, 1, 1, 2, 2, 0, 2], dtype=torch.int64, device=DEVICE)
        distributions = torch.nn.functional.one_hot(codes, 3).to(DTYPE)
        X_cont = torch.linspace(0.0, 1.0, codes.numel(), dtype=DTYPE, device=DEVICE)[:, None]
        if not continuous:
            X_cont = X_cont[:, :0]
        target = codes if task == "classification" else codes.to(DTYPE)[:, None]
        head = (
            ClassificationHead(coupling=coupling) if task == "classification" else RegressionHead()
        )
        recipe = Recipe.chain(
            Input(K=widths[0], epsilon=0.2),
            Hidden(K=widths[1], epsilon=0.2),
            head,
            coupling=coupling,
        )

        hard = Network.fit(recipe, X_cont, target, X_cat=[codes], max_iter=1, seed=3)
        soft = Network.fit(recipe, X_cont, target, X_cat=[distributions], max_iter=1, seed=3)

        torch.testing.assert_close(
            hard.predict(X_cont, X_cat=[codes]), soft.predict(X_cont, X_cat=[distributions])
        )
        torch.testing.assert_close(
            torch.tensor(hard.diagnostics.loss_history, dtype=DTYPE, device=DEVICE),
            torch.tensor(soft.diagnostics.loss_history, dtype=DTYPE, device=DEVICE),
        )

    @pytest.mark.parametrize("hidden_epsilon", [0.0, 0.2], ids=["hard", "soft"])
    @pytest.mark.parametrize("theta_alpha", [1.0, 2.0])
    def test_repeated_s_selection_takes_the_uniform_row(
        self, monkeypatch: pytest.MonkeyPatch, theta_alpha: float, hidden_epsilon: float
    ) -> None:
        """Two hard source profiles cannot seed three distinct rows.

        A hard block leaves the repeated selection empty and loses it, with a warning. A soft
        block gives it a share of every row and keeps it, with its uniform row.
        """
        X_cont = torch.tensor([[0.0], [0.0], [1.0], [1.0]], dtype=DTYPE, device=DEVICE)
        target = torch.tensor([0, 0, 1, 1], dtype=torch.int64, device=DEVICE)
        recipe = Recipe.chain(
            Input(K=2),
            Hidden(K=3, epsilon=hidden_epsilon),
            ClassificationHead(coupling=Coupling.S),
            coupling=Coupling.S,
            theta_alpha=theta_alpha,
        )
        original = initialisation.initialise_connection_
        seeded = []

        def capture(state, *args, **kwargs):
            geometry = original(state, *args, **kwargs)
            seeded.append(state.theta.clone())
            return geometry

        monkeypatch.setattr(initialisation, "initialise_connection_", capture)
        hard = hidden_epsilon == 0.0
        expected = pytest.warns(UserWarning, match="'hidden_1' K=3 exceeds its 2 seeded")

        with expected if hard else nullcontext():
            session = materialise(recipe, X_cont, target, seed=3)

        (theta,) = seeded
        repeated = (theta == 0.5).all(dim=1)
        assert int(repeated.sum()) == 1
        assert torch.unique(theta[~repeated].argmax(dim=1)).numel() == 2
        hidden = session.graph.blocks["hidden_1"]
        assert isinstance(hidden, _HiddenBlock)
        kept = hidden.incoming["input_to_hidden_1"].theta
        if hard:
            torch.testing.assert_close(kept, theta[~repeated])
        else:
            torch.testing.assert_close(kept, theta)
            assert bool((hidden.gamma[:, repeated] > 0.0).all())

    def test_reconstructed_representatives_are_affiliation_weighted_means(self) -> None:
        """A transferred chain rebuilds weighted means, falling back globally when empty."""
        X_cont = torch.tensor(
            [[0.0, 1.0], [0.2, 0.8], [0.6, 0.4], [1.0, 0.0]], dtype=DTYPE, device=DEVICE
        )
        codes = torch.tensor([0, 0, 1, 1], dtype=torch.int64, device=DEVICE)
        sample_weights = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=DTYPE, device=DEVICE)
        recipe = Recipe.chain(
            Input(K=2, epsilon=0.2),
            Hidden(K=2, epsilon=0.2),
            ClassificationHead(coupling=Coupling.M),
            coupling=Coupling.M,
        )
        session = materialise(recipe, X_cont, codes, X_cat=[codes], sample_weights=sample_weights)
        # Staging normalises the weights to the simplex. Multiplying them by 3 detects a
        # missing division when computing the fallback centroid for an empty cluster.
        session.data.sample_weights.mul_(3.0)
        # The third group is deliberately unoccupied, so its representatives take the fallback.
        gamma = torch.tensor(
            [[0.75, 0.25, 0.0], [0.5, 0.5, 0.0], [0.25, 0.75, 0.0], [0.0, 1.0, 0.0]],
            dtype=DTYPE,
            device=DEVICE,
        )
        feature_weights = session.graph.input.feature_weights

        geometry = initialisation._geometry_from_affiliations(gamma, feature_weights, session)

        row_weights = session.data.sample_weights[:, None]
        weighted = gamma * row_weights
        masses = weighted.sum(dim=0)
        staged = session.data.X_cont
        one_hot = torch.nn.functional.one_hot(codes, 2).to(DTYPE)
        assert geometry.feature_weights is feature_weights
        torch.testing.assert_close(geometry.group_masses, masses)
        torch.testing.assert_close(
            geometry.continuous_centroids,
            torch.stack(
                [
                    (weighted[:, group, None] * staged).sum(dim=0) / masses[group]
                    if masses[group] > 0
                    else (row_weights * staged).sum(dim=0) / row_weights.sum()
                    for group in range(gamma.shape[1])
                ]
            ),
        )
        (categories,) = geometry.categorical_centroids
        torch.testing.assert_close(
            categories,
            torch.stack(
                [
                    (weighted[:, group, None] * one_hot).sum(dim=0) / masses[group]
                    if masses[group] > 0
                    else torch.full((2,), 0.5, dtype=DTYPE, device=DEVICE)
                    for group in range(gamma.shape[1])
                ]
            ),
        )

    def test_a_deeper_merge_counts_the_masses_its_source_groups_inherit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        X_cont, target = connection_initialisation_data()
        sample_weights = torch.arange(1, 9, dtype=DTYPE, device=DEVICE)
        # The first connection has the same cluster count (3); the second merges 3 clusters
        # into 2.
        recipe = Recipe.chain(
            Input(K=3, epsilon=0.2),
            Hidden(K=3, epsilon=0.2),
            Hidden(K=2, epsilon=0.2),
            ClassificationHead(coupling=Coupling.M),
            coupling=Coupling.M,
            theta_alpha=2.0,
        )
        skip_fit_iterations(monkeypatch)
        network = Network.fit(
            recipe,
            X_cont,
            target,
            X_cat=[target],
            sample_weights=sample_weights,
            seed=5,
            max_iter=1,
        )

        affiliations = network.inspect("training_affiliations")
        row_weights = sample_weights / sample_weights.sum()
        inherited = row_weights @ affiliations["input"]
        propagated = row_weights @ affiliations["hidden_1"]
        assert not torch.allclose(inherited, propagated)
        theta = network.inspect("transition_matrices")["hidden_1_to_hidden_2"]
        # The documented prior: (theta_alpha - 1) / (K_target * K_source) in every entry.
        pseudocount = 1.0 / (2 * 3)
        for column, mass in zip(theta.T, inherited, strict=True):
            # Each source column puts all its mass on its group, then adds the prior.
            counts = torch.stack((torch.full_like(mass, pseudocount), mass + pseudocount))
            torch.testing.assert_close(column.sort().values, counts / counts.sum())


class TestSeededWidthCap:
    @pytest.mark.parametrize("seed", by_depth((0,), (0, 1, 2), (0, 1, 2, 3, 4, 5)))
    @pytest.mark.parametrize(
        ("coupling", "widths"),
        [
            ((Coupling.S, Coupling.S), (4, 4)),
            ((Coupling.S, Coupling.M), (4, 2)),
            ((Coupling.M, Coupling.M), (4, 3)),
            ((Coupling.M, Coupling.S), (4, 4)),
        ],
        ids=["S-S", "S-narrowing-M", "M-M", "M-S"],
    )
    def test_chain_over_a_hard_input_keeps_one_cluster_per_source_profile(
        self, coupling: tuple[Coupling, Coupling], widths: tuple[int, int], seed: int
    ) -> None:
        """Clusters beyond the two hard source profiles are removed at seeding, with a warning."""
        offsets = torch.linspace(0.0, 0.07, 16, dtype=DTYPE, device=DEVICE)
        X_cont = torch.cat([offsets, 1.0 - offsets])[:, None]
        target = torch.tensor([0] * 15 + [1] + [1] * 15 + [0], dtype=torch.int64, device=DEVICE)
        recipe = Recipe.chain(
            Input(K=2),
            Hidden(K=widths[0]),
            Hidden(K=widths[1]),
            ClassificationHead(),
            coupling=coupling,
        )

        with pytest.warns(UserWarning, match="exceeds its 2 seeded selections") as caught:
            network = Network.fit(recipe, X_cont, target, seed=seed)

        capped = [
            f"hidden block {name!r} K={K} exceeds its 2 seeded selections with distinct "
            "source affiliations; reducing the initial hidden cluster count to 2"
            for name, K in zip(("hidden_1", "hidden_2"), widths, strict=True)
            if K > 2
        ]
        assert [str(w.message) for w in caught] == capped
        assert [m for m in network.diagnostics.warnings if "hidden block" in m] == capped
        active = dict(network.schema.K_active)
        assert (active["input"], active["hidden_1"], active["hidden_2"]) == (2, 2, 2)
        trained = network.inspect("training_affiliations")
        predicted = network.predict_with_details(X_cont, details=("affiliations",)).affiliations
        for name in ("hidden_1", "hidden_2"):
            torch.testing.assert_close(predicted[name], trained[name])

    # The narrower input regenerates the captured input and the hidden block it feeds.
    @pytest.mark.filterwarnings("ignore:InitialState block:UserWarning")
    def test_cap_compacts_a_transferred_transition_downstream(self) -> None:
        """A capped block's surplus clusters leave the source axis of the next, transferred block."""
        centres = torch.tensor([[0.05], [0.35], [0.65], [0.95]], dtype=DTYPE, device=DEVICE)
        X_cont = (centres + torch.linspace(-0.02, 0.02, 5, dtype=DTYPE, device=DEVICE)).reshape(
            -1, 1
        )
        target = torch.tensor([0, 1, 0, 1], device=DEVICE).repeat_interleave(5)

        def recipe(K_in: int) -> Recipe:
            return Recipe.chain(
                Input(K=K_in), Hidden(K=4), Hidden(K=2), ClassificationHead(), coupling=Coupling.S
            )

        start = Network.initialise(recipe(4), X_cont, target, continuous_centroids=centres)
        source = Network.fit(recipe(4), X_cont, target, initial_state=start)
        assert dict(source.schema.K_active) == {
            "input": 4,
            "hidden_1": 4,
            "hidden_2": 2,
            "output": 2,
        }

        with pytest.warns(UserWarning, match="'hidden_1' K=4 exceeds") as caught:
            network = Network.fit(
                recipe(2), X_cont, target, initial_state=source.capture_current_state(), seed=3
            )

        assert not any("'hidden_2'" in str(w.message) for w in caught)
        transitions = network.inspect("transition_matrices")
        assert transitions["input_to_hidden_1"].shape == (2, 2)
        assert transitions["hidden_1_to_hidden_2"].shape[1] == 2

    @pytest.mark.parametrize("coupling", [Coupling.M, Coupling.S])
    def test_soft_source_without_repeated_selections_keeps_the_declared_width(
        self, coupling: Coupling
    ) -> None:
        """Only a repeated selection is removed, not every cluster the first propagation empties."""
        X_cont, target = blobs(3, 0.04, rows_per_blob=8, n_features=1)
        recipe = Recipe.chain(
            Input(K=3, epsilon=0.05), Hidden(K=6), ClassificationHead(), coupling=coupling
        )

        # Pytest turns any capping warning into an error.
        session = materialise(recipe, X_cont, target, seed=0)

        hidden = session.graph.blocks["hidden_1"]
        assert isinstance(hidden, _HiddenBlock)
        assert hidden.K == 6
        assert session.warning_messages == []
        if coupling is Coupling.M:
            # The M seed holds rows in every group, yet propagating it empties some clusters.
            assert bool((hidden.gamma.sum(dim=0) == 0.0).any())


def _groups(network: Network, connection: str) -> list[int]:
    """Return the target cluster each source cluster's transition column favours."""
    return network.inspect("transition_matrices")[connection].argmax(dim=0).tolist()


class TestLabelEntropies:
    def test_empty_pure_and_mixed_clusters_use_labelled_mass(self) -> None:
        tiny_mass = torch.finfo(DTYPE).eps / 8
        sums = torch.tensor(
            [[0.0, 0.0], [tiny_mass, 0.0], [0.25, 0.25]], dtype=DTYPE, device=DEVICE
        )

        result = initialisation._label_entropies(sums, sums.sum(dim=1))

        expected = torch.tensor([0.0, 0.0, 0.5 * math.log(2.0)], dtype=DTYPE, device=DEVICE)
        torch.testing.assert_close(result, expected, rtol=8 * torch.finfo(DTYPE).eps, atol=0.0)


class TestSeedHeadLoss:
    def test_regression_keeps_small_positive_labelled_mass(self) -> None:
        mass = torch.finfo(DTYPE).eps / 8
        sums = torch.tensor([[0.0], [mass * 0.5]], dtype=DTYPE, device=DEVICE)
        masses = torch.tensor([0.0, mass], dtype=DTYPE, device=DEVICE)

        result = initialisation._head_loss(sums, masses, classification=False)

        expected = torch.tensor(-mass * 0.25, dtype=DTYPE, device=DEVICE)
        torch.testing.assert_close(result, expected, rtol=8 * torch.finfo(DTYPE).eps, atol=0.0)


class TestLabelAwareSeeding:
    def test_regression_merges_keep_small_positive_labelled_masses(self) -> None:
        masses = torch.tensor([0.1, 0.1, 0.1, 0.7], dtype=DTYPE, device=DEVICE)
        masses *= torch.finfo(DTYPE).eps / 8
        sums = masses[:, None] * torch.tensor(
            [[0.0], [1.0], [0.0], [1.0]], dtype=DTYPE, device=DEVICE
        )
        points = torch.zeros((4, 1), dtype=DTYPE, device=DEVICE)

        groups = initialisation._group_by_head_loss(sums, masses, points, masses, 2, False, 0.0)

        assert groups.tolist() == [0, 1, 0, 1]

    @pytest.mark.parametrize("task", ["classification", "regression"])
    def test_narrowing_m_groups_same_label_clusters_that_are_apart(self, task: str) -> None:
        """No grouping of neighbours joins the outer intervals; the labels do."""
        X_cont, centres, codes = outer_intervals_share_a_label()
        classification = task == "classification"
        target = codes if classification else codes.to(DTYPE)
        head = ClassificationHead() if classification else RegressionHead()
        recipe = Recipe.chain(Input(K=3), Hidden(K=2), head)

        state = Network.initialise(
            recipe, X_cont, target, continuous_centroids=centres, seed=SEEDS[0]
        )
        network = Network.fit(recipe, X_cont, target, initial_state=state)

        outer_left, middle, outer_right = _groups(network, "input_to_hidden_1")
        assert outer_left == outer_right != middle
        prediction = network.predict(X_cont)
        if classification:
            assert prediction.argmax(dim=1).tolist() == codes.tolist()
        else:
            torch.testing.assert_close(prediction[:, 0], target)

    @pytest.mark.parametrize("scale", [1e-9, 1e9])
    def test_regression_label_rows_do_not_depend_on_the_target_scale(self, scale: float) -> None:
        """A merge costs the same at any target scale, below machine precision included."""
        X_cont, _, codes = outer_intervals_share_a_label()
        target = codes.to(DTYPE)
        recipe = Recipe.chain(Input(K=3), Hidden(K=2), RegressionHead())

        reference, _, _ = initialisation._label_rows(materialise(recipe, X_cont, target))
        scaled, _, _ = initialisation._label_rows(materialise(recipe, X_cont, target * scale))

        torch.testing.assert_close(scaled, reference)

    def test_constant_regression_targets_carry_no_label_signal(self) -> None:
        """Round-off in centring a constant target is not scaled up to unit variance."""
        X_cont, _, _ = outer_intervals_share_a_label()
        target = torch.full((X_cont.shape[0],), 1e6 + 1 / 3, dtype=DTYPE, device=DEVICE)
        recipe = Recipe.chain(Input(K=3), Hidden(K=2), RegressionHead())

        rows, _, _ = initialisation._label_rows(materialise(recipe, X_cont, target))

        assert float(rows.abs().max()) <= math.sqrt(torch.finfo(DTYPE).eps)

    def test_every_narrowing_block_groups_by_the_labels(self) -> None:
        """An earlier block keeps neighbours together; the last joins the outer intervals."""
        X_cont, centres, codes = outer_intervals_share_a_label()
        halves = torch.tensor([-0.03, 0.03], dtype=DTYPE, device=DEVICE)
        recipe = Recipe.chain(Input(K=6), Hidden(K=3), Hidden(K=2), ClassificationHead())

        state = Network.initialise(
            recipe,
            X_cont,
            codes,
            continuous_centroids=(centres + halves).reshape(-1, 1),
            seed=SEEDS[0],
        )
        network = Network.fit(recipe, X_cont, codes, initial_state=state)

        first = _groups(network, "input_to_hidden_1")
        assert first[0] == first[1] and first[2] == first[3] and first[4] == first[5]
        assert len(set(first)) == 3
        second = _groups(network, "hidden_1_to_hidden_2")
        assert second[first[0]] == second[first[4]] != second[first[2]]
        assert network.predict(X_cont).argmax(dim=1).tolist() == codes.tolist()

    @pytest.mark.parametrize("decisive", [0, 1])
    @FIXED_BUDGET
    def test_regression_outputs_count_as_the_head_weights_them(self, decisive: int) -> None:
        """Each output asks for another grouping; the one the head weights decides."""
        X_cont, centres, codes = outer_intervals_share_a_label()
        first_two = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1], device=DEVICE)
        target = torch.stack([codes, first_two], dim=1).to(DTYPE)
        W_M = (1.0, 1e-6) if decisive == 0 else (1e-6, 1.0)
        recipe = Recipe.chain(Input(K=3), Hidden(K=2), RegressionHead(W_M=W_M))

        state = Network.initialise(
            recipe, X_cont, target, continuous_centroids=centres, seed=SEEDS[0]
        )
        network = Network.fit(recipe, X_cont, target, initial_state=state, max_iter=1)

        left, middle, right = _groups(network, "input_to_hidden_1")
        if decisive == 0:
            assert left == right != middle
        else:
            assert left == middle != right

    @FIXED_BUDGET
    def test_unlabelled_rows_leave_the_grouping_to_the_labelled_rows(self) -> None:
        X_cont, centres, codes = outer_intervals_share_a_label()
        target = codes.to(DTYPE)
        target[torch.tensor([1, 4, 7], device=DEVICE)] = torch.nan
        recipe = Recipe.chain(Input(K=3), Hidden(K=2), RegressionHead())

        state = Network.initialise(
            recipe, X_cont, target, continuous_centroids=centres, seed=SEEDS[0]
        )
        network = Network.fit(recipe, X_cont, target, initial_state=state, max_iter=1)

        left, middle, right = _groups(network, "input_to_hidden_1")
        assert left == right != middle

    def test_a_cluster_without_labels_joins_its_nearest_group(self) -> None:
        """A cluster with no labelled mass costs the head nothing, so position places it."""
        sums = torch.tensor(
            [[0.25, 0.0], [0.0, 0.25], [0.0, 0.0], [0.25, 0.0]], dtype=DTYPE, device=DEVICE
        )
        points = torch.tensor([[0.0], [0.3], [0.4], [1.0]], dtype=DTYPE, device=DEVICE)
        point_masses = torch.full((4,), 0.25, dtype=DTYPE, device=DEVICE)

        groups = initialisation._group_by_head_loss(
            sums, sums.sum(dim=1), points, point_masses, 2, True, 1e-3
        )

        assert groups.tolist() == [0, 1, 1, 0]

    @FIXED_BUDGET
    def test_a_geometric_grouping_replaces_a_merge_that_serves_the_head_worse(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The kept grouping is the one whose propagated affiliations the head prefers."""
        X_cont, codes = connection_initialisation_data()
        centres = torch.tensor([[0.05], [0.25], [0.75], [0.95]], dtype=DTYPE, device=DEVICE)
        recipe = Recipe.chain(Input(K=4), Hidden(K=2), ClassificationHead())

        def mixing_merge(sums, *args, **kwargs):
            return torch.tensor([0, 1, 0, 1], device=sums.device)

        monkeypatch.setattr(initialisation, "_group_by_head_loss", mixing_merge)
        state = Network.initialise(
            recipe, X_cont, codes, continuous_centroids=centres, seed=SEEDS[0]
        )
        network = Network.fit(recipe, X_cont, codes, initial_state=state, max_iter=1)

        groups = _groups(network, "input_to_hidden_1")
        assert groups[0] == groups[1] != groups[2] == groups[3]

    def test_equal_width_m_keeps_the_identity(self) -> None:
        X_cont, centres, codes = outer_intervals_share_a_label()
        recipe = Recipe.chain(Input(K=3), Hidden(K=3), ClassificationHead())

        state = Network.initialise(
            recipe, X_cont, codes, continuous_centroids=centres, seed=SEEDS[0]
        )
        session = materialise(recipe, X_cont, codes, initial_state=state)

        hidden = session.graph.blocks["hidden_1"]
        assert isinstance(hidden, _HiddenBlock)
        torch.testing.assert_close(
            hidden.incoming["input_to_hidden_1"].theta,
            torch.eye(3, dtype=DTYPE, device=DEVICE),
        )


class TestLabelAwareWideningS:
    @pytest.mark.parametrize("theta_alpha", [1.0, 1.01])
    def test_the_partition_the_head_prefers_is_kept(
        self, monkeypatch: pytest.MonkeyPatch, theta_alpha: float
    ) -> None:
        """Of the drawn partitions, the transition takes its update from the purest one."""
        X_cont, codes = banded_rows()
        bands = torch.arange(4, device=DEVICE).repeat_interleave(4)
        mixed = torch.arange(4, device=DEVICE).repeat(4)
        drawn = [mixed, bands, *[mixed] * (initialisation._SEED_CANDIDATES - 2)]
        recipe = Recipe.chain(
            Input(K=2, epsilon=0.05),
            Hidden(K=4),
            ClassificationHead(),
            coupling=Coupling.S,
            theta_alpha=theta_alpha,
            delta=[0.005, 0.0005],
        )
        monkeypatch.setattr(initialisation, "_group_source", lambda *args: torch.stack(drawn))

        session = materialise(recipe, X_cont, codes, seed=SEEDS[0])

        hidden = session.graph.blocks["hidden_1"]
        assert isinstance(hidden, _HiddenBlock)
        state = hidden.incoming["input_to_hidden_1"]
        gamma_source = session.graph.input.gamma
        counts = torch.nn.functional.one_hot(bands, 4).to(DTYPE).T @ gamma_source / 16
        counts += state.pseudocount
        torch.testing.assert_close(state.theta, counts / counts.sum(dim=1, keepdim=True))
        torch.testing.assert_close(state.log_theta, state.theta.clamp_min(_tiny(DTYPE)).log())

    @pytest.mark.parametrize(
        ("epsilon", "delta", "label_aware"),
        [(0.05, 0.005, True), (0.05, 0.05, False), (0.05, 1.0, False), (0.0, 0.005, False)],
        ids=("below", "equal", "above", "hard-source"),
    )
    @pytest.mark.parametrize("categorical", [False, True])
    @pytest.mark.filterwarnings("ignore:hidden block:UserWarning")
    def test_only_a_delta_below_the_source_temperature_is_seeded_by_the_labels(
        self,
        monkeypatch: pytest.MonkeyPatch,
        epsilon: float,
        delta: float,
        label_aware: bool,
        categorical: bool,
    ) -> None:
        X_cont, codes = banded_rows()
        X_cat = [codes] if categorical else None
        recipe = Recipe.chain(
            Input(K=2, epsilon=epsilon),
            Hidden(K=4),
            ClassificationHead(),
            coupling=Coupling.S,
            delta=[delta, 0.1 * delta],
        )
        unused = "_profile_partition" if label_aware else "_partition_rows_by_labels_"

        def forbidden(*args, **kwargs):
            pytest.fail(f"{unused} must not seed this connection")

        monkeypatch.setattr(initialisation, unused, forbidden)

        session = materialise(recipe, X_cont, codes, X_cat=X_cat, seed=SEEDS[0])

        hidden = session.graph.blocks["hidden_1"]
        assert isinstance(hidden, _HiddenBlock)
        theta = hidden.incoming["input_to_hidden_1"].theta
        # A profile seed over a nearly hard source repeats selections, so the block may be
        # capped below its declared K; every kept row is still a distribution.
        assert theta.shape == (hidden.K, 2)
        torch.testing.assert_close(
            theta.sum(dim=1), torch.ones(hidden.K, dtype=DTYPE, device=DEVICE)
        )

    @pytest.mark.parametrize("coupling", [Coupling.M, Coupling.S])
    def test_other_profile_seeds_do_not_read_the_labels(
        self, monkeypatch: pytest.MonkeyPatch, coupling: Coupling
    ) -> None:
        """Widening M, and S of equal or smaller width, keep the profile seed."""
        X_cont, codes = banded_rows()
        widths = (2, 4) if coupling is Coupling.M else (4, 2)
        recipe = Recipe.chain(
            Input(K=widths[0], epsilon=0.05),
            Hidden(K=widths[1]),
            ClassificationHead(),
            coupling=coupling,
            delta=[0.005, 0.0005],
        )

        def forbidden(*args, **kwargs):
            pytest.fail("this connection is seeded without the labels")

        monkeypatch.setattr(initialisation, "_label_rows", forbidden)

        materialise(recipe, X_cont, codes, seed=SEEDS[0])

    def test_the_extra_clusters_are_used_at_prediction(self) -> None:
        """Below the source temperature, a widening S connection splits its source clusters."""
        X_cont = torch.linspace(0.0, 1.0, 80, dtype=DTYPE, device=DEVICE)[:, None]
        codes = (X_cont[:, 0] * 4).clamp(max=3.5).to(torch.int64) % 2
        recipe = Recipe.chain(
            Input(K=2, epsilon=0.05),
            Hidden(K=4),
            ClassificationHead(),
            coupling=Coupling.S,
            delta=[0.005, 0.0005],
        )

        network = Network.fit(recipe, X_cont, codes, seed=SEEDS[0])

        result = network.predict_with_details(X_cont, details=("affiliations",))
        assert result.affiliations is not None
        assert torch.unique(result.affiliations["hidden_1"].argmax(dim=1)).numel() == 4
        accuracy = (result.prediction.argmax(dim=1) == codes).to(DTYPE).mean()
        assert accuracy > 0.7


class TestProfileCapacity:
    @pytest.mark.parametrize(
        ("operation", "coupling", "widths"),
        (
            ("fit", Coupling.M, (2, 5)),
            ("fit", Coupling.M, (2, 3, 5)),
            ("fit", Coupling.S, (5, 5)),
            ("initialise", Coupling.M, (2, 5)),
            ("fit-supplied", Coupling.M, (2, 5)),
            ("fit-fold", Coupling.M, (2, 5)),
        ),
        ids=("M-widening", "M-deeper", "S-equal", "initialise", "fit-supplied", "fit-fold"),
    )
    def test_rejects_more_profiles_than_rows(
        self, operation: str, coupling: Coupling, widths: tuple[int, ...]
    ) -> None:
        X_cont, target = blobs(2, 0.04, rows_per_blob=2, n_features=1)
        recipe = Recipe.chain(
            Input(K=widths[0]),
            *(Hidden(K=width) for width in widths[1:]),
            ClassificationHead(coupling=coupling),
            coupling=coupling,
        )
        kwargs = {}
        if operation == "fit-supplied":
            kwargs["initial_state"] = Network.initialise(
                recipe, X_cont.repeat(2, 1), target.repeat(2), seed=3
            )
        if operation == "fit-fold":
            # All eight rows could seed five profiles; the fold trains on four.
            X_cont, target = X_cont.repeat(2, 1), target.repeat(2)
            rows = torch.arange(8, device=DEVICE)
            kwargs["validation_pairs"] = ((rows[:4], rows[4:]),)
            kwargs["selection_loss"] = squared_error

        target_name = f"hidden_{len(widths) - 1}"
        source_name = "input" if len(widths) == 2 else f"hidden_{len(widths) - 2}"
        message = f"{source_name}_to_{target_name}.*5 profiles.*4 rows"
        if operation == "initialise":
            call = partial(Network.initialise, recipe, X_cont, target)
        else:
            call = partial(Network.fit, recipe, X_cont, target, max_iter=1, **kwargs)
        with pytest.raises(ValueError, match=message):
            call()

    @FIXED_BUDGET
    @pytest.mark.parametrize("coupling", (Coupling.M, Coupling.S))
    def test_profile_width_equal_to_the_row_count_is_allowed(self, coupling: Coupling) -> None:
        X_cont, target = blobs(2, 0.04, rows_per_blob=2, n_features=1)
        recipe = Recipe.chain(
            Input(K=2, epsilon=0.2),
            Hidden(K=X_cont.shape[0], epsilon=0.2),
            ClassificationHead(coupling=coupling),
            coupling=coupling,
        )

        state = Network.initialise(recipe, X_cont, target, seed=3)
        network = Network.fit(recipe, X_cont, target, initial_state=state, max_iter=1)

        assert network.predict(X_cont).shape == (4, 2)

    @FIXED_BUDGET
    def test_m_merge_can_group_more_representatives_than_training_rows(self) -> None:
        X_cont, target = blobs(2, 0.04, rows_per_blob=2, n_features=1)
        recipe = Recipe.chain(
            Input(K=6, epsilon=0.2),
            Hidden(K=5, epsilon=0.2),
            ClassificationHead(coupling=Coupling.M),
            coupling=Coupling.M,
        )
        centroids = torch.linspace(0.0, 1.0, 6, dtype=DTYPE, device=DEVICE)[:, None]

        state = Network.initialise(recipe, X_cont, target, continuous_centroids=centroids)
        network = Network.fit(recipe, X_cont, target, initial_state=state, max_iter=1)

        assert network.predict(X_cont).shape == (4, 2)
