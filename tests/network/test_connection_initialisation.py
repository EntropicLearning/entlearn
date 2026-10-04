"""Connection seeding preserves categorical geometry and selected profiles."""

from __future__ import annotations

import pytest
import torch
from conftest import DEVICE, DTYPE
from network._fixtures import blobs, connection_initialisation_data, materialise

from entlearn import ClassificationHead, Coupling, Hidden, Input, Network, Recipe, RegressionHead
from entlearn.network import model
from entlearn.network.blocks.hidden import _HiddenBlock
from entlearn.network.blocks.input import _StandardInputBlock
from entlearn.network.initialisation import connections as initialisation
from entlearn.primitives.encoding import densify_categorical
from entlearn.primitives.geometry import grouped_means


class TestConnectionInitialisation:
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

    def test_prior_off_keeps_duplicate_selected_s_profiles(self) -> None:
        X_cont = torch.tensor([[0.0], [0.0], [1.0], [1.0]], dtype=DTYPE, device=DEVICE)
        target = torch.tensor([0, 0, 1, 1], dtype=torch.int64, device=DEVICE)
        recipe = Recipe.chain(
            Input(K=2), Hidden(K=3), ClassificationHead(coupling=Coupling.S), coupling=Coupling.S
        )

        session = materialise(recipe, X_cont, target, seed=3)
        hidden = session.graph.blocks["hidden_1"]
        assert isinstance(hidden, _HiddenBlock)
        theta = hidden.incoming["input_to_hidden_1"].theta

        assert bool(((theta == 0.0) | (theta == 1.0)).all())
        assert bool((theta.sum(dim=1) == 1.0).all())
        assert torch.unique(theta, dim=0).shape[0] == 2

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

    def test_soft_representatives_match_grouped_means_on_one_hot_affiliations(self) -> None:
        """One-hot affiliations reproduce the hard route, the empty-group fallback included."""
        X_cont, target = connection_initialisation_data()
        recipe = Recipe.chain(
            Input(K=2, epsilon=0.2),
            Hidden(K=2, epsilon=0.2),
            ClassificationHead(coupling=Coupling.M),
            coupling=Coupling.M,
        )
        session = materialise(
            recipe,
            X_cont,
            target,
            X_cat=[target],
            sample_weights=torch.arange(1, 9, dtype=DTYPE, device=DEVICE),
        )
        data = session.data
        # Staging scales the weights to unit total; moving them off it makes the fallback's
        # division by the total mass observable.
        data.sample_weights.mul_(3.0)
        labels = torch.tensor([0, 1, 0, 1, 1, 0, 1, 0], device=DEVICE)
        gamma = torch.nn.functional.one_hot(labels, 3).to(DTYPE)  # the third group is empty

        soft = initialisation._geometry_from_affiliations(
            gamma, session.graph.input.feature_weights, session
        )
        continuous, categories, masses = grouped_means(
            labels,
            data.sample_weights,
            data.X_cont,
            densify_categorical(list(data.X_cat), list(data.schema.M_cat), DTYPE),
            3,
        )

        assert masses[2] == 0.0
        torch.testing.assert_close(soft.group_masses, masses)
        torch.testing.assert_close(soft.continuous_centroids, continuous)
        torch.testing.assert_close(soft.categorical_centroids[0], categories[0])

    @pytest.mark.parametrize("weighted", [False, True])
    @pytest.mark.parametrize(
        ("widths", "first_coupling"),
        [((4, 3, 2), Coupling.M), ((2, 4, 2), Coupling.M), ((3, 4, 2), Coupling.S)],
    )
    def test_deeper_merge_counts_use_the_inherited_geometry_masses(
        self,
        monkeypatch: pytest.MonkeyPatch,
        widths: tuple[int, int, int],
        first_coupling: Coupling,
        weighted: bool,
    ) -> None:
        X_cont, target = connection_initialisation_data()
        sample_weights = torch.arange(1, 9, dtype=DTYPE, device=DEVICE) if weighted else None
        recipe = Recipe.chain(
            Input(K=widths[0], epsilon=0.2),
            Hidden(K=widths[1], epsilon=0.2),
            Hidden(K=widths[2], epsilon=0.2),
            ClassificationHead(coupling=Coupling.M),
            coupling=(first_coupling, Coupling.M),
            theta_alpha=2.0,
        )
        original = initialisation.initialise_connection_
        preceding_geometry = []

        def check_handoff(state, gamma_source, source_geometry, session, *, generator):
            if preceding_geometry:
                assert source_geometry is preceding_geometry[-1]
                row_weights = session.data.sample_weights
                propagated_masses = (row_weights[:, None] * gamma_source).sum(dim=0)
                assert not torch.allclose(source_geometry.group_masses, propagated_masses)
            geometry = original(state, gamma_source, source_geometry, session, generator=generator)
            assert geometry.continuous_centroids.shape == (state.theta.shape[0], 1)
            assert geometry.categorical_centroids[0].shape == (state.theta.shape[0], 2)
            torch.testing.assert_close(
                geometry.group_masses.sum(),
                torch.tensor(1.0, dtype=DTYPE, device=DEVICE),
            )
            if preceding_geometry:
                group_labels = state.theta.argmax(dim=0)
                expected_counts = (
                    torch.nn.functional.one_hot(group_labels, state.theta.shape[0]).T
                    * source_geometry.group_masses
                ) + state.pseudocount
                expected_theta = expected_counts / expected_counts.sum(dim=0)
                torch.testing.assert_close(state.theta, expected_theta)
            preceding_geometry.append(geometry)
            return geometry

        monkeypatch.setattr(initialisation, "initialise_connection_", check_handoff)

        materialise(recipe, X_cont, target, X_cat=[target], sample_weights=sample_weights, seed=5)

        assert len(preceding_geometry) == 2


class TestProfileCapacity:
    @pytest.mark.parametrize("operation", ("initialise", "fit", "fit-supplied"))
    @pytest.mark.parametrize("task", ("classification", "regression"))
    @pytest.mark.parametrize(
        ("coupling", "widths"),
        (
            (Coupling.M, (2, 5)),
            (Coupling.M, (2, 3, 5)),
            (Coupling.S, (2, 5)),
            (Coupling.S, (5, 5)),
            (Coupling.S, (6, 5)),
        ),
        ids=("M-widening", "M-deeper", "S-widening", "S-equal", "S-narrowing"),
    )
    def test_rejects_before_initial_geometry_or_model_allocation(
        self,
        monkeypatch: pytest.MonkeyPatch,
        operation: str,
        task: str,
        coupling: Coupling,
        widths: tuple[int, ...],
    ) -> None:
        X_cont, codes = blobs(2, 0.04, rows_per_blob=2, n_features=1)
        target = codes if task == "classification" else codes.to(DTYPE)
        head = (
            ClassificationHead(coupling=coupling) if task == "classification" else RegressionHead()
        )
        recipe = Recipe.chain(
            Input(K=widths[0]),
            *(Hidden(K=width) for width in widths[1:]),
            head,
            coupling=coupling,
        )
        kwargs = {}
        if operation == "fit-supplied":
            kwargs["initial_state"] = Network.initialise(
                recipe, X_cont.repeat(2, 1), target.repeat(2), seed=3
            )

        def forbidden(*args, **kwargs):
            pytest.fail("profile capacity must be checked before geometry or model allocation")

        monkeypatch.setattr(model, "_seed_initial_state", forbidden)
        monkeypatch.setattr(_StandardInputBlock, "make_block", forbidden)
        target_name = f"hidden_{len(widths) - 1}"
        source_name = "input" if len(widths) == 2 else f"hidden_{len(widths) - 2}"
        message = f"{source_name}_to_{target_name}.*5 profiles.*4 rows"
        with pytest.raises(ValueError, match=message):
            if operation == "initialise":
                Network.initialise(recipe, X_cont, target)
            else:
                Network.fit(recipe, X_cont, target, max_iter=1, **kwargs)

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
