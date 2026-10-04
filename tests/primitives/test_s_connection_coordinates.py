"""S-connection initialisation and coordinate contracts."""

from __future__ import annotations

import pytest
import torch
from conftest import DEVICE, DTYPE
from network._fixtures import (
    BLOB_SPREADS,
    assert_deep_coordinates_non_increasing,
    blobs,
    deep_classification_recipe,
    deep_classification_session,
    hidden_connection,
    materialise,
)

from entlearn import Coupling
from entlearn.primitives.normalise import _eps


class TestSConnectionCoordinates:
    def test_selected_source_profiles_are_the_unregularised_prototypes(self) -> None:
        X_cont, y = blobs(3, BLOB_SPREADS["overlapping"], seed=3)
        session = materialise(
            deep_classification_recipe(
                (3, 5), coupling=Coupling.S, head_coupling=Coupling.S, theta_alpha=1.0
            ),
            X_cont,
            y,
            seed=11,
        )
        theta = hidden_connection(session).theta
        gamma_source = session.graph.input.gamma

        distances = (theta[:, None, :] - gamma_source[None, :, :]).abs().amax(dim=2)

        torch.testing.assert_close(
            distances.amin(dim=1),
            torch.zeros(theta.shape[0], dtype=DTYPE, device=DEVICE),
            atol=8 * _eps(DTYPE),
            rtol=0.0,
        )

    @pytest.mark.parametrize("weighted", [False, True])
    def test_assigned_mass_scales_each_prototype_before_the_prior(self, weighted) -> None:
        X_cont, y = blobs(3, BLOB_SPREADS["overlapping"], seed=3)
        raw_weights = torch.arange(1, X_cont.shape[0] + 1, dtype=DTYPE, device=DEVICE)
        sample_weights = raw_weights if weighted else None
        unregularised = materialise(
            deep_classification_recipe(
                (3, 5), coupling=Coupling.S, head_coupling=Coupling.S, theta_alpha=1.0
            ),
            X_cont,
            y,
            sample_weights=sample_weights,
            seed=11,
        )
        regularised = materialise(
            deep_classification_recipe(
                (3, 5), coupling=Coupling.S, head_coupling=Coupling.S, theta_alpha=2.0
            ),
            X_cont,
            y,
            sample_weights=sample_weights,
            seed=11,
        )
        prototypes = hidden_connection(unregularised).theta
        gamma_source = unregularised.graph.input.gamma
        cross_entropy = -(
            gamma_source[:, None, :] * prototypes.clamp_min(_eps(DTYPE)).log()[None, :, :]
        ).sum(dim=2)
        labels = cross_entropy.argmin(dim=1)
        assigned_masses = torch.zeros(prototypes.shape[0], dtype=DTYPE, device=DEVICE)
        native_weights = (
            raw_weights / raw_weights.sum() if weighted else torch.ones_like(raw_weights)
        )
        assigned_masses.index_add_(
            0,
            labels,
            native_weights,
        )
        # Evaluate the native prior independently of the rescaled connection implementation.
        pseudocount = native_weights.sum() / prototypes.numel()
        expected = prototypes * assigned_masses.unsqueeze(1) + pseudocount
        expected /= expected.sum(dim=1, keepdim=True)

        torch.testing.assert_close(hidden_connection(regularised).theta, expected)
        assert bool((hidden_connection(regularised).theta > 0.0).all())

    def test_explicit_uniform_weights_preserve_the_initial_transition(self) -> None:
        X_cont, y = blobs(3, BLOB_SPREADS["overlapping"], seed=3)
        recipe = deep_classification_recipe(
            (3, 5), coupling=Coupling.S, head_coupling=Coupling.S, theta_alpha=2.0
        )
        unweighted = materialise(recipe, X_cont, y, seed=11)
        explicit_uniform = materialise(
            recipe,
            X_cont,
            y,
            sample_weights=torch.full(
                (X_cont.shape[0],),
                7.0,
                dtype=DTYPE,
                device=DEVICE,
            ),
            seed=11,
        )

        torch.testing.assert_close(
            hidden_connection(unweighted).theta, hidden_connection(explicit_uniform).theta
        )

    @pytest.mark.parametrize("K_target", [2, 5], ids=["narrow", "wide"])
    @pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weighted"])
    def test_each_s_coordinate_is_non_increasing(self, K_target: int, weighted: bool) -> None:
        session = deep_classification_session((3, K_target), coupling=Coupling.S, weighted=weighted)
        assert_deep_coordinates_non_increasing(session)
