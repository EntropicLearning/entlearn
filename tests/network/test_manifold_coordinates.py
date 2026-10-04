"""Narrow coordinate diagnostics for manifold optimisation and scratch reuse."""

import pytest
import torch
from _alloc import assert_zero_alloc
from conftest import DEVICE, DTYPE
from network._fixtures import assert_non_increasing, materialise

from entlearn import ClassificationHead, Coupling, Hidden, ManifoldInput, Recipe, RegressionHead
from entlearn.network.blocks import input_common as inp
from entlearn.network.blocks import manifold as mfld
from entlearn.network.fit import _loss


class TestManifoldCoordinates:
    @pytest.mark.parametrize("task", ["classification", "regression"])
    @pytest.mark.parametrize("coupling", [Coupling.M, Coupling.S])
    @pytest.mark.parametrize("epsilon", [0.0, 0.2])
    @pytest.mark.parametrize("weighted", [False, True])
    @pytest.mark.parametrize("shape", [(24, 3), (12, 20)])
    @torch.inference_mode()
    def test_manifold_coordinates_descend_and_reuse_allocated_scratch(
        self, task, coupling, epsilon, weighted, shape
    ):
        generator = torch.Generator(device=DEVICE).manual_seed(61)
        X = torch.randn(*shape, generator=generator, dtype=DTYPE, device=DEVICE) * 0.1
        X[shape[0] // 2 :] += 1
        y = (X[:, 0] > 0.5).to(torch.int64) if task == "classification" else X[:, :2]
        head = (
            ClassificationHead(coupling=coupling) if task == "classification" else RegressionHead()
        )
        recipe = Recipe.chain(
            ManifoldInput(K=3, subspace_dimension=2, epsilon=epsilon, epsilon_T=0.4),
            Hidden(K=2, epsilon=epsilon),
            head,
            coupling=coupling,
            theta_alpha=1.1,
        )
        weights = torch.arange(1, shape[0] + 1, dtype=DTYPE, device=DEVICE) if weighted else None
        session = materialise(recipe, X, y, sample_weights=weights, seed=9, warmup=2)
        block = session.graph.input
        before = _loss(session)
        session.graph.accumulate_outgoing_cost_(block, block.live_cache.disc_cost, session)
        inp.update_affiliations_(block, session)
        inp.prune_(block, session)
        mfld.refresh_cache_(block, session)
        after = _loss(session)
        assert_non_increasing(before, after, "input affiliation")
        for coordinate in (
            inp.update_instance_weights_,
            mfld.update_centroids_,
            mfld.update_projectors_,
        ):
            before = _loss(session)
            coordinate(block, session)
            mfld.refresh_cache_(block, session)
            after = _loss(session)
            assert_non_increasing(before, after, coordinate.__name__)
        # The factory-level oracle allows the eigensolver's internal LAPACK workspace,
        # but rejects fresh tensor storage in these steady-state coordinate operations.
        for coordinate in (
            inp.update_affiliations_,
            inp.update_instance_weights_,
            mfld.update_centroids_,
            mfld.update_projectors_,
            mfld.refresh_cache_,
        ):
            assert_zero_alloc(coordinate, block, session)
