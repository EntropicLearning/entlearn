"""Continuation from original geometry shared across different requested widths."""

import pickle

import pytest
import torch
from network._fixtures import resume_case

from entlearn import Network


class TestResumeSharedState:
    @pytest.mark.parametrize("task", ["classification", "regression"])
    @pytest.mark.parametrize("kind", ["standard", "categorical", "manifold"])
    def test_prefix_fit_resumes_the_same_trajectory_after_transport(self, task, kind):
        recipe, X, y, categories = resume_case(task, kind)
        original = Network.initialise(
            recipe.replace_block("input", K=6), X, y, X_cat=categories, seed=5
        )
        controls = dict(X_cat=categories, initial_state=original, seed=5, tol=0)
        full = Network.fit(recipe, X, y, max_iter=8, **controls)
        source = Network.fit(recipe, X, y, max_iter=3, **controls)
        source, state = pickle.loads(pickle.dumps((source, source.initial_state)))
        torch.testing.assert_close(
            state.input_geometry.continuous_centroids,
            source.initial_state.input_geometry.continuous_centroids,
            rtol=0,
            atol=0,
        )
        assert original.input_geometry.K_active > recipe.blocks[0].K
        snapshot = source.predict(X, X_cat=categories)

        continued = source.resume(X, y, X_cat=categories, max_iter=8, tol=0)

        assert source.can_resume and continued.can_resume
        assert continued.diagnostics.loss_history == full.diagnostics.loss_history
        assert continued.diagnostics.n_iter == full.diagnostics.n_iter
        torch.testing.assert_close(
            continued.predict(X, X_cat=categories), full.predict(X, X_cat=categories)
        )
        torch.testing.assert_close(source.predict(X, X_cat=categories), snapshot, rtol=0, atol=0)
        torch.testing.assert_close(
            continued.initial_state.input_geometry.continuous_centroids,
            state.input_geometry.continuous_centroids,
            rtol=0,
            atol=0,
        )
