"""Narrow diagnostics for parameter transfer before optimisation.

The public lifecycle always optimises. Suppressing its iteration isolates exact
parameter transfer without exposing an unfitted public Network. Fine-tuning and capture
from a prediction-only Network go through the public loader in
``test_prediction_persistence.py``.
"""

from dataclasses import replace

import pytest
import torch
from conftest import FIXED_BUDGET
from network._fixtures import assert_same_values, resume_case, skip_fit_iterations

from entlearn import Coupling, Network, PredictConfig
from entlearn.primitives.normalise import _tiny

pytestmark = FIXED_BUDGET


class TestParameterTransfer:
    @pytest.mark.parametrize("task", ["classification", "regression"])
    @pytest.mark.parametrize("kind", ["standard", "categorical", "manifold"])
    @pytest.mark.parametrize("coupling", [Coupling.M, Coupling.S])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_all_active_parameters_transfer_exactly(self, monkeypatch, task, kind, coupling, dtype):
        recipe, X, y, cats = resume_case(task, kind, coupling)
        source = Network.fit(recipe, X, y, X_cat=cats, max_iter=3, seed=5)
        skip_fit_iterations(monkeypatch)
        result = source.fine_tune(
            X[:2], y[:2], X_cat=tuple(c[:2] for c in cats), computation_dtype=dtype, max_iter=1
        )
        names = ["continuous_centroids", "transition_matrices", "head_parameters"]
        names += ["manifold_projectors"] if kind == "manifold" else ["feature_weights"]
        if cats:
            names.append("categorical_centroids")
        for name in names:
            assert_same_values(result.inspect(name), source.inspect(name), dtype=dtype)
        assert result.schema.computation_dtype == dtype
        assert result.schema.K_active == source.schema.K_active
        assert result.can_resume


class TestFineTunePrecision:
    @pytest.mark.parametrize(
        "change,soft", [("rows", False), ("temperature", True), ("dtype", True)]
    )
    def test_regimes_resolve_before_calibration_and_prediction(self, monkeypatch, change, soft):
        recipe, X, y, _ = resume_case()
        dtype = torch.float32
        X = X.to(dtype)
        eps = torch.finfo(dtype).eps
        # A positive temperature below float32's tiny is soft in both dtypes
        start = _tiny(dtype) / 2 if change == "dtype" else 0.0
        recipe = replace(
            recipe,
            blocks=(
                replace(recipe.blocks[0], epsilon=start),
                replace(recipe.blocks[1], epsilon=start),
                recipe.blocks[-1],
            ),
        )
        source = Network.fit(recipe, X, y, max_iter=1, predict_config=PredictConfig(epsilon_P=0.2))
        initial = "soft" if change == "dtype" else "hard"
        hard = {"input": initial, "hidden_1": initial}
        assert source.inspect("affiliation_regimes") == hard
        skip_fit_iterations(monkeypatch)
        if change == "temperature":
            recipe = replace(
                recipe,
                blocks=(
                    replace(recipe.blocks[0], epsilon=100 * eps),
                    replace(recipe.blocks[1], epsilon=100 * eps),
                    recipe.blocks[-1],
                ),
            )
        count = 2 if change == "rows" else len(X)
        result = source.fine_tune(
            X[:count],
            y[:count],
            recipe=recipe,
            computation_dtype=torch.float64 if change == "dtype" else dtype,
            max_iter=1,
        )
        # Only the temperature decides the regime: neither the rows nor the dtype change it.
        regime = "soft" if soft else "hard"
        assert result.inspect("affiliation_regimes") == {"input": regime, "hidden_1": regime}
        assert source.inspect("affiliation_regimes") == hard
        assert result.predict_config.epsilon_P == 0.2
        query = X.to(result.schema.computation_dtype)
        torch.testing.assert_close(
            result.predict(query), torch.cat([result.predict(row[None]) for row in query])
        )

    def test_float32_operation_under_float64_default(self):
        recipe, X, y, _ = resume_case()
        source = Network.fit(recipe, X.to(torch.float32), y, max_iter=2)
        old = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            result = source.fine_tune(X[:10].to(torch.float32), y[:10], max_iter=2)
            assert result.schema.computation_dtype is torch.float32
            assert (
                result.capture_current_state().input_geometry.continuous_centroids.dtype
                is torch.float32
            )
            assert result.predict(X.to(torch.float32)).dtype is torch.float32
        finally:
            torch.set_default_dtype(old)
