"""Reconstruction uses fitted input geometry and prediction affiliations."""

import pytest
import torch
from conftest import DEVICE, DTYPE
from network._fixtures import prediction_model

from entlearn import (
    Coupling,
    Input,
    ManifoldInput,
    Network,
    PredictConfig,
    Recipe,
    ReconstructionResult,
    RegressionHead,
)


class TestManifoldReconstruction:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_single_cluster_reconstructs_the_principal_line(self, dtype):
        X = (
            torch.tensor(
                [[-3.0, 1.0], [-1.0, -1.0], [1.0, -1.0], [3.0, 1.0]], dtype=dtype, device=DEVICE
            )
            / 12
            + 5 / 12
        )
        model = Network.fit(
            Recipe.chain(ManifoldInput(K=1), RegressionHead()), X, X[:, 0], max_iter=3
        )
        query = torch.tensor([[5.0, 7.0], [-5.0, -2.0]], dtype=dtype, device=X.device) / 12 + 5 / 12
        expected = (
            torch.tensor([[5.0, 0.0], [-5.0, 0.0]], dtype=dtype, device=X.device) / 12 + 5 / 12
        )
        original_query = query.clone()
        torch.testing.assert_close(model.reconstruct(query).continuous, expected)
        reconstructed = model.reconstruct(query)
        assert isinstance(reconstructed, ReconstructionResult)
        assert reconstructed.categorical == ()
        with torch.inference_mode():
            reconstructed.continuous.zero_()
        torch.testing.assert_close(model.reconstruct(query).continuous, expected)
        torch.testing.assert_close(query, original_query)

    def test_reconstruction_blends_local_projections_and_preserves_query_batching(self):
        X = (
            torch.tensor(
                [[-3.0, 0.0], [-2.0, 0.0], [-1.0, 0.0], [1.0, 4.0], [2.0, 4.0], [3.0, 4.0]],
                dtype=DTYPE,
                device=DEVICE,
            )
            / 8
            + 3 / 8
        )
        model = Network.fit(
            Recipe.chain(ManifoldInput(K=2, epsilon=0.001 / 64), RegressionHead()),
            X,
            X[:, 1],
            max_iter=1,
            seed=7,
        )
        # Hard-separated clouds yield horizontal affine subspaces at heights 3/8 and 7/8.
        # The midpoint has equal costs and must blend local projections, not select either.
        query = torch.tensor([[0.0, 2.0], [2.0, 3.0]], dtype=X.dtype, device=X.device) / 8 + 3 / 8
        together = model.reconstruct(query).continuous
        apart = torch.cat([model.reconstruct(row[None]).continuous for row in query])
        torch.testing.assert_close(together, apart)
        torch.testing.assert_close(
            together,
            torch.tensor([[0.0, 2.0], [2.0, 4.0]], dtype=X.dtype, device=X.device) / 8 + 3 / 8,
        )

    @pytest.mark.parametrize("shape", [(8, 3), (3, 8)])
    def test_full_dimensional_subspaces_reconstruct_unseen_rows_exactly(self, shape):
        generator = torch.Generator(device=DEVICE).manual_seed(19)
        X = torch.rand(*shape, dtype=DTYPE, device=DEVICE, generator=generator)
        model = Network.fit(
            Recipe.chain(
                ManifoldInput(K=2, subspace_dimension=shape[1], epsilon=0.4),
                RegressionHead(),
            ),
            X,
            X[:, 0],
            max_iter=2,
        )
        query = torch.rand(5, shape[1], dtype=X.dtype, device=X.device, generator=generator)
        tolerance = 1e-12 if X.dtype is torch.float64 else 50 * torch.finfo(X.dtype).eps
        torch.testing.assert_close(
            model.reconstruct(query).continuous, query, atol=tolerance, rtol=tolerance
        )


class TestStandardReconstruction:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("epsilon", [0.0, 0.025])
    def test_returns_centroid_blend_with_prediction_affiliations(self, dtype, epsilon):
        X = torch.tensor([[0.0], [1.0]], dtype=dtype, device=DEVICE)
        recipe = Recipe.chain(Input(K=2, epsilon=epsilon), RegressionHead())
        state = Network.initialise(recipe, X, X[:, 0], continuous_centroids=X)
        model = Network.fit(
            recipe, X, torch.zeros(2, dtype=dtype, device=X.device), initial_state=state, max_iter=1
        )
        query = torch.tensor([[0.5], [0.525], [0.45]], dtype=dtype, device=X.device)
        costs = (query - X.T).square()
        gamma = (
            torch.softmax(-costs / epsilon, dim=1)
            if epsilon
            else torch.nn.functional.one_hot(costs.argmin(dim=1), 2).to(dtype)
        )
        expected = gamma @ X

        result = model.reconstruct(query)
        assert isinstance(result, ReconstructionResult)
        assert result.categorical == ()
        together = result.continuous
        apart = torch.cat([model.reconstruct(row[None]).continuous for row in query])
        torch.testing.assert_close(together, expected)
        torch.testing.assert_close(apart, expected)
        with torch.inference_mode():
            together.zero_()
        torch.testing.assert_close(model.reconstruct(query).continuous, expected)
        assert not together.requires_grad

    def test_mixed_reconstruction_retains_category_distributions(self):
        X = torch.tensor([[0.0], [1.0]], dtype=DTYPE, device=DEVICE)
        codes = torch.tensor([0, 1], device=X.device)
        model = Network.fit(
            Recipe.chain(Input(K=1), RegressionHead()), X, X[:, 0], X_cat=[codes], max_iter=1
        )
        result = model.reconstruct(X, X_cat=[codes])
        torch.testing.assert_close(result.continuous, torch.full_like(X, 0.5))
        torch.testing.assert_close(
            result.categorical[0], torch.full((2, 2), 0.5, dtype=DTYPE, device=DEVICE)
        )


class TestDetailedReconstruction:
    @pytest.mark.parametrize(
        ("task", "head", "readout"),
        [
            ("classification", Coupling.M, "geometric"),
            ("classification", Coupling.M, "arithmetic"),
            ("classification", Coupling.S, "geometric"),
            ("regression", Coupling.M, None),
        ],
    )
    @pytest.mark.parametrize(
        "kind", ["continuous", "mixed", "distribution", "categorical", "manifold"]
    )
    @pytest.mark.parametrize("mode", ["single", "iterative"])
    def test_standalone_and_details_share_affiliations_and_ownership(
        self, task, kind, mode, head, readout
    ):
        model, X, categories = prediction_model(task, input_kind=kind, head_coupling=head)
        config = PredictConfig(predict_mode=mode, output_mode=readout, max_iter=3, tol=0)
        detailed = model.predict_with_details(
            X,
            X_cat=categories,
            predict_config=config,
            details=("reconstruction", "affiliations", "diagnostics"),
        )
        alone = model.reconstruct(X, X_cat=categories, predict_config=config)
        diagnostics = model.predict_with_details(
            X, X_cat=categories, predict_config=config, details=("diagnostics",)
        )
        assert detailed.n_iter == diagnostics.n_iter
        assert detailed.converged == diagnostics.converged
        assert detailed.loss_history == diagnostics.loss_history
        assert detailed.n_iter == len(detailed.loss_history)
        minimal = model.predict_with_details(
            X, X_cat=categories, predict_config=config, details=("reconstruction",)
        )
        assert minimal.n_iter is None and minimal.converged is None and minimal.loss_history is None
        assert diagnostics.reconstruction is None
        torch.testing.assert_close(
            detailed.prediction, model.predict(X, X_cat=categories, predict_config=config)
        )
        input_name = model.recipe.blocks[0].name
        gamma = detailed.affiliations[input_name]
        reconstruction = detailed.reconstruction
        # Every input kind returns the same record; only the categorical tuple's length varies.
        assert isinstance(alone, ReconstructionResult)
        assert isinstance(reconstruction, ReconstructionResult)
        assert len(alone.categorical) == len(reconstruction.categorical) == len(categories)
        pairs = [(alone.continuous, reconstruction.continuous)]
        pairs.extend(zip(alone.categorical, reconstruction.categorical, strict=True))
        if categories:
            centroids = model.inspect("categorical_centroids")[input_name]
            for value, centre in zip(alone.categorical, centroids, strict=True):
                torch.testing.assert_close(value, gamma @ centre)
                torch.testing.assert_close(
                    value.sum(dim=1), torch.ones(X.shape[0], dtype=DTYPE, device=DEVICE)
                )
        for standalone, detail in pairs:
            torch.testing.assert_close(standalone, detail)
            assert not detail.requires_grad
            saved = standalone.clone()
            with torch.inference_mode():
                detail.zero_()
            torch.testing.assert_close(standalone, saved)
        if kind != "manifold":
            torch.testing.assert_close(
                alone.continuous, gamma @ model.inspect("continuous_centroids")[input_name]
            )

    @pytest.mark.parametrize("task", ["classification", "regression"])
    @pytest.mark.parametrize("mode", ["single", "iterative"])
    def test_combined_request_runs_one_engine_and_rejects_bad_details_before_it(
        self, monkeypatch, task, mode
    ):
        # Instrument only the entry count: numerical assertions stay on public results.
        from entlearn.network import predict as engine

        model, X, categories = prediction_model(task, input_kind="mixed")
        forward = engine._forward
        calls = []

        def counted(*args, **kwargs):
            calls.append(1)
            return forward(*args, **kwargs)

        monkeypatch.setattr(engine, "_forward", counted)
        result = model.predict_with_details(
            X,
            X_cat=categories,
            details=("reconstruction", "diagnostics"),
            predict_config=PredictConfig(predict_mode=mode, max_iter=2),
        )
        assert len(calls) == 1
        assert result.n_iter == len(result.loss_history)
        assert result.reconstruction is not None
        for name in ("unsupported", "n_iter", "converged", "loss_history"):
            with pytest.raises(ValueError, match="detail"):
                model.predict_with_details(None, details=("reconstruction", name))
        assert len(calls) == 1


class TestReconstructionValidation:
    @pytest.mark.parametrize("input_type", [Input, ManifoldInput])
    @pytest.mark.parametrize("method", ["predict", "reconstruct"])
    @pytest.mark.parametrize("bad", ["dtype", "width", "nan", "empty", "device"])
    def test_queries_share_strict_validation(self, input_type, method, bad):
        X = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=DTYPE, device=DEVICE)
        model = Network.fit(Recipe.chain(input_type(K=1), RegressionHead()), X, X[:, 0], max_iter=1)
        query = X.clone()
        if bad == "dtype":
            query = query.to(torch.float32 if X.dtype is torch.float64 else torch.float64)
        elif bad == "width":
            query = query[:, :1]
        elif bad == "nan":
            query[0, 0] = float("nan")
        elif bad == "empty":
            query = query[:0]
        else:
            query = torch.empty_like(query, device="meta")
        with pytest.raises(ValueError):
            getattr(model, method)(query)
