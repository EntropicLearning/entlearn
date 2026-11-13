"""Reconstruction uses fitted input geometry and prediction affiliations."""

import pytest
import torch
from conftest import DEVICE, DTYPE, FIXED_BUDGET, RANK_DEFICIENT
from network._fixtures import prediction_model

from entlearn import (
    Coupling,
    Input,
    ManifoldInput,
    Network,
    PredictConfig,
    Recipe,
    RegressionHead,
)

pytestmark = FIXED_BUDGET


class TestManifoldReconstruction:
    def test_single_cluster_reconstructs_the_principal_line(self):
        X = (
            torch.tensor(
                [[-3.0, 1.0], [-1.0, -1.0], [1.0, -1.0], [3.0, 1.0]], dtype=DTYPE, device=DEVICE
            )
            / 12
            + 5 / 12
        )
        model = Network.fit(
            Recipe.chain(ManifoldInput(K=1), RegressionHead()), X, X[:, 0], max_iter=3
        )
        query = torch.tensor([[5.0, 7.0], [-5.0, -2.0]], dtype=DTYPE, device=DEVICE) / 12 + 5 / 12
        expected = torch.tensor([[5.0, 0.0], [-5.0, 0.0]], dtype=DTYPE, device=DEVICE) / 12 + 5 / 12
        original_query = query.clone()
        torch.testing.assert_close(model.reconstruct(query).continuous, expected)
        reconstructed = model.reconstruct(query)
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

    @RANK_DEFICIENT
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
        # A full-dimensional basis gives P Pᵀ = I up to its orthonormality error, a few eps
        # per feature; 4 D eps bounds the D-term sum (about 2 eps is observed).
        tolerance = 4 * shape[1] * torch.finfo(DTYPE).eps
        torch.testing.assert_close(
            model.reconstruct(query).continuous, query, atol=tolerance, rtol=0
        )


class TestStandardReconstruction:
    @pytest.mark.parametrize("epsilon", [0.0, 0.025])
    def test_returns_centroid_blend_with_prediction_affiliations(self, epsilon):
        X = torch.tensor([[0.0], [1.0]], dtype=DTYPE, device=DEVICE)
        recipe = Recipe.chain(Input(K=2, epsilon=epsilon), RegressionHead())
        state = Network.initialise(recipe, X, X[:, 0], continuous_centroids=X)
        model = Network.fit(
            recipe, X, torch.zeros(2, dtype=DTYPE, device=DEVICE), initial_state=state, max_iter=1
        )
        query = torch.tensor([[0.5], [0.525], [0.45]], dtype=DTYPE, device=DEVICE)
        costs = (query - X.T).square()
        gamma = (
            torch.softmax(-costs / epsilon, dim=1)
            if epsilon
            else torch.nn.functional.one_hot(costs.argmin(dim=1), 2).to(DTYPE)
        )
        expected = gamma @ X

        result = model.reconstruct(query)
        assert result.categorical == ()
        together = result.continuous
        apart = torch.cat([model.reconstruct(row[None]).continuous for row in query])
        torch.testing.assert_close(together, expected)
        torch.testing.assert_close(apart, expected)
        with torch.inference_mode():
            together.zero_()
        torch.testing.assert_close(model.reconstruct(query).continuous, expected)

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


def _blended_projections(X, centroids, projectors, gamma):
    """Blend each cluster's projection c + P Pᵀ (x - c) with the affiliations."""
    offsets = X[:, None, :] - centroids[None]
    projected = centroids[None] + torch.einsum("tkd,kde,kfe->tkf", offsets, projectors, projectors)
    return torch.einsum("tk,tkf->tf", gamma, projected)


# Single-mode affiliations come from the forward pass, which the head and readout never
# reach; iterative refinement reads both, so only that mode crosses them.
_HEAD_CASES = [
    ("classification", Coupling.M, "geometric"),
    ("classification", Coupling.M, "arithmetic"),
    ("classification", Coupling.S, "geometric"),
    ("regression", Coupling.M, None),
]


class TestDetailedReconstruction:
    @RANK_DEFICIENT
    @pytest.mark.parametrize(
        ("mode", "task", "head", "readout"),
        [("single", *_HEAD_CASES[0]), *(("iterative", *case) for case in _HEAD_CASES)],
    )
    @pytest.mark.parametrize(
        "kind", ["continuous", "mixed", "distribution", "categorical", "manifold"]
    )
    def test_reconstruction_blends_the_fitted_geometry_with_the_prediction_affiliations(
        self, task, kind, mode, head, readout
    ):
        model, X, categories = prediction_model(task, input_kind=kind, head_coupling=head)
        config = PredictConfig(predict_mode=mode, output_mode=readout, max_iter=3, tol=0)
        detailed = model.predict_with_details(
            X,
            X_cat=categories,
            predict_config=config,
            details=("reconstruction", "affiliations"),
        )
        name = model.recipe.blocks[0].name
        gamma = detailed.affiliations[name]
        reconstruction = detailed.reconstruction
        assert reconstruction is not None
        centroids = model.inspect("continuous_centroids")[name]
        if kind == "manifold":
            projectors = model.inspect("manifold_projectors")[name]
            expected = _blended_projections(X, centroids, projectors, gamma)
        else:
            expected = gamma @ centroids
        torch.testing.assert_close(reconstruction.continuous, expected)
        expected_categorical = (
            [gamma @ centre for centre in model.inspect("categorical_centroids")[name]]
            if categories
            else []
        )
        assert len(reconstruction.categorical) == len(expected_categorical)
        for value, wanted in zip(reconstruction.categorical, expected_categorical, strict=True):
            torch.testing.assert_close(value, wanted)


def _query_case(input_type, bad):
    """Fit a one-cluster model and return it with a query broken by ``bad`` and its reason."""
    X = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=DTYPE, device=DEVICE)
    model = Network.fit(Recipe.chain(input_type(K=1), RegressionHead()), X, X[:, 0], max_iter=1)
    query = X.clone()
    if bad == "dtype":
        query = query.to(torch.float32 if X.dtype is torch.float64 else torch.float64)
        message = "fitted computation dtype"
    elif bad == "width":
        query = query[:, :1]
        message = "wrong continuous feature count"
    elif bad == "nan":
        query[0, 0] = float("nan")
        message = "X_cont contains NaN"
    elif bad == "empty":
        query = query[:0]
        message = "X_cont must be a non-empty"
    else:
        query = torch.empty_like(query, device="meta")
        message = "device that contains data"
    return model, query, message


class TestReconstructionValidation:
    @pytest.mark.parametrize("input_type", [Input, ManifoldInput])
    @pytest.mark.parametrize("bad", ["dtype", "width", "nan", "empty", "device"])
    def test_queries_share_strict_validation(self, input_type, bad):
        model, query, message = _query_case(input_type, bad)
        with pytest.raises(ValueError, match=message):
            model.predict(query)

    def test_reconstruct_reaches_the_query_validation(self):
        model, query, message = _query_case(ManifoldInput, "width")
        with pytest.raises(ValueError, match=message):
            model.reconstruct(query)
