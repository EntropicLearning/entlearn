"""Manifold geometry and fitting through the tensor lifecycle."""

import math
from contextlib import nullcontext
from dataclasses import replace
from itertools import pairwise

import pytest
import torch
from _alloc import assert_no_float64
from conftest import DEVICE, DTYPE
from network._fixtures import affine_subspace_blobs, assert_simplex_rows, manifold_line

from entlearn import (
    ClassificationHead,
    Coupling,
    Hidden,
    Input,
    ManifoldInput,
    Network,
    Recipe,
    RegressionHead,
)
from entlearn.primitives.normalise import _eps


class TestManifoldFit:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("strategy", ["kmeans++", "greedy-kmeans++"])
    def test_manifold_seeds_use_uniform_scaling_and_local_subspaces(self, dtype, strategy):
        X = torch.tensor(
            [[-3.0, 0.0], [-2.0, 0.0], [-1.0, 0.0], [1.0, 5.0], [2.0, 5.0], [3.0, 5.0]],
            dtype=dtype,
            device=DEVICE,
        )
        y = X[:, 0]
        controls = dict(K=2, centroid_strategy=strategy)
        manifold = Recipe.chain(ManifoldInput(subspace_dimension=1, **controls), RegressionHead())
        standard = Recipe.chain(Input(**controls), RegressionHead())
        state = Network.initialise(manifold, X, y, seed=7)
        reference = Network.initialise(standard, X, y, seed=7)
        torch.testing.assert_close(
            state.input_geometry.continuous_centroids, reference.input_geometry.continuous_centroids
        )
        assert state.input_geometry.feature_weights is None
        assert state.input_geometry.categorical_centroids == ()
        projectors = state.input_geometry.manifold_projectors
        assert projectors is not None
        projection = projectors @ projectors.transpose(1, 2)
        expected = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=dtype, device=X.device).expand(
            2, -1, -1
        )
        torch.testing.assert_close(projection, expected)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_supplied_centroids_derive_deterministically_completed_projectors(self, dtype):
        X = torch.zeros(8, 3, dtype=dtype, device=DEVICE)
        y = torch.arange(8, dtype=dtype, device=DEVICE)
        recipe = Recipe.chain(ManifoldInput(K=1, subspace_dimension=2), RegressionHead())
        centre = torch.zeros(1, 3, dtype=dtype, device=X.device)
        with pytest.warns(UserWarning, match="insufficient rank"):
            states = [
                Network.initialise(recipe, X, y, continuous_centroids=centre, seed=seed)
                for seed in (5, 5, 6)
            ]
        a, b, c = [state.input_geometry.manifold_projectors for state in states]
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        torch.testing.assert_close(a, c, atol=0, rtol=0)
        torch.testing.assert_close(
            a.transpose(1, 2) @ a, torch.eye(2, dtype=dtype, device=X.device).unsqueeze(0)
        )
        centre.fill_(100)
        torch.testing.assert_close(
            states[0].input_geometry.continuous_centroids,
            torch.zeros(1, 3, dtype=dtype, device=X.device),
        )

    @pytest.mark.parametrize("operation", ["initialise", "fit"])
    def test_manifold_rejects_categorical_features_and_excess_dimension(self, operation):
        X = torch.zeros(4, 2, dtype=DTYPE, device=DEVICE)
        y = X[:, 0]
        recipe = Recipe.chain(ManifoldInput(K=1), RegressionHead())
        with pytest.raises(ValueError, match="continuous-only"):
            getattr(Network, operation)(
                recipe, X, y, X_cat=[torch.tensor([0, 1, 0, 1], device=X.device)]
            )
        oversized = Recipe.chain(ManifoldInput(K=1, subspace_dimension=3), RegressionHead())
        with pytest.raises(ValueError, match="subspace_dimension"):
            getattr(Network, operation)(oversized, X, y)

    def test_manifold_rejects_feature_weights(self):
        X = torch.zeros(4, 2, dtype=DTYPE, device=DEVICE)
        recipe = Recipe.chain(ManifoldInput(K=1), RegressionHead())
        with pytest.raises(ValueError, match="feature_weights"):
            Network.initialise(
                recipe, X, X[:, 0], feature_weights=torch.ones(2, dtype=X.dtype, device=X.device)
            )

    def test_manifold_fits_the_global_mean_regression(self):
        X = torch.tensor(
            [[-3.0, 1.0], [-1.0, -1.0], [1.0, -1.0], [3.0, 1.0]], dtype=DTYPE, device=DEVICE
        )
        y = torch.tensor([1.0, 2.0, 3.0, 6.0], dtype=X.dtype, device=X.device)
        recipe = Recipe.chain(ManifoldInput(K=1), RegressionHead())
        model = Network.fit(recipe, X, y, max_iter=3, seed=3)
        torch.testing.assert_close(
            model.predict(X), torch.full((4, 1), 3.0, dtype=X.dtype, device=X.device)
        )
        assert model.diagnostics.loss_history[-1] <= model.diagnostics.loss_history[0]

    @pytest.mark.parametrize("task", ["classification", "regression"])
    @pytest.mark.parametrize("coupling", [Coupling.M, Coupling.S])
    @pytest.mark.parametrize("widths", [(), (2,), (5, 2)])
    @pytest.mark.parametrize("epsilon", [0.0, 0.1])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_manifold_fits_both_tasks_and_couplings_with_uniform_weight_invariance(
        self, task, coupling, widths, epsilon, dtype
    ):
        generator = torch.Generator().manual_seed(31)  # CPU stream: same data on every lane
        X = torch.randn(24, 3, generator=generator, dtype=dtype).to(DEVICE) * 0.1
        X[12:] += 1
        labels = torch.tensor([0] * 12 + [1] * 12, device=X.device)
        y = labels if task == "classification" else torch.stack((labels.to(dtype), X[:, 1]), dim=1)
        head = (
            ClassificationHead(coupling=coupling)
            if task == "classification"
            else RegressionHead(epsilon_M=0.5)
        )
        recipe = Recipe.chain(
            ManifoldInput(K=3, subspace_dimension=1, epsilon=epsilon, epsilon_T=0.5),
            *(Hidden(K=width, epsilon=0.1) for width in widths),
            head,
            coupling=coupling if widths else None,
            theta_alpha=1.1,
        )
        state = Network.initialise(recipe, X, y, seed=17)
        first = Network.fit(recipe, X, y, initial_state=state, max_iter=5, tol=0)
        weighted = Network.fit(
            recipe,
            X,
            y,
            initial_state=state,
            sample_weights=torch.ones(X.shape[0], dtype=dtype, device=X.device),
            max_iter=5,
            tol=0,
        )
        prediction = first.predict(X)
        torch.testing.assert_close(prediction, weighted.predict(X))
        if task == "classification":
            assert_simplex_rows(prediction)
            if epsilon > 0:
                # Hard affiliations (epsilon=0) can settle on a poor local optimum whose
                # identity follows the device's seeded draw (seeds.py: replay is
                # device-specific), so a fixed seed cannot pin fit quality across lanes.
                assert (prediction.argmax(1) == y).to(dtype).mean() > 0.8
        else:
            assert (prediction - y).square().mean() < (y - y.mean(0)).square().mean()
        tolerance = 100 * _eps(dtype)
        history = first.diagnostics.loss_history
        assert all(b <= a + tolerance for a, b in pairwise(history))
        assert first.reconstruct(X).continuous.dtype == dtype
        assert state.input_geometry.feature_weights is None

    @pytest.mark.parametrize(
        "bad",
        [
            "missing",
            "shape",
            "nan",
            "zero",
            "nonorthogonal",
            "integer",
            "sparse",
            "meta",
            "feature_weights",
            "categorical",
        ],
    )
    def test_supplied_manifold_state_rejects_invalid_geometry(self, bad):
        X = manifold_line()
        recipe = Recipe.chain(ManifoldInput(K=1), RegressionHead())
        state = Network.initialise(recipe, X, X[:, 0])
        projectors = state.input_geometry.manifold_projectors.clone()
        if bad == "missing":
            state = replace(
                state, input_geometry=replace(state.input_geometry, manifold_projectors=None)
            )
        elif bad == "feature_weights":
            state = replace(
                state,
                input_geometry=replace(
                    state.input_geometry,
                    feature_weights=torch.ones(2, dtype=X.dtype, device=X.device),
                ),
            )
        elif bad == "categorical":
            state = replace(
                state,
                input_geometry=replace(
                    state.input_geometry,
                    categorical_centroids=(torch.ones(1, 2, dtype=X.dtype, device=X.device),),
                ),
            )
        else:
            if bad == "shape":
                projectors = projectors[:, :1]
            elif bad == "nan":
                projectors.fill_(float("nan"))
            elif bad == "zero":
                projectors.zero_()
            elif bad == "nonorthogonal":
                projectors.mul_(2)
            elif bad == "integer":
                projectors = projectors.to(torch.int64)
            elif bad == "sparse":
                projectors = projectors.to_sparse()
            elif bad == "meta":
                projectors = projectors.to("meta")
            state = replace(
                state, input_geometry=replace(state.input_geometry, manifold_projectors=projectors)
            )
        with pytest.raises(ValueError, match=r"manifold_projectors|feature_weights|categorical"):
            Network.fit(recipe, X, X[:, 0], initial_state=state, max_iter=1)

    @pytest.mark.parametrize(
        ("excess", "outcome"),
        [
            (32, nullcontext()),
            (128, pytest.raises(ValueError, match="must be orthonormal")),
        ],
    )
    def test_supplied_projector_orthonormality_has_an_absolute_bound(self, excess, outcome):
        # The bound is 32 float32 eps per feature, and D = 2 here.
        X = manifold_line(dtype=torch.float64)
        recipe = Recipe.chain(ManifoldInput(K=1), RegressionHead())
        state = Network.initialise(recipe, X, X[:, 0])
        geometry = state.input_geometry
        scale = math.sqrt(1 + excess * _eps(torch.float32))
        projectors = geometry.manifold_projectors * scale
        state = replace(state, input_geometry=replace(geometry, manifold_projectors=projectors))
        with outcome:
            Network.fit(recipe, X, X[:, 0], initial_state=state, max_iter=1)

    def test_supplied_pair_is_copied_and_not_rederived(self):
        X = manifold_line()
        recipe = Recipe.chain(ManifoldInput(K=1), RegressionHead())
        state = Network.initialise(recipe, X, X[:, 0])
        # A supplied vertical projector deliberately disagrees with the initial PCA subspace.
        supplied = replace(
            state,
            input_geometry=replace(
                state.input_geometry,
                manifold_projectors=torch.tensor([[[0.0], [1.0]]], dtype=X.dtype, device=X.device),
            ),
        )
        fitted = Network.fit(recipe, X, X[:, 0], initial_state=supplied, max_iter=1)
        ordinary = Network.fit(recipe, X, X[:, 0], initial_state=state, max_iter=1)
        assert fitted.diagnostics.loss_history[0] > ordinary.diagnostics.loss_history[0]
        torch.testing.assert_close(fitted.predict(X), ordinary.predict(X))
        expected = fitted.reconstruct(X).continuous.clone()
        supplied.input_geometry.manifold_projectors.zero_()
        torch.testing.assert_close(fitted.reconstruct(X).continuous, expected)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_empty_manifold_cluster_is_pruned_before_centroid_update(self, dtype):
        X = manifold_line(dtype=dtype)
        recipe = Recipe.chain(
            ManifoldInput(K=2, subspace_dimension=1),
            Hidden(K=2, epsilon=0.1),
            RegressionHead(),
            coupling=Coupling.S,
        )
        state = Network.initialise(
            recipe,
            X,
            X[:, 0],
            continuous_centroids=torch.tensor(
                [[0.0, 0.0], [1.0, 1.0]], dtype=dtype, device=X.device
            ),
        )
        model = Network.fit(recipe, X, X[:, 0], initial_state=state, max_iter=3)
        assert dict(model.schema.K_active)["input"] == 1
        query = torch.tensor([[0.7, 0.3], [0.05, 0.2]], dtype=dtype, device=X.device)
        torch.testing.assert_close(
            model.reconstruct(query).continuous,
            torch.tensor([[0.7, 0.0], [0.05, 0.0]], dtype=dtype, device=X.device),
        )
        assert torch.isfinite(model.predict(query)).all()
        assert all(torch.isfinite(torch.tensor(model.diagnostics.loss_history)))

    def test_manifold_float32_never_promotes_and_respects_device(self):
        X = (
            torch.tensor(
                [[-3.0, 1.0], [-1.0, -1.0], [1.0, -1.0], [3.0, 1.0]],
                dtype=torch.float32,
                device=DEVICE,
            )
            / 6
            + 0.5
        )
        recipe = Recipe.chain(ManifoldInput(K=1), RegressionHead())
        original = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            state = assert_no_float64(Network.initialise, recipe, X, X[:, 0])
            fitted = assert_no_float64(
                Network.fit, recipe, X, X[:, 0], initial_state=state, max_iter=2
            )
            predicted = assert_no_float64(fitted.predict, X)
            reconstructed = assert_no_float64(fitted.reconstruct, X).continuous
        finally:
            torch.set_default_dtype(original)
        assert predicted.dtype is torch.float32
        assert reconstructed.dtype is torch.float32
        assert state.input_geometry.manifold_projectors.dtype is torch.float32
        assert reconstructed.device == DEVICE

    def test_supplied_projectors_convert_only_with_explicit_computation_dtype(self):
        X = (
            torch.tensor(
                [[-3.0, 1.0], [-1.0, -1.0], [1.0, -1.0], [3.0, 1.0]],
                dtype=torch.float64,
                device=DEVICE,
            )
            / 6
            + 0.5
        )
        recipe = Recipe.chain(ManifoldInput(K=1), RegressionHead())
        state = Network.initialise(recipe, X, X[:, 0])
        mixed = replace(
            state,
            input_geometry=replace(
                state.input_geometry,
                manifold_projectors=state.input_geometry.manifold_projectors.float(),
            ),
        )
        with pytest.raises(ValueError, match="computation dtype"):
            Network.fit(recipe, X, X[:, 0], initial_state=mixed, max_iter=1)
        converted = Network.fit(
            recipe, X, X[:, 0], initial_state=state, computation_dtype=torch.float32, max_iter=2
        )
        native = Network.fit(recipe, X.float(), X[:, 0].float(), max_iter=2)
        torch.testing.assert_close(
            converted.reconstruct(X.float()).continuous, native.reconstruct(X.float()).continuous
        )


class TestManifoldPCA:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("scale", [1.0, 1e-8, 1e-17])
    @pytest.mark.parametrize("shape", [(24, 4), (8, 12)])
    @pytest.mark.parametrize("weighted", [False, True], ids=["standard-pca", "weighted-pca"])
    def test_single_cluster_recovers_independent_centred_svd(self, dtype, scale, shape, weighted):
        X, _ = affine_subspace_blobs(
            n_subspaces=1,
            rows_per_subspace=shape[0],
            n_features=shape[1],
            subspace_dimension=2,
            orthogonal_noise=0.03,
            dtype=dtype,
        )
        X = X * scale
        weights = torch.arange(1, shape[0] + 1, dtype=dtype, device=DEVICE) if weighted else None
        model = Network.fit(
            Recipe.chain(
                ManifoldInput(K=1, subspace_dimension=2, epsilon_T=float("inf")),
                RegressionHead(),
            ),
            X,
            X[:, 0],
            sample_weights=weights,
            max_iter=2,
        )
        # Independent PCA oracle: no entlearn primitive or fitted projector is used.
        q = torch.ones(shape[0], dtype=dtype, device=DEVICE) if weights is None else weights
        q = q / q.sum()
        centre = q @ X
        _, singular_values, Vh = torch.linalg.svd(q.sqrt()[:, None] * (X - centre))
        assert singular_values[1] > 2 * singular_values[2]
        basis = Vh[:2].T
        query = X[:4] + torch.linspace(-0.4, 0.6, shape[1], dtype=dtype, device=DEVICE)
        expected = centre + ((query - centre) @ basis) @ basis.T
        torch.testing.assert_close(model.reconstruct(query).continuous, expected)


class TestAffineSubspaceData:
    @pytest.mark.parametrize("n_subspaces", [1, 2])
    @pytest.mark.parametrize("orthogonal_noise", [0.0, 0.05], ids=["aligned", "noisy"])
    def test_manifold_fits_rotated_affine_subspace_blobs(self, n_subspaces, orthogonal_noise):
        X, labels = affine_subspace_blobs(
            n_subspaces=n_subspaces,
            subspace_dimension=2,
            orthogonal_noise=orthogonal_noise,
        )
        for index in range(n_subspaces):
            local = X[labels == index]
            singular_values = torch.linalg.svdvals(local - local.mean(dim=0))
            ratio = singular_values[-1] / singular_values[0]
            assert ratio < 1e-6 if orthogonal_noise == 0 else ratio > 1e-3
        model = Network.fit(
            Recipe.chain(
                ManifoldInput(K=n_subspaces, subspace_dimension=2),
                RegressionHead(),
            ),
            X,
            X[:, 0],
            max_iter=4,
        )
        assert torch.isfinite(model.reconstruct(X).continuous).all()
        assert torch.isfinite(model.predict(X)).all()
