"""Computation dtype and device contract through Network initialisation."""

from dataclasses import replace

import pytest
import torch
from conftest import DEVICE, DTYPE, FIXED_BUDGET
from network._fixtures import (
    assert_same_checkpoint,
    blobs,
    classification_recipe,
    deep_regression_recipe,
    materialise,
    regression_recipe,
)

from entlearn import (
    ClassificationHead,
    Coupling,
    Input,
    ManifoldInput,
    Network,
    Recipe,
    RegressionHead,
)
from entlearn.network.session import _Workspace


class TestComputationDtype:
    @pytest.mark.parametrize("computation_dtype", (torch.float32, torch.float64))
    def test_explicit_dtype_converts_every_floating_boundary_tensor_once(self, computation_dtype):
        source_dtype = torch.float64 if computation_dtype is torch.float32 else torch.float32
        X_cont = torch.arange(8, dtype=source_dtype).reshape(4, 2) / 8
        X_cat = (
            torch.tensor(
                [[1.0, 0.0], [0.25, 0.75], [0.0, 1.0], [0.5, 0.5]],
                dtype=source_dtype,
            ),
        )
        target = torch.tensor(
            [[1.0, 0.0], [0.25, 0.75], [0.0, 1.0], [0.5, 0.5]],
            dtype=source_dtype,
        )
        sample_weights = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=source_dtype)
        class_weights = torch.tensor([2.0, 3.0], dtype=source_dtype)
        originals = tuple(
            tensor.clone() for tensor in (X_cont, X_cat[0], target, sample_weights, class_weights)
        )

        state = Network.initialise(
            classification_recipe(),
            X_cont,
            target,
            X_cat=X_cat,
            sample_weights=sample_weights,
            class_weights=class_weights,
            computation_dtype=computation_dtype,
            seed=4,
        )

        assert state.input_geometry.continuous_centroids.dtype is computation_dtype
        assert state.input_geometry.categorical_centroids[0].dtype is computation_dtype
        assert state.input_geometry.feature_weights is not None
        assert state.input_geometry.feature_weights.dtype is computation_dtype
        for tensor, original in zip(
            (X_cont, X_cat[0], target, sample_weights, class_weights), originals, strict=True
        ):
            assert torch.equal(tensor, original)

    @FIXED_BUDGET
    @pytest.mark.parametrize("weights", ("sample_weights", "class_weights", "task_weights"))
    def test_float64_weights_fit_as_their_float32_values(self, weights):
        # The dtypes are fixed: a float32 fit is the one that cannot run on float64 weights.
        X_cont = torch.linspace(0, 1, 40, dtype=torch.float32, device=DEVICE).reshape(20, 2)
        head = RegressionHead() if weights == "task_weights" else ClassificationHead()
        y = X_cont.sum(dim=1) if weights == "task_weights" else (X_cont[:, 0] > 0.5).long()
        given = torch.linspace(
            0.5, 1.5, 2 if weights == "class_weights" else 20, dtype=torch.float64, device=DEVICE
        )
        recipe = Recipe.chain(Input(K=3), head)

        fitted = Network.fit(recipe, X_cont, y, max_iter=3, **{weights: given})
        converted = Network.fit(recipe, X_cont, y, max_iter=3, **{weights: given.float()})

        assert fitted.diagnostics.loss_history == converted.diagnostics.loss_history

    @pytest.mark.parametrize("computation_dtype", (torch.float16, "float32"))
    def test_rejects_an_invalid_computation_dtype(self, computation_dtype):
        X_cont = torch.arange(4, dtype=torch.float64).unsqueeze(1) / 4
        target = torch.tensor([0, 1, 0, 1], dtype=torch.int64)

        with pytest.raises(ValueError, match=r"computation_dtype must be torch\.float32"):
            Network.initialise(
                classification_recipe(), X_cont, target, computation_dtype=computation_dtype
            )

    def test_a_provided_state_reaches_the_computation_dtype_check(self):
        X_cont = torch.arange(4, dtype=torch.float64, device=DEVICE).unsqueeze(1) / 4
        target = torch.tensor([0, 1, 0, 1], dtype=torch.int64, device=DEVICE)
        state = Network.initialise(classification_recipe(), X_cont, target, seed=3)

        with pytest.raises(ValueError, match=r"computation_dtype must be torch\.float32"):
            Network.fit(
                classification_recipe(),
                X_cont,
                target,
                initial_state=state,
                computation_dtype="float32",
            )

    def test_rejects_unsupported_direct_continuous_dtypes(self):
        X_cont = torch.zeros(4, 1, dtype=torch.int64)
        target = torch.tensor([0, 1, 0, 1], dtype=torch.int64)

        with pytest.raises(ValueError, match="float32 or float64"):
            Network.initialise(classification_recipe(), X_cont, target)

    @pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
    def test_rejects_non_finite_continuous_data(self, value):
        X_cont = torch.arange(4, dtype=torch.float64).unsqueeze(1) / 4
        X_cont[2, 0] = value
        target = torch.tensor([0, 1, 0, 1], dtype=torch.int64)

        with pytest.raises(ValueError, match="X_cont contains NaN, infinity"):
            Network.initialise(classification_recipe(), X_cont, target)

    @pytest.mark.parametrize(
        "X_cont,message",
        (
            pytest.param(
                torch.empty(4, dtype=torch.float64), r"X_cont .*matrix", id="one-dimensional"
            ),
            pytest.param(
                torch.empty(4, 0, dtype=torch.float64),
                "at least one continuous or categorical",
                id="no-features",
            ),
        ),
    )
    def test_rejects_invalid_continuous_shapes(self, X_cont, message):
        target = torch.tensor([0, 1, 0, 1], dtype=torch.int64)

        with pytest.raises(ValueError, match=message):
            Network.initialise(classification_recipe(), X_cont, target)

    def test_rejects_continuous_data_without_rows(self):
        X_cont = torch.empty(0, 1, dtype=torch.float64)
        target = torch.empty(0, dtype=torch.int64)

        with pytest.raises(ValueError, match="X_cont must be a non-empty"):
            Network.initialise(classification_recipe(), X_cont, target)

    @FIXED_BUDGET
    @pytest.mark.parametrize(
        ("group", "subject"),
        [("continuous_centroids", "continuous_centroids"), ("C_y", "regression centroids")],
    )
    def test_a_provided_state_must_stay_finite_in_the_computation_dtype(self, group, subject):
        # A finite float64 value beyond the float32 range becomes infinite on conversion.
        X_cont = torch.arange(8, dtype=torch.float64, device=DEVICE).reshape(4, 2) / 8
        y = X_cont[:, 0].clone()
        recipe = regression_recipe()
        state = Network.fit(recipe, X_cont, y, max_iter=1).capture_current_state()
        huge = 2 * float(torch.finfo(torch.float32).max)
        if group == "continuous_centroids":
            geometry = state.input_geometry
            centroids = torch.full_like(geometry.continuous_centroids, huge)
            state = replace(state, input_geometry=replace(geometry, continuous_centroids=centroids))
        else:
            (head,) = state.parameters
            state = replace(state, parameters=(replace(head, C_y=torch.full_like(head.C_y, huge)),))

        with pytest.raises(ValueError, match=f"{subject} must be finite"):
            Network.fit(
                recipe,
                X_cont,
                y,
                initial_state=state,
                computation_dtype=torch.float32,
                max_iter=1,
            )

    @pytest.mark.parametrize("input_kind", ("mixed", "categorical", "manifold"))
    def test_query_rows_are_priced_exactly_as_the_fit_prices_them(self, input_kind):
        X, codes = blobs(3, 0.28, n_features=4, seed=3)
        if input_kind == "manifold":
            first = ManifoldInput(K=4, subspace_dimension=1, epsilon=0.1)
            categories = ()
        else:
            # Uniform float32 weights of 1/6 on a three-category channel: a channel scale
            # multiplied on the host in float64 and rounded afterwards differs in its last bit.
            first = Input(K=4, epsilon=0.1)
            categories = (codes, (X[:, 0] > 0.5).long())
            X = X if input_kind == "mixed" else X[:, :0]
        recipe = Recipe.chain(first, ClassificationHead(coupling=Coupling.M))
        session = materialise(recipe, X, codes, X_cat=categories, warmup=1)
        block = session.graph.input
        if input_kind != "manifold":
            block.feature_weights.fill_(1 / block.feature_weights.shape[0])
            block.prepare_cache_(session)
        cache = block.live_cache
        workspace = _Workspace.allocate(X.shape[0], block.K, 3, dtype=DTYPE, device=DEVICE)

        total, _, categorical = block.price_query_rows_(
            session.data.X_cont, session.data.X_cat, workspace
        )

        assert torch.equal(total, cache.disc_cost)
        if input_kind == "categorical":
            assert torch.equal(categorical, cache.disc_cost)


class TestDevicePlacement:
    def test_initialises_on_the_active_test_device_and_dtype(self):
        X_cont = torch.arange(8, dtype=DTYPE, device=DEVICE).reshape(4, 2)
        X_cat = (torch.tensor([0, 1, 0, 1], dtype=torch.int64, device=DEVICE),)
        target = torch.tensor([0, 1, 0, 1], dtype=torch.int64, device=DEVICE)

        state = Network.initialise(classification_recipe(), X_cont, target, X_cat=X_cat)

        assert state.input_geometry.continuous_centroids.device == DEVICE
        assert state.input_geometry.continuous_centroids.dtype is DTYPE
        assert state.input_geometry.categorical_centroids[0].device == DEVICE
        assert state.input_geometry.categorical_centroids[0].dtype is DTYPE
        assert state.input_geometry.feature_weights is not None
        assert state.input_geometry.feature_weights.device == DEVICE
        assert state.input_geometry.feature_weights.dtype is DTYPE

    @pytest.mark.parametrize(
        "X_cont,message",
        (
            pytest.param(
                torch.eye(4, dtype=torch.float64).to_sparse(), "strided tensor storage", id="sparse"
            ),
            pytest.param(torch.empty(4, 1, device="meta"), "device that contains data", id="meta"),
        ),
    )
    def test_rejects_non_concrete_or_non_strided_continuous_data(self, X_cont, message):
        target = torch.tensor([0, 1, 0, 1], dtype=torch.int64, device=X_cont.device)

        with pytest.raises(ValueError, match=message):
            Network.initialise(classification_recipe(), X_cont, target)

    @pytest.mark.parametrize("value", ("target", "categorical", "sample", "class"))
    def test_rejects_fit_tensors_on_a_different_device(self, value):
        X_cont = torch.arange(4, dtype=torch.float64).unsqueeze(1)
        target = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
        kwargs = {}
        if value == "target":
            target = torch.empty(4, dtype=torch.int64, device="meta")
        elif value == "categorical":
            kwargs["X_cat"] = [torch.empty(4, dtype=torch.int64, device="meta")]
        elif value == "sample":
            kwargs["sample_weights"] = torch.empty(4, device="meta")
        else:
            kwargs["class_weights"] = torch.empty(2, device="meta")

        with pytest.raises(ValueError, match="device"):
            Network.initialise(classification_recipe(), X_cont, target, **kwargs)

    @pytest.mark.parametrize("value", ("target", "task"))
    def test_rejects_regression_tensors_on_a_different_device(self, value):
        X_cont = torch.arange(4, dtype=torch.float64).unsqueeze(1)
        target = torch.arange(4, dtype=torch.float64)
        task_weights = None
        if value == "target":
            target = torch.empty(4, device="meta")
        else:
            task_weights = torch.empty(4, device="meta")
        recipe = regression_recipe(K=2)

        with pytest.raises(ValueError, match="device"):
            Network.initialise(recipe, X_cont, target, task_weights=task_weights)

    @FIXED_BUDGET
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_a_cuda_fit_repeats_without_deterministic_algorithms(self, dtype):
        # At this many rows CUDA's one-dimensional scans and atomic additions vary from run
        # to run, enough to move seed draws and categorical statistics.
        generator = torch.Generator().manual_seed(0)
        T = 2**20
        X_cont = torch.rand(T, 2, generator=generator, dtype=dtype).cuda()
        X_cat = (torch.randint(4, (T,), generator=generator).cuda(),)
        target = X_cont.sum(dim=1, keepdim=True) + X_cat[0].unsqueeze(1) / 4
        sample_weights = (0.5 + torch.rand(T, generator=generator, dtype=dtype)).cuda()
        # The widening S connection weighs its grouped rows into its initial transitions
        recipe = deep_regression_recipe((16, 32), coupling=Coupling.S)
        deterministic = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        torch.use_deterministic_algorithms(False)
        try:
            first, second = (
                Network.fit(
                    recipe,
                    X_cont,
                    target,
                    X_cat=X_cat,
                    sample_weights=sample_weights,
                    seed=0,
                    max_iter=2,
                )
                for _ in range(2)
            )
        finally:
            torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)

        assert_same_checkpoint(second, first)
