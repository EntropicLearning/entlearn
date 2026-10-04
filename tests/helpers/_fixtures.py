"""Shared data and assertions for optional feature-weight helpers."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import torch


def run_python_probe(source: str) -> object:
    """Run a fresh Python process and decode its JSON result."""
    result = subprocess.run(
        [sys.executable, "-c", source],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def informative_cases() -> dict[
    str,
    tuple[np.ndarray, tuple[np.ndarray, ...], np.ndarray, str, int],
]:
    """Return the four feature-modality and task combinations."""
    cases: dict[str, tuple[np.ndarray, tuple[np.ndarray, ...], np.ndarray, str, int]] = {}

    rng = np.random.default_rng(1)
    target = rng.integers(0, 2, 200)
    continuous = np.column_stack((target + rng.normal(0, 0.2, 200), rng.normal(0, 1, 200)))
    cases["continuous-classification"] = (
        continuous,
        (rng.integers(0, 3, 200),),
        target,
        "classification",
        0,
    )

    rng = np.random.default_rng(2)
    target = rng.integers(0, 3, 200)
    cases["categorical-classification"] = (
        rng.normal(0, 1, 200).reshape(-1, 1),
        (target.copy(), rng.integers(0, 3, 200)),
        target,
        "classification",
        1,
    )

    rng = np.random.default_rng(3)
    informative = rng.normal(0, 1, 200)
    target = 3.0 * informative + rng.normal(0, 0.1, 200)
    cases["continuous-regression"] = (
        np.column_stack((informative, rng.normal(0, 1, 200))),
        (rng.integers(0, 3, 200),),
        target,
        "regression",
        0,
    )

    rng = np.random.default_rng(4)
    informative_codes = rng.integers(0, 4, 240)
    target = informative_codes.astype(float) + rng.normal(0, 0.05, 240)
    cases["categorical-regression"] = (
        rng.normal(0, 1, 240).reshape(-1, 1),
        (informative_codes, rng.integers(0, 4, 240)),
        target,
        "regression",
        1,
    )
    return cases


def mixed_classification_data() -> tuple[np.ndarray, tuple[np.ndarray, ...], np.ndarray]:
    """Return mixed inputs where the first continuous feature carries the class signal."""
    target = np.array([0, 0, 0, 1, 1, 1])
    continuous = np.column_stack(
        (
            target.astype(np.float64),
            np.array([0.2, 0.9, 0.4, 0.6, 0.1, 0.8]),
        )
    )
    categorical = (np.array([0, 1, 0, 1, 0, 1]),)
    return continuous, categorical, target


def classification_tensors(
    X_cont: np.ndarray,
    categorical_codes: tuple[np.ndarray, ...],
    target: np.ndarray,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], torch.Tensor]:
    """Stage helper test data at the Network tensor boundary."""
    return (
        torch.as_tensor(X_cont, dtype=dtype, device=device),
        tuple(
            torch.as_tensor(code, dtype=torch.int64, device=device) for code in categorical_codes
        ),
        torch.as_tensor(target, dtype=torch.int64, device=device),
    )


def two_target_regression_data() -> tuple[np.ndarray, np.ndarray]:
    """Return two features that each drive a different regression output."""
    rng = np.random.default_rng(7)
    first = rng.normal(0, 1, 400)
    second = rng.normal(0, 1, 400)
    noise = rng.normal(0, 1, 400)
    target = np.column_stack(
        (
            first + rng.normal(0, 0.05, 400),
            second + rng.normal(0, 0.05, 400),
        )
    )
    return np.column_stack((first, second, noise)), target


def assert_feature_weights(
    weights: torch.Tensor,
    *,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    """Assert that feature weights are a detached simplex on the requested placement."""
    assert weights.shape == (width,)
    assert weights.dtype is dtype
    assert weights.device == device
    assert not weights.requires_grad
    assert bool(torch.isfinite(weights).all())
    assert bool((weights >= 0).all())
    torch.testing.assert_close(
        weights.sum(),
        torch.ones((), dtype=dtype, device=device),
    )


def reporting_model(
    *,
    weights: tuple[float, ...] = (0.6, 0.05, 0.3, 0.05),
    learn_features: bool = False,
    epsilon_M: float = float("inf"),
    W_M: tuple[float, ...] | None = None,
    categorical_features: int = 1,
):
    """Fit mixed geometry with known, pinned feature relevance and two outputs."""
    from conftest import DEVICE, DTYPE

    from entlearn import Input, Network, Recipe, RegressionHead

    X = torch.tensor([[0.0, 0.1, 0.3], [0.8, 1.0, 0.6]], dtype=DTYPE, device=DEVICE)
    categories = tuple(
        torch.tensor([0, 1], dtype=torch.int64, device=DEVICE) for _ in range(categorical_features)
    )
    y = X[:, :2]
    recipe = Recipe.chain(
        Input(K=1, epsilon_D=0.0 if learn_features else float("inf")),
        RegressionHead(epsilon_M=epsilon_M, W_M=W_M),
    )
    state = Network.initialise(
        recipe,
        X,
        y,
        X_cat=categories,
        feature_weights=torch.tensor(weights, dtype=DTYPE, device=DEVICE),
    )
    return Network.fit(recipe, X, y, X_cat=categories, initial_state=state, max_iter=1)


def reporting_chain(*, task: str, kind: str, coupling, head_coupling):
    """Return an asymmetric chain, bounded mixed data and one reusable initial state."""
    from conftest import DEVICE, DTYPE

    from entlearn import (
        ClassificationHead,
        Hidden,
        Input,
        ManifoldInput,
        Network,
        Recipe,
        RegressionHead,
    )

    rng = torch.Generator(device=DEVICE).manual_seed(37)
    X = torch.rand(12, 3, dtype=DTYPE, device=DEVICE, generator=rng)
    codes = torch.arange(12, dtype=torch.int64, device=DEVICE) % 3
    categories = () if kind == "manifold" else (codes % 2, codes)
    first = (
        ManifoldInput(name="features", K=4, subspace_dimension=2, epsilon=0.4, epsilon_T=0.5)
        if kind == "manifold"
        else Input(name="features", K=4, epsilon=0.4, epsilon_T=0.5, epsilon_D=0.5)
    )
    head = (
        ClassificationHead(name="response", coupling=head_coupling, n_classes=3)
        if task == "classification"
        else RegressionHead(name="response", epsilon_M=0.5)
    )
    recipe = Recipe.chain(
        first,
        Hidden(name="wide", K=3, epsilon=0.4),
        Hidden(name="narrow", K=2, epsilon=0.4),
        head,
        coupling=coupling,
        theta_alpha=2.0,
    )
    y = codes if task == "classification" else X[:, :2].square()
    state = Network.initialise(recipe, X, y, X_cat=categories, seed=7)
    return recipe, X, y, categories, state


def fitted_manifold_report():
    """Fit a two-dimensional tangent geometry for reporting invariants."""
    from entlearn import Coupling, Network

    recipe, X, y, cats, state = reporting_chain(
        task="regression", kind="manifold", coupling=Coupling.M, head_coupling=Coupling.M
    )
    return Network.fit(recipe, X, y, X_cat=cats, initial_state=state, max_iter=2)


def manifold_count_model(d: int, alpha: float, *, prune: bool = False):
    """Fit three-dimensional geometry with two fixed-weight regression outputs."""
    from conftest import DEVICE, DTYPE

    from entlearn import ManifoldInput, Network, Recipe, RegressionHead

    X = torch.tensor(
        [[0.1, 0.2, 0.3], [0.4, 0.7, 0.8], [0.8, 0.5, 0.2], [0.2, 0.9, 0.4]],
        dtype=DTYPE,
        device=DEVICE,
    )
    if prune:
        X = torch.full_like(X, 0.5)
    recipe = Recipe.chain(
        ManifoldInput(K=3 if prune else 1, subspace_dimension=d, alpha=alpha),
        RegressionHead(W_M=(2.0, 1.0)),
    )
    return Network.fit(recipe, X, X[:, :2], seed=3, max_iter=1)
