"""Small tabular problems and Recipes for estimator contract tests."""

import numpy as np
import pandas as pd
import polars as pl
import torch
from conftest import DEVICE, DTYPE

from entlearn import (
    ClassificationHead,
    Coupling,
    Hidden,
    Input,
    ManifoldInput,
    Recipe,
    RegressionHead,
)
from entlearn.scikit_adapter import EONClassifier


def protocol_estimator(estimator_type):
    """Build the default-precision estimator used for sklearn compatibility checks."""
    recipe = (
        Recipe.chain(Input(K=3), ClassificationHead(Coupling.M))
        if estimator_type is EONClassifier
        else regression_recipe()
    )
    return estimator_type(recipe, random_state=0, device=DEVICE)


def fitted_protocol_estimator(estimator_type):
    """Fit the compatibility-check configuration on bounded tabular data."""
    X, labels = tabular_data()
    model = protocol_estimator(estimator_type)
    model.fit(X, labels if estimator_type is EONClassifier else X[:, 0])
    return model, X


def classifier_recipe(*, hidden: bool = False, coupling: Coupling = Coupling.M) -> Recipe:
    blocks = [Input(K=3, name="features", epsilon=0.05)]
    if hidden:
        blocks.append(Hidden(K=2, name="hidden_b", epsilon=0.04))
    return Recipe.chain(
        *blocks,
        ClassificationHead(coupling, name="labels"),
        coupling=coupling if hidden else None,
    )


def tabular_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(27)
    centres = np.array([[0.15, 0.2], [0.75, 0.25], [0.4, 0.8]])
    X = np.concatenate([c + rng.uniform(-0.06, 0.06, (16, 2)) for c in centres])
    y = np.repeat(["east", "north", "west"], 16)
    return X, y


def regression_data() -> tuple[np.ndarray, np.ndarray]:
    """Two real outputs over the shared tabular features."""
    X, _ = tabular_data()
    return X, np.column_stack((2 * X[:, 0] - X[:, 1], X[:, 0] + X[:, 1]))


def regression_recipe() -> Recipe:
    return Recipe.chain(Input(K=3, epsilon=0.05), RegressionHead())


def continuation_case(estimator_type, **controls):
    """Build an unfitted estimator and compatible data for lifecycle workflows."""
    X, labels = tabular_data()
    classifier = estimator_type is EONClassifier
    model = estimator_type(
        classifier_recipe() if classifier else regression_recipe(),
        **(dict(dtype=DTYPE, device=DEVICE, random_state=7, max_iter=3, tol=0) | controls),
    )
    return model, X, labels if classifier else X[:, 0] - 2 * X[:, 1]


def continuation_frame(X, *, categories=None):
    """Give continuation features stable names and explicit category values."""
    if categories is None:
        categories = np.tile(["a", "b"], len(X) // 2)
    return pd.DataFrame(X, columns=["x", "z"]).assign(category=categories)


def tensor_data(X: np.ndarray, y: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.as_tensor(X, dtype=DTYPE, device=DEVICE),
        torch.as_tensor(np.unique(y, return_inverse=True)[1], dtype=torch.int64, device=DEVICE),
    )


def mixed_frames():
    """Equivalent eager frames with one categorical and two continuous columns."""
    X, y = tabular_data()
    columns = {"category": y, "northing": X[:, 1], "easting": X[:, 0]}
    return pd.DataFrame(columns), pl.DataFrame(columns), y


def dataframe_classifier() -> EONClassifier:
    """An unfitted classifier with shared dataframe inference and execution controls."""
    return EONClassifier(
        classifier_recipe(),
        categorical_features="from_dtype",
        dtype=DTYPE,
        device=DEVICE,
        random_state=3,
    )


def parameter_variants():
    """Descriptions differing at every tunable field, including linked endpoint names."""
    for input_type in (Input, ManifoldInput):
        for head_type in (ClassificationHead, RegressionHead):
            old_head = (
                ClassificationHead(Coupling.M, name="labels")
                if head_type is ClassificationHead
                else RegressionHead(name="labels")
            )
            new_head = (
                ClassificationHead(Coupling.S, n_classes=4, name="response")
                if head_type is ClassificationHead
                else RegressionHead(W_M=(0.2, 0.8), name="response")
            )
            changes = dict(
                K=5,
                epsilon=0.2,
                epsilon_T=0.3,
                centroid_strategy="greedy-kmeans++",
                greedy_candidates=4,
                balanced=head_type is ClassificationHead,
                name="new_features",
            )
            if input_type is Input:
                changes.update(epsilon_D=0.4, W_std=0.2, delta_cat=0.8)
            else:
                changes.update(subspace_dimension=2, alpha=0.3)
            old = Recipe.chain(
                input_type(name="features"), Hidden(name="hidden_b"), old_head, coupling=Coupling.M
            )
            new = Recipe.chain(
                input_type(**changes),
                Hidden(K=4, epsilon=0.1, name="middle"),
                new_head,
                coupling=Coupling.S,
                delta=0.6,
                theta_alpha=2.0,
            )
            yield old, new
    old = Recipe.chain(Input(), RegressionHead())
    yield old, old.replace_block("output", epsilon_M=0.2)


def fitted_query_model(estimator_type, manifold=False, *, epsilon_T=0.5, sample_weight=False):
    """Fit a small query-capable model with optional non-uniform row weights."""
    X, labels = tabular_data()
    first = (
        ManifoldInput(K=3, subspace_dimension=1, epsilon=0.1, epsilon_T=epsilon_T)
        if manifold
        else Input(K=3, epsilon=0.1, epsilon_T=epsilon_T, epsilon_D=0.2)
    )
    head = ClassificationHead(Coupling.M) if estimator_type is EONClassifier else RegressionHead()
    model = estimator_type(
        Recipe.chain(first, head), dtype=DTYPE, device=DEVICE, random_state=3, max_iter=5
    )
    model.fit(
        X,
        labels if estimator_type is EONClassifier else X[:, 0],
        sample_weight=np.linspace(0.1, 2, len(X)) if sample_weight else None,
    )
    return model, X
