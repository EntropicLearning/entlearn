"""Private staged tensor data."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import assert_never

import torch

from entlearn.network.build import _BuiltRecipe
from entlearn.network.state import DataSchema
from entlearn.network.validation import (
    _normalise_weights,
    _stage_classification_target,
    _stage_regression_target,
    _stage_sample_weights,
    _stage_weights,
    _validate_categorical,
    _validate_continuous,
)
from entlearn.recipe import ClassificationHead, ManifoldInput, RegressionHead


@dataclass(frozen=True)
class _ClassificationSupervision:
    """Target probabilities carrying the normalised sample and class coefficients."""

    weighted_target: torch.Tensor


@dataclass(frozen=True)
class _RegressionSupervision:
    """Labelled sample-task row weights, their weighted target mean, and fixed output weights."""

    row_weights: torch.Tensor
    target_mean: torch.Tensor
    output_weights: torch.Tensor | None


@dataclass(frozen=True)
class _StagedData:
    """Validated features, raw targets, task-specific supervision and inferred schema."""

    X_cont: torch.Tensor
    X_cat: tuple[torch.Tensor, ...]
    target: torch.Tensor
    labelled: torch.Tensor
    sample_weights: torch.Tensor
    raw_sample_weights: torch.Tensor | None
    class_weights: torch.Tensor | None
    task_weights: torch.Tensor | None
    supervision: _ClassificationSupervision | _RegressionSupervision
    schema: DataSchema


def _training_subset(recipe: _BuiltRecipe, data: _StagedData, rows: torch.Tensor) -> _StagedData:
    """Prepare fit rows with their own weight normalisation and the full feature schema."""
    target = data.target[rows]
    if data.schema.task == "regression":
        target = torch.where(data.labelled[rows, None], target, torch.nan)
    return _stage_data(
        recipe,
        data.X_cont[rows],
        target,
        X_cat=tuple(feature[rows] for feature in data.X_cat),
        sample_weights=None if data.raw_sample_weights is None else data.raw_sample_weights[rows],
        class_weights=data.class_weights,
        task_weights=None if data.task_weights is None else data.task_weights[rows],
        computation_dtype=None,
        categorical_cardinalities=data.schema.M_cat,
    )


def _labelled_share(sample_weights: torch.Tensor, labelled: torch.Tensor) -> torch.Tensor:
    """Return the labelled rows' share of the sample weight, exactly one when all are labelled."""
    labelled_weight = sample_weights[labelled].sum()
    return labelled_weight / (labelled_weight + sample_weights[~labelled].sum())


def _classification_weights(
    target: torch.Tensor,
    labelled: torch.Tensor,
    sample_weights: torch.Tensor,
    raw_sample_weights: torch.Tensor,
    class_weights: torch.Tensor,
    labelled_share: torch.Tensor,
) -> torch.Tensor:
    """Return classification target-component weights that sum to the labelled share.

    Class weights average one over the labelled rows. A log-domain path keeps extreme products.
    """
    raw_sample_grid = raw_sample_weights[:, None].expand_as(target)
    class_grid = class_weights[None, :].expand_as(target)
    support = (target > 0) & (class_grid > 0)
    if not bool(support.any()):
        raise ValueError("classification requires positive-weight labelled mass")

    weighted_target = sample_weights[:, None] * target * class_grid
    labelled_mass = weighted_target.sum()
    if (
        bool(torch.isfinite(labelled_mass))
        and bool(labelled_mass > 0)
        and bool(torch.isfinite(weighted_target).all())
        and bool((weighted_target[support] > 0).all())
    ):
        return weighted_target / labelled_mass * labelled_share

    # Add logs, then subtract their maximum, to keep extreme positive products in range.
    component_log_weights = (
        raw_sample_grid[support].log() + target[support].log() + class_grid[support].log()
    )
    weighted_target = torch.zeros_like(target)
    weighted_target[support] = (component_log_weights - component_log_weights.max()).exp()
    return weighted_target / weighted_target.sum() * labelled_share


def _regression_weights(
    labelled: torch.Tensor,
    sample_weights: torch.Tensor,
    raw_sample_weights: torch.Tensor,
    task_weights: torch.Tensor,
    labelled_share: torch.Tensor,
) -> torch.Tensor:
    """Return regression row weights that sum to the labelled share.

    Task weights average one over the labelled rows. A log-domain path keeps extreme products.
    """
    support = labelled & (task_weights > 0)
    if not bool(support.any()):
        raise ValueError("regression requires positive-weight labelled mass")
    weights = torch.where(
        labelled,
        sample_weights * task_weights,
        torch.zeros_like(sample_weights),
    )
    if bool(torch.isfinite(weights).all()) and bool((weights[support] > 0).all()):
        return _normalise_weights(weights, name="regression labelled weights") * labelled_share

    # Add logs before the max shift so opposing scales do not underflow during multiplication.
    log_weights = raw_sample_weights[support].log() + task_weights[support].log()
    weights = torch.zeros_like(raw_sample_weights)
    weights[support] = (log_weights - log_weights.max()).exp()
    return _normalise_weights(weights, name="regression labelled weights") * labelled_share


def _materialise_callable_task_weights(
    weighting: Callable[[torch.Tensor], torch.Tensor],
    target: torch.Tensor,
    labelled: torch.Tensor,
) -> torch.Tensor:
    """Invoke a regression weighting once and map its result onto all rows.

    ``target`` retains the rank supplied through the public API: labelled values are
    one-dimensional for a one-dimensional target and ``(T_labelled, M)`` otherwise.
    """
    labelled_target = target[labelled]
    resolved = weighting(labelled_target)
    n_labelled = labelled_target.shape[0]
    if (
        not isinstance(resolved, torch.Tensor)
        or resolved.ndim != 1
        or resolved.shape[0] != n_labelled
    ):
        raise ValueError(
            f"callable task_weights must return one weight for each of {n_labelled} labelled rows"
        )
    staged = _stage_weights(
        resolved,
        name="callable task_weights",
        length=n_labelled,
        dtype=target.dtype,
        device=target.device,
    )
    if bool((staged <= 0).any()):
        raise ValueError("callable task_weights must return strictly positive values")
    task_weights = torch.ones(target.shape[0], dtype=target.dtype, device=target.device)
    task_weights[labelled] = staged
    return task_weights


def _check_leading_shapes(
    recipe: _BuiltRecipe, X_cont: object, target: object
) -> tuple[torch.Tensor, torch.Tensor]:
    """Check the feature and target tensors' ranks, row counts and fixed ``W_M`` length."""
    if not isinstance(X_cont, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise ValueError("X_cont and y must be tensors")
    if target.ndim == 0:
        raise ValueError("y must be at least one-dimensional")
    if X_cont.ndim == 2 and target.shape[0] != X_cont.shape[0]:
        raise ValueError("X_cont and y must have the same row count")
    if (
        isinstance(recipe.head, RegressionHead)
        and target.ndim in (1, 2)
        and recipe.head.W_M is not None
    ):
        output_width = 1 if target.ndim == 1 else target.shape[1]
        if len(recipe.head.W_M) != output_width:
            raise ValueError(
                f"fixed W_M has length {len(recipe.head.W_M)} "
                f"but the output dimension is {output_width}"
            )
    return X_cont, target


def _stage_features(
    recipe: _BuiltRecipe,
    X_cont: torch.Tensor,
    X_cat: Sequence[torch.Tensor] | None,
    computation_dtype: torch.dtype | None,
    categorical_cardinalities: tuple[int, ...] | None,
    fitted_schema: DataSchema | None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[int, ...]]:
    """Return the checked continuous and categorical features and the categorical cardinalities."""
    continuous = _validate_continuous(X_cont, computation_dtype)
    if fitted_schema is not None:
        if continuous.shape[1] != fitted_schema.D_cont:
            raise ValueError("continuous feature count does not match the fitted schema")
        categorical_cardinalities = fitted_schema.M_cat
    categorical, cardinalities = _validate_categorical(
        X_cat,
        rows=continuous.shape[0],
        dtype=continuous.dtype,
        device=continuous.device,
        allow_conversion=computation_dtype is not None,
        expected=categorical_cardinalities,
    )
    if isinstance(recipe.input, ManifoldInput):
        if categorical:
            raise ValueError("ManifoldInput is continuous-only")
        if recipe.input.subspace_dimension > continuous.shape[1]:
            raise ValueError("subspace_dimension exceeds the continuous feature count")
    if continuous.shape[1] == 0 and not categorical:
        raise ValueError("at least one continuous or categorical input feature is required")
    return continuous, categorical, cardinalities


def _stage_classification(
    target: torch.Tensor,
    continuous: torch.Tensor,
    staged_sample_weights: torch.Tensor,
    raw_sample_weights: torch.Tensor,
    class_weights: torch.Tensor | None,
    task_weights: object,
    n_classes: int | None,
) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor, _ClassificationSupervision]:
    """Return the class target, its labelled mask, class count, class weights and supervision."""
    if task_weights is not None:
        raise ValueError("classification does not accept task_weights")
    staged_target, labelled, M = _stage_classification_target(
        target,
        rows=continuous.shape[0],
        dtype=continuous.dtype,
        device=continuous.device,
        n_classes=n_classes,
    )
    if class_weights is None:
        staged_class_weights = torch.ones(M, dtype=continuous.dtype, device=continuous.device)
    else:
        staged_class_weights = _stage_weights(
            class_weights,
            name="class_weights",
            length=M,
            dtype=continuous.dtype,
            device=continuous.device,
        )
    weighted_target = _classification_weights(
        staged_target,
        labelled,
        staged_sample_weights,
        raw_sample_weights,
        staged_class_weights,
        _labelled_share(staged_sample_weights, labelled),
    )
    return (
        staged_target,
        labelled,
        M,
        staged_class_weights,
        _ClassificationSupervision(weighted_target),
    )


def _stage_regression(
    head: RegressionHead,
    target: torch.Tensor,
    continuous: torch.Tensor,
    staged_sample_weights: torch.Tensor,
    raw_sample_weights: torch.Tensor,
    class_weights: torch.Tensor | None,
    task_weights: torch.Tensor | Callable[[torch.Tensor], torch.Tensor] | None,
) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor, _RegressionSupervision]:
    """Return the regression target, its labelled mask, width, task weights and supervision."""
    if class_weights is not None:
        raise ValueError("regression does not accept class_weights")
    staged_target, labelled, M = _stage_regression_target(
        target,
        rows=continuous.shape[0],
        dtype=continuous.dtype,
        device=continuous.device,
    )
    if task_weights is None:
        staged_task_weights = torch.ones(
            continuous.shape[0],
            dtype=continuous.dtype,
            device=continuous.device,
        )
    elif isinstance(task_weights, torch.Tensor):
        staged_task_weights = _stage_weights(
            task_weights,
            name="task_weights",
            length=continuous.shape[0],
            dtype=continuous.dtype,
            device=continuous.device,
        )
    elif callable(task_weights):
        callable_target = staged_target[:, 0] if target.ndim == 1 else staged_target
        staged_task_weights = _materialise_callable_task_weights(
            task_weights,
            callable_target,
            labelled,
        )
    else:
        raise ValueError("task_weights must be a tensor or callable")
    labelled_share = _labelled_share(staged_sample_weights, labelled)
    row_weights = _regression_weights(
        labelled,
        staged_sample_weights,
        raw_sample_weights,
        staged_task_weights,
        labelled_share,
    )
    output_weights = _fixed_output_weights(head, continuous)
    target_mean = torch.mv(staged_target.transpose(0, 1), row_weights) / labelled_share
    return (
        staged_target,
        labelled,
        M,
        staged_task_weights,
        _RegressionSupervision(row_weights, target_mean, output_weights),
    )


def _fixed_output_weights(head: RegressionHead, continuous: torch.Tensor) -> torch.Tensor | None:
    """Return the head's fixed ``W_M`` normalised in the computation dtype, or ``None``."""
    if head.W_M is None:
        return None
    output_weights = torch.tensor(head.W_M, dtype=continuous.dtype, device=continuous.device)
    if not bool(torch.isfinite(output_weights).all()) or bool((output_weights <= 0).any()):
        raise ValueError("fixed W_M must be finite and positive in the computation dtype")
    return _normalise_weights(output_weights, name="fixed W_M")


def _stage_data(
    recipe: _BuiltRecipe,
    X_cont: object,
    target: object,
    *,
    X_cat: Sequence[torch.Tensor] | None,
    sample_weights: torch.Tensor | None,
    class_weights: torch.Tensor | None,
    task_weights: torch.Tensor | Callable[[torch.Tensor], torch.Tensor] | None,
    computation_dtype: torch.dtype | None,
    categorical_cardinalities: tuple[int, ...] | None = None,
    fitted_schema: DataSchema | None = None,
    known_n_classes: int | None = None,
) -> _StagedData:
    """Stage shared features and weights, then the task target and inferred schema."""
    X_cont, target = _check_leading_shapes(recipe, X_cont, target)
    continuous, categorical, cardinalities = _stage_features(
        recipe, X_cont, X_cat, computation_dtype, categorical_cardinalities, fitted_schema
    )
    # Initialisation uses normalised weights. Raw weights preserve extreme supervised products.
    if sample_weights is None:
        raw_sample_weights = torch.ones(
            continuous.shape[0],
            dtype=continuous.dtype,
            device=continuous.device,
        )
        staged_sample_weights = torch.full(
            (continuous.shape[0],),
            1 / continuous.shape[0],
            dtype=continuous.dtype,
            device=continuous.device,
        )
    else:
        raw_sample_weights, staged_sample_weights = _stage_sample_weights(
            sample_weights,
            length=continuous.shape[0],
            dtype=continuous.dtype,
            device=continuous.device,
        )

    head = recipe.head
    supervision: _ClassificationSupervision | _RegressionSupervision
    if isinstance(head, ClassificationHead):
        n_classes = (
            (head.n_classes or known_n_classes) if fitted_schema is None else fitted_schema.M
        )
        staged_target, labelled, M, staged_class_weights, supervision = _stage_classification(
            target,
            continuous,
            staged_sample_weights,
            raw_sample_weights,
            class_weights,
            task_weights,
            n_classes,
        )
        staged_task_weights = None
    elif isinstance(head, RegressionHead):
        staged_target, labelled, M, staged_task_weights, supervision = _stage_regression(
            head,
            target,
            continuous,
            staged_sample_weights,
            raw_sample_weights,
            class_weights,
            task_weights,
        )
        staged_class_weights = None
    else:
        assert_never(head)

    if fitted_schema is not None and M != fitted_schema.M:
        quantity = "classes" if recipe.task == "classification" else "regression targets"
        raise ValueError(f"number of {quantity} does not match the fitted schema")
    return _StagedData(
        X_cont=continuous,
        X_cat=categorical,
        target=staged_target,
        labelled=labelled,
        sample_weights=staged_sample_weights,
        raw_sample_weights=raw_sample_weights if sample_weights is not None else None,
        class_weights=staged_class_weights if class_weights is not None else None,
        task_weights=staged_task_weights if task_weights is not None else None,
        supervision=supervision,
        schema=DataSchema(
            task=recipe.task,
            D_cont=continuous.shape[1],
            M_cat=cardinalities,
            M=M,
            K_active=(),
            computation_dtype=continuous.dtype,
        ),
    )
