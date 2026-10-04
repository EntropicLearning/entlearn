"""Immutable prediction policy and task-specific resolution."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from entlearn.recipe import (
    ClassificationHead,
    Coupling,
    RegressionHead,
    _require_nonnegative_finite,
    _require_positive_integer,
)

_DEFAULT_TOL = 1e-4
_DEFAULT_MAX_ITER = 100


@dataclass(frozen=True)
class PredictConfig:
    """Complete prediction policy, never implicitly merged with another policy.

    Attributes:
        predict_mode: ``"single"`` for a single forward pass or ``"iterative"`` for
            iterative prediction and affiliation refinement with frozen parameters.
        output_mode: ``"geometric"`` or (M-only) ``"arithmetic"`` for classification.
            ``None`` selects geometric classification or ordinary regression.
        epsilon_P: Finite non-negative geometric output temperature. Omission at
            fit calibrates it. Omission in a complete prediction override uses
            ``epsilon_P = delta``, the head connection strength, because the
            uncalibrated fallback is ``beta = delta / epsilon_P = 1``.
            Invalid for arithmetic or regression.
        tol: Non-negative iterative tolerance, independent of a fit's controls.
        max_iter: Positive iterative iteration limit, independent of a fit's controls.
    """

    predict_mode: Literal["single", "iterative"] = "single"
    output_mode: Literal["geometric", "arithmetic"] | None = None
    epsilon_P: float | None = None
    tol: float = _DEFAULT_TOL
    max_iter: int = _DEFAULT_MAX_ITER

    def __post_init__(self) -> None:
        """Validate the policy."""
        if self.predict_mode not in ("single", "iterative"):
            raise ValueError("predict_mode must be 'single' or 'iterative'")
        if self.output_mode not in (None, "geometric", "arithmetic"):
            raise ValueError("output_mode must be 'geometric', 'arithmetic' or None")
        _require_nonnegative_finite(self.tol, "tol")
        _require_positive_integer(self.max_iter, "max_iter")
        if self.epsilon_P is not None:
            _require_nonnegative_finite(self.epsilon_P, "epsilon_P")
            if self.output_mode == "arithmetic":
                raise ValueError("arithmetic read-out has no epsilon_P")


def _resolve_predict_config(
    config: PredictConfig | None, head: ClassificationHead | RegressionHead
) -> PredictConfig:
    """Resolve the task default and reject unsupported combinations before staging."""
    if config is None:
        config = PredictConfig()
    if type(config) is not PredictConfig:
        raise ValueError("predict_config must be an exact PredictConfig")
    if isinstance(head, RegressionHead):
        if config.output_mode is not None or config.epsilon_P is not None:
            raise ValueError("regression has no classification read-out or epsilon_P")
        return config
    if config.output_mode == "arithmetic" and head.coupling is Coupling.S:
        raise ValueError("S classification supports geometric read-out only")
    return replace(config, output_mode=config.output_mode or "geometric")
