"""Tensor-native weighted log-loss calibration of geometric classification."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

from entlearn.primitives.normalise import _eps
from entlearn.primitives.output import geometric_probabilities_, scale_logits_

# Search six decades either side of the uncalibrated inverse temperature beta = delta / epsilon_P = 1.
_BETA_MIN = 1e-6
_BETA_MAX = 1e6
# Safety cap, separate from the dtype-derived convergence tolerance in log beta.
_MAX_ROOT_ITER = 128


def _chandrupatla(
    derivative: Callable[[float], float], lower: float, upper: float, tolerance: float
) -> float:
    """Find the derivative's zero, with all coordinates and tolerances in log beta.

    Chandrupatla's inverse-quadratic interpolation uses bisection whenever its
    admissibility test fails. Endpoints cover objectives whose minimum is at a bound.
    """
    x1, x2 = lower, upper
    f1, f2 = derivative(x1), derivative(x2)
    if f1 >= 0:
        return x1
    if f2 <= 0:
        return x2
    fraction = 0.5
    for _ in range(_MAX_ROOT_ITER):
        x = x1 + fraction * (x2 - x1)
        f = derivative(x)
        if f == 0:
            return x
        # x1 is the latest point; x2 retains the opposite sign and x3 the discarded point.
        x3, f3 = x2, f2
        if (f > 0) == (f1 > 0):
            x3, f3 = x1, f1
        else:
            x2, f2 = x1, f1
        x1, f1 = x, f
        width = abs(x2 - x1)
        if width <= tolerance:
            return x1 if abs(f1) < abs(f2) else x2
        fraction = 0.5
        if x3 != x2 and f3 != f2:
            xi = (x1 - x2) / (x3 - x2)
            phi = (f1 - f2) / (f3 - f2)
            if 0 < xi < 1 and 1 - math.sqrt(1 - xi) < phi < math.sqrt(xi):
                alpha = (x3 - x1) / (x2 - x1)
                fraction = f1 / (f1 - f2) * f3 / (f3 - f2) - alpha * f1 / (f3 - f1) * f2 / (f2 - f3)
        margin = tolerance / (2 * width)
        fraction = min(1 - margin, max(margin, fraction))
    raise RuntimeError("calibration did not converge in log beta")


def _calibrate_epsilon_P(
    logits: torch.Tensor, weighted_target: torch.Tensor, delta: float
) -> float:
    """Select an output temperature by evaluating the read-out's log-loss.

    For single-pass source affiliations ``Gamma_src`` of shape ``(T, K)``
    and head transition matrix ``theta`` of shape ``(M, K)``, the unscaled logits are
    ``Z = Gamma_src @ L.T``,
    where ``L[m, k] = log(max(theta[m, k], _eps(dtype)))`` is the stored safe log.
    The read-out is ``P[t, :] = softmax(beta * Z[t, :])`` with ``beta = delta / epsilon_P``.

    ``weighted_target`` carries sample and class weights. Zero-mass rows are excluded
    from the loss.

    The uncalibrated point wins ties. A hard read-out with any positive target mass
    outside its chosen class has infinite cross-entropy.
    Soft candidates that underflow a positive-weight target probability to zero are also rejected.
    """
    positive = weighted_target.sum(dim=1) > 0
    z = logits[positive]
    components = weighted_target[positive]
    if not z.numel() or not bool(torch.isfinite(z).all()):
        raise RuntimeError("calibration needs finite logits on positive-weight labelled rows")
    components = components / components.sum()
    z = z - z.max(dim=1, keepdim=True).values
    row_mass = components.sum(dim=1)
    target_mean = (components * z).sum(dim=1)
    floor = _eps(z.dtype)
    scratch = torch.empty_like(row_mass[:, None])
    indices = torch.empty(scratch.shape, dtype=torch.int64, device=z.device)
    scaled = torch.empty_like(z)
    probabilities = torch.empty_like(z)

    def log_probabilities(beta: float) -> torch.Tensor:
        scale_logits_(scaled, z, beta, scratch)
        value = torch.log_softmax(scaled, dim=1)
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError("calibration produced non-finite log probabilities")
        return value

    def derivative(log_beta: float) -> float:
        probabilities = log_probabilities(math.exp(log_beta)).exp()
        value = float(((probabilities * z).sum(dim=1) * row_mass - target_mean).sum())
        if not math.isfinite(value):
            raise RuntimeError("calibration produced a non-finite derivative")
        return value

    def loss(epsilon: float) -> float:
        geometric_probabilities_(probabilities, z, delta, epsilon, scratch, indices)
        selected = components > 0
        if bool((probabilities[selected] == 0).any()):
            return math.inf
        value = float(-(components[selected] * probabilities[selected].log()).sum())
        if not math.isfinite(value):
            raise RuntimeError("calibration produced a non-finite loss")
        return value

    best_epsilon = delta
    best_loss = loss(best_epsilon)
    candidates = []
    soft_floor = math.nextafter(floor, math.inf)
    soft_upper = min(_BETA_MAX, delta / soft_floor)
    if soft_upper >= _BETA_MIN:
        root = _chandrupatla(
            derivative, math.log(_BETA_MIN), math.log(soft_upper), math.sqrt(floor)
        )
        beta = min(soft_upper, max(_BETA_MIN, math.exp(root)))
        candidates.append(max(soft_floor, delta / beta))
    hard_epsilon = delta / _BETA_MAX
    if hard_epsilon <= floor:
        candidates.append(hard_epsilon)
    for epsilon in candidates:
        if not math.isfinite(epsilon):
            raise RuntimeError("calibration produced a non-finite output temperature")
        candidate_loss = loss(epsilon)
        if candidate_loss < best_loss:
            best_epsilon, best_loss = epsilon, candidate_loss
    if not math.isfinite(best_loss):
        raise RuntimeError("calibration has no finite-loss read-out within the beta bounds")
    return best_epsilon
