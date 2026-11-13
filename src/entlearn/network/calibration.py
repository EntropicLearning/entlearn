"""Tensor-native weighted log-loss calibration of geometric classification."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

from entlearn.network.validation import _all_finite
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


def _calibrate_epsilon_P(  # noqa: C901
    logits: torch.Tensor, weighted_target: torch.Tensor, delta: float
) -> float:
    """Select an output temperature by evaluating the read-out's log-loss.

    For single-pass source affiliations ``Gamma_src`` of shape ``(T, K)``
    and head transition matrix ``theta`` of shape ``(M, K)``, the unscaled logits are
    ``Z = Gamma_src @ L.T``,
    where ``L[m, k] = log(max(theta[m, k], tiny))`` is the stored log (:func:`floored_log_`).
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
    if not z.numel() or not _all_finite(z):
        raise RuntimeError("calibration needs finite logits on positive-weight labelled rows")
    components = components / components.sum()
    z = z - z.max(dim=1, keepdim=True).values
    row_mass = components.sum(dim=1)
    target_mean = (components * z).sum(dim=1)
    precision = _eps(z.dtype)
    scratch = torch.empty_like(row_mass[:, None])
    indices = torch.empty(scratch.shape, dtype=torch.int64, device=z.device)
    scaled = torch.empty_like(z)
    probabilities = torch.empty_like(z)
    row_terms = torch.empty_like(row_mass)
    selected = components > 0
    selected_components = components[selected]

    def derivative(log_beta: float) -> float:
        scale_logits_(scaled, z, math.exp(log_beta), 1.0, scratch)
        torch.log_softmax(scaled, dim=1, out=probabilities)
        torch.sum(probabilities.exp_().mul_(z), dim=1, out=row_terms)
        value = float(row_terms.mul_(row_mass).sub_(target_mean).sum())
        if not math.isfinite(value):
            raise RuntimeError("calibration produced a non-finite derivative")
        return value

    def loss(epsilon: float) -> float:
        geometric_probabilities_(probabilities, z, delta, epsilon, scratch, indices)
        selected_probabilities = probabilities[selected]
        if bool((selected_probabilities == 0).any()):
            return math.inf
        value = float(-(selected_components * selected_probabilities.log()).sum())
        if not math.isfinite(value):
            raise RuntimeError("calibration produced a non-finite loss")
        return value

    best_epsilon = delta
    best_loss = loss(best_epsilon)
    candidates = []
    root = _chandrupatla(derivative, math.log(_BETA_MIN), math.log(_BETA_MAX), math.sqrt(precision))
    beta = min(_BETA_MAX, max(_BETA_MIN, math.exp(root)))
    if delta / beta > 0.0:  # an underflowed temperature would repeat the hard read-out
        candidates.append(delta / beta)
    candidates.append(0.0)  # the hard read-out
    for epsilon in candidates:
        if not math.isfinite(epsilon):
            raise RuntimeError("calibration produced a non-finite output temperature")
        candidate_loss = loss(epsilon)
        if candidate_loss < best_loss:
            best_epsilon, best_loss = epsilon, candidate_loss
    if not math.isfinite(best_loss):
        raise RuntimeError("calibration has no finite-loss read-out within the beta bounds")
    return best_epsilon
