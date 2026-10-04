"""Optional state-free helpers."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from entlearn.helpers import reporting
    from entlearn.helpers._feature_weights import feature_weights

__all__ = [
    "feature_weights",
    "reporting",
]


def __getattr__(name: str) -> object:
    """Load statistical helpers only when explicitly requested."""
    if name == "feature_weights":
        from entlearn.helpers import _feature_weights

        return _feature_weights.feature_weights
    if name == "reporting":
        return import_module("entlearn.helpers.reporting")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
