"""Optional Plotly figures of fitted networks.

Install ``entlearn[plotting]``. The small functions take fitted tensors and draw one
panel each. ``plot_decision``, ``plot_manifold`` and ``plot_parallel`` take the network itself.
``Network.plot`` combines them into one figure per block. Importing this namespace
does not load Plotly.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from entlearn.plotting.basic import plot_affiliations as plot_affiliations
    from entlearn.plotting.basic import plot_centroids as plot_centroids
    from entlearn.plotting.basic import plot_feature_importance as plot_feature_importance
    from entlearn.plotting.basic import plot_loss as plot_loss
    from entlearn.plotting.basic import plot_theta as plot_theta
    from entlearn.plotting.decision import plot_decision as plot_decision
    from entlearn.plotting.manifold import plot_manifold as plot_manifold
    from entlearn.plotting.parallel import plot_parallel as plot_parallel

__all__ = [
    "plot_affiliations",
    "plot_centroids",
    "plot_decision",
    "plot_feature_importance",
    "plot_loss",
    "plot_manifold",
    "plot_parallel",
    "plot_theta",
]

_MODULES = {
    "plot_affiliations": "basic",
    "plot_centroids": "basic",
    "plot_decision": "decision",
    "plot_feature_importance": "basic",
    "plot_loss": "basic",
    "plot_manifold": "manifold",
    "plot_parallel": "parallel",
    "plot_theta": "basic",
}


def __getattr__(name: str) -> Any:
    if name not in _MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        module = import_module(f"{__name__}.{_MODULES[name]}")
    except ModuleNotFoundError as error:
        if error.name == "plotly" or (error.name or "").startswith("plotly."):
            raise ModuleNotFoundError(
                "Plotting requires Plotly; install it with `pip install 'entlearn[plotting]'`"
            ) from error
        raise
    value = getattr(module, name)
    globals()[name] = value
    return value
