"""Warning emission attributed to the caller's own line."""

import sys
import warnings
from pathlib import Path


def _caller_prefixes() -> tuple[str, ...]:
    """Return this package's directory and, when loaded, torch's, for ``skip_file_prefixes``."""
    prefixes = (str(Path(__file__).parent),)
    torch_module = sys.modules.get("torch")
    if torch_module is None or torch_module.__file__ is None:
        return prefixes
    return (*prefixes, str(Path(torch_module.__file__).parent))


def _warn(message: str, category: type[Warning]) -> None:
    """Emit a warning on the first calling line outside this package and torch."""
    warnings.warn(message, category, stacklevel=2, skip_file_prefixes=_caller_prefixes())
