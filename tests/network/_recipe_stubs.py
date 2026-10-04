"""Deliberately malformed descriptions, for the Recipe validation tests."""

from dataclasses import dataclass, field

from entlearn import Coupling, Input


class UnreadableData:
    """Raises on any attribute read, to prove validation happens before data access."""

    def __getattribute__(self, name):
        raise AssertionError(f"data was accessed through {name}")


@dataclass
class MutableConnection:
    """A Connection look-alike that is not frozen."""

    name: str
    source: str
    target: str
    delta: float = 1.0
    coupling: Coupling | None = None
    theta_alpha: float | None = None


@dataclass(frozen=True)
class MutableInput(Input):
    """A frozen Input subclass carrying a mutable field."""

    values: list[int] = field(default_factory=list)
