"""Shared dependency-availability setup across public API test suites."""

import builtins

import pytest


@pytest.fixture
def without_joblib(monkeypatch):
    """Reject joblib imports while allowing every other dependency."""
    original = builtins.__import__

    def unavailable(name, *args, **kwargs):
        if name.partition(".")[0] == "joblib":
            raise ImportError("joblib is unavailable")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", unavailable)
