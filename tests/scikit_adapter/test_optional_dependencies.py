"""The adapter validates object values without requiring a dataframe backend."""

from conftest import DEVICE, DTYPE
from helpers._fixtures import run_python_probe


class TestOptionalDependencies:
    def test_missing_object_categories_are_rejected_without_pandas(self):
        result = run_python_probe(
            f"""
import sys
import importlib.abc

class NoPandas(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pandas" or fullname.startswith("pandas."):
            raise ModuleNotFoundError("pandas is not installed")

sys.meta_path.insert(0, NoPandas())
import json
import numpy as np
import torch
from entlearn import ClassificationHead, Coupling, Input, Recipe
from entlearn.scikit_adapter import EONClassifier

recipe = Recipe.chain(Input(K=2), ClassificationHead(Coupling.M))
y = np.repeat(["low", "high"], 6)
results = []
for scalar in (np.datetime64, np.timedelta64):
    X = np.empty((len(y), 2), dtype=object)
    X[:, 0] = np.linspace(0, 1, len(y))
    X[:, 1] = [scalar(int(code), "D") for code in np.repeat([1, 2], 6)]
    estimator = EONClassifier(
        recipe, categorical_features=[1], random_state=3,
        dtype={DTYPE}, device={str(DEVICE)!r},
    ).fit(X, y)
    previous = estimator.network_
    invalid = X.copy()
    invalid[0, 1] = scalar("NaT")
    for operation in (estimator.fit, estimator.predict):
        try:
            operation(invalid, y) if operation == estimator.fit else operation(invalid)
        except ValueError as exc:
            results.append("missing" in str(exc))
        else:
            results.append(False)
    results.append(estimator.network_ is previous)
print(json.dumps(results))
"""
        )
        assert result == [True] * 6
