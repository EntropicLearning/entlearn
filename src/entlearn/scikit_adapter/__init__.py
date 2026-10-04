"""Optional scikit-learn adapters for tensor-native Networks."""

from entlearn.scikit_adapter.classifier import EONClassifier
from entlearn.scikit_adapter.features import FeatureLayout, TabularReconstruction
from entlearn.scikit_adapter.regressor import EONRegressor
from entlearn.scikit_adapter.selection import common_train_rows

__all__ = [
    "EONClassifier",
    "EONRegressor",
    "FeatureLayout",
    "TabularReconstruction",
    "common_train_rows",
]
