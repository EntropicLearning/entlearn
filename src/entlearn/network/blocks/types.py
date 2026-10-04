"""Closed unions of fitted block variants."""

from entlearn.network.blocks.classification import _ClassificationBlock
from entlearn.network.blocks.hidden import _HiddenBlock
from entlearn.network.blocks.input import _StandardInputBlock
from entlearn.network.blocks.manifold import _ManifoldInputBlock
from entlearn.network.blocks.regression import _RegressionBlock

_InputBlock = _StandardInputBlock | _ManifoldInputBlock
_OutputBlock = _ClassificationBlock | _RegressionBlock
_ClusteringBlock = _InputBlock | _HiddenBlock
_RuntimeBlock = _ClusteringBlock | _OutputBlock
