"""Original-space reconstruction from shared prediction affiliations."""

import torch

from entlearn.network.blocks.manifold import _ManifoldInputBlock
from entlearn.network.blocks.types import _InputBlock
from entlearn.network.state import ReconstructionResult
from entlearn.primitives.manifold import project


def reconstruct_from_affiliations(
    block: _InputBlock, X: torch.Tensor, gamma: torch.Tensor
) -> ReconstructionResult:
    """Blend fitted centroids or local projections using already-computed affiliations."""
    if isinstance(block, _ManifoldInputBlock):
        # Manifold input has no categorical centroids to blend.
        projected = project(X, block.continuous_centroids, block.manifold_projectors, gamma)
        return ReconstructionResult(projected.detach(), ())
    return ReconstructionResult(
        (gamma @ block.continuous_centroids).detach(),
        tuple((gamma @ centroid).detach() for centroid in block.categorical_centroids),
    )
