"""Reductions over the rows: the instance-weight cost, the input loss term and chunked products."""

from __future__ import annotations

import torch

# Rows reduced by one product in a chunked product; the chunk products are then summed pairwise.
_CHUNK_ROWS = 4096
# cuBLAS sums one long product accurately; its batched chunks are faster only in float32.
_CUDA_MIN_CHUNKS = 16


def _chunked_product_(
    out: torch.Tensor, A: torch.Tensor, B: torch.Tensor, scratch: torch.Tensor
) -> None:
    """Write ``out = Aᵀ B`` as products over row chunks summed pairwise.

    It takes one product instead when ``A`` has fewer than two chunks of rows, when the
    contiguous ``scratch`` cannot hold the chunk products, and on CUDA in float64 or below
    ``_CUDA_MIN_CHUNKS`` chunks.
    """
    n = A.shape[0] // _CHUNK_ROWS
    cuda_one = A.is_cuda and (A.dtype == torch.float64 or n < _CUDA_MIN_CHUNKS)
    if n < 2 or cuda_one or n * out.numel() > scratch.numel():
        torch.mm(A.transpose(0, 1), B, out=out)
        return
    rows = n * _CHUNK_ROWS
    partial = scratch.view(-1)[: n * out.numel()].view(n, *out.shape)
    torch.bmm(
        A[:rows].view(n, _CHUNK_ROWS, -1).transpose(1, 2),
        B[:rows].view(n, _CHUNK_ROWS, -1),
        out=partial,
    )
    torch.sum(partial, dim=0, out=out)
    out.addmm_(A[rows:].transpose(0, 1), B[rows:])


def compute_wt_cost_(
    Wt_cost: torch.Tensor,
    gamma: torch.Tensor,
    sqdist_wd: torch.Tensor | None,
    cat_cost: torch.Tensor | None,
    scratch_TK: torch.Tensor,
) -> None:
    """Write the per-instance weight cost into ``Wt_cost``.

    Reduces the two cached costs over the clusters::

        Wt_cost[t] = Σ_k gamma[t,k]·(sqdist_wd[t,k] + cat_cost[t,k])

    Either cost may be ``None`` when that modality is absent, but not both.

    Args:
        Wt_cost: ``(T,)`` output, overwritten.
        gamma: ``(T, K)`` end-of-iteration affiliations Γ.
        sqdist_wd: ``(T, K)`` Wd-weighted Euclidean distances, or ``None`` when
            ``D_cont == 0``.
        cat_cost: ``(T, K)`` summed categorical cost, weighted by ``Wd`` but not by
            ``Wt``, or ``None`` when ``D_cat == 0``.
        scratch_TK: ``(T, K)`` workspace.
    """
    if sqdist_wd is not None and cat_cost is not None:  # continuous and categorical features
        torch.add(sqdist_wd, cat_cost, out=scratch_TK)  # sqdist_wd + cat_cost
        scratch_TK.mul_(gamma)  # gamma ∘ (sqdist_wd + cat_cost)
    elif sqdist_wd is not None:  # continuous only
        torch.mul(gamma, sqdist_wd, out=scratch_TK)  # gamma ∘ sqdist_wd
    else:  # categorical only
        assert cat_cost is not None, "at least one of sqdist_wd / cat_cost is required"
        torch.mul(gamma, cat_cost, out=scratch_TK)  # gamma ∘ cat_cost
    torch.sum(scratch_TK, dim=1, out=Wt_cost)  # Σ_k -> (T,) row reduction


def reduce_input_cost_(
    out_scalar: torch.Tensor,
    disc_cost: torch.Tensor,
    gamma: torch.Tensor,
    scratch_TK: torch.Tensor,
    scratch_scalar: torch.Tensor,
) -> None:
    """Add the input block's discretisation error to the loss in ``out_scalar``.

    Adds ``Σ_{t,k} gamma[t,k]·disc_cost[t,k]``. The caller must zero ``out_scalar``
    first.

    Args:
        out_scalar: ``()`` loss accumulator, added into.
        disc_cost: ``(T, K)`` per-instance input costs.
        gamma: ``(T, K)`` input-block affiliations.
        scratch_TK: ``(T, K)`` workspace.
        scratch_scalar: ``()`` workspace.
    """
    torch.mul(gamma, disc_cost, out=scratch_TK)  # gamma ∘ disc_cost
    torch.sum(scratch_TK, dim=(0, 1), out=scratch_scalar)  # full reduction into the scalar scratch
    out_scalar.add_(scratch_scalar)  # += Σ_{t,k} gamma·disc_cost
