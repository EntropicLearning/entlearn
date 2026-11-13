"""Tests for the feature-modality encoding helpers."""

import pytest
import torch
from conftest import DEVICE

from entlearn.primitives.encoding import densify_categorical, split_wd


class TestDensifyCategorical:
    @pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
    def test_codes_become_one_hot_rows(self, dtype):
        codes = torch.tensor([0, 2, 1], device=DEVICE)
        got = densify_categorical([codes], [4], dtype)
        expected = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            dtype=dtype,
            device=DEVICE,
        )
        assert got[0].dtype == dtype
        assert torch.equal(got[0], expected)

    def test_codes_read_only(self):
        codes = torch.tensor([0, 2, 1], device=DEVICE)
        codes_c = codes.clone()
        densify_categorical([codes], [3], torch.float64)
        assert torch.equal(codes, codes_c)

    def test_distribution_column_passes_through_unchanged(self):
        dist = torch.tensor([[0.2, 0.8], [0.5, 0.5]], dtype=torch.float64, device=DEVICE)
        got = densify_categorical([dist], [2], torch.float64)
        assert got[0] is dist  # passed through, not copied


class TestSplitWd:
    def test_returns_boundary_views_not_copies(self):
        wd = torch.arange(7, dtype=torch.float64, device=DEVICE)
        cont, cat = split_wd(wd, 4)
        assert torch.equal(cont, wd[:4]) and torch.equal(cat, wd[4:])
        assert cont.data_ptr() == wd.data_ptr()  # views share storage
        assert cat.data_ptr() == wd[4:].data_ptr()

    @pytest.mark.parametrize("d_cont", [0, 7])
    def test_degenerate_boundaries(self, d_cont):
        wd = torch.arange(7, dtype=torch.float64, device=DEVICE)
        cont, cat = split_wd(wd, d_cont)
        assert cont.numel() == d_cont and cat.numel() == 7 - d_cont
