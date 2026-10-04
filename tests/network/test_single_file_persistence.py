"""A portable Network is one self-contained safetensors file."""

import json
import shutil
from importlib.metadata import version

import pytest
import torch
from network._fixtures import read_saved_metadata, resume_case, write_saved_metadata
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from entlearn import Network


class TestSingleFilePersistence:
    @pytest.mark.parametrize("kind", ["standard", "categorical", "manifold"])
    def test_one_file_can_be_moved_and_loaded_without_its_source_directory(self, tmp_path, kind):
        recipe, X, y, cats = resume_case(kind=kind)
        source = Network.fit(recipe, X, y, X_cat=cats, max_iter=2)
        original = tmp_path / "original"
        original.mkdir()
        path = original / "model.safetensors"
        source.save(path)
        assert list(original.iterdir()) == [path]
        with safe_open(path, framework="pt") as bundle:
            metadata = json.loads(bundle.metadata()["entlearn.network"])
            assert metadata["format"] == "entlearn.network"
            assert metadata["version"] == 1
            assert metadata["package_version"] == version("entlearn")
            names = set(bundle.keys())
        assert "states.0.continuous_centroids" in names
        if kind == "categorical":
            assert "states.0.categorical_centroids.0" in names
        if kind == "manifold":
            assert "states.0.manifold_projectors" in names
        moved = tmp_path / "moved.safetensors"
        shutil.move(path, moved)
        original.rmdir()
        # A stale or unrelated sidecar must have no influence on loading.
        moved.with_suffix(".json").write_text("not model metadata")
        loaded = Network.load(moved, device=X.device)
        torch.testing.assert_close(
            loaded.predict(X, X_cat=cats), source.predict(X, X_cat=cats), rtol=0, atol=0
        )

    def test_producing_package_version_does_not_control_loading(self, tmp_path):
        recipe, X, y, _ = resume_case()
        source = Network.fit(recipe, X, y, max_iter=2)
        path = tmp_path / "model.safetensors"
        source.save(path)
        metadata = read_saved_metadata(path)
        metadata["package_version"] = "999.0.0"
        write_saved_metadata(path, metadata)
        loaded = Network.load(path, device=X.device)
        torch.testing.assert_close(loaded.predict(X), source.predict(X), rtol=0, atol=0)
        loaded.save(path)
        assert read_saved_metadata(path)["package_version"] == version("entlearn")

    @pytest.mark.parametrize("metadata", [None, {}, {"unrelated": "{}"}])
    def test_tensor_only_files_do_not_fall_back_to_a_sidecar(self, tmp_path, metadata):
        recipe, X, y, _ = resume_case()
        path = tmp_path / "model.safetensors"
        Network.fit(recipe, X, y, max_iter=2).save(path)
        with safe_open(path, framework="pt") as bundle:
            # Even valid metadata in the former location cannot repair a tensor-only file.
            path.with_suffix(".json").write_text(bundle.metadata()["entlearn.network"])
        tensors = load_file(path)
        save_file(tensors, path, metadata=metadata)
        with pytest.raises(ValueError, match="metadata"):
            Network.load(path)
