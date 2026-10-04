"""Detached fitted inspection through the public model."""

import pytest
import torch
from conftest import DEVICE, DTYPE
from network._fixtures import blobs, prediction_model


class TestInspection:
    @pytest.mark.parametrize("task", ["classification", "regression"])
    @pytest.mark.parametrize("kind", ["continuous", "mixed", "categorical", "manifold"])
    def test_fitted_values_have_named_owners_and_detached_storage(self, task, kind):
        model, X, categories = prediction_model(task, input_kind=kind)
        before = model.predict(X, X_cat=categories)
        input_name = model.recipe.blocks[0].name
        head_name = model.recipe.blocks[-1].name
        values = model.inspect("continuous_centroids")
        assert list(values) == [input_name]
        assert values[input_name].shape == (dict(model.schema.K_active)[input_name], X.shape[1])
        requests = [
            "continuous_centroids",
            "transition_matrices",
            "head_parameters",
            "training_affiliations",
            "training_instance_weights",
            "manifold_projectors" if kind == "manifold" else "feature_weights",
        ]
        if categories:
            requests.append("categorical_centroids")
            assert len(model.inspect("categorical_centroids")[input_name]) == len(categories)
        head = model.inspect("head_parameters")
        assert list(head) == [head_name]
        assert set(head[head_name]) == ({"theta"} if task == "classification" else {"C_y", "W_M"})
        for name in requests:
            first = model.inspect(name)
            second = model.inspect(name)
            assert first.keys() == second.keys()
            for owner, value in first.items():
                tensors = (
                    value.values()
                    if isinstance(value, dict)
                    else (value if isinstance(value, tuple) else (value,))
                )
                other = second[owner]
                copies = (
                    other.values()
                    if isinstance(other, dict)
                    else (other if isinstance(other, tuple) else (other,))
                )
                for tensor, copy in zip(tensors, copies, strict=True):
                    assert tensor.dtype == DTYPE and tensor.device == DEVICE
                    assert not tensor.requires_grad
                    torch.testing.assert_close(tensor, copy)
                    with torch.inference_mode():
                        tensor.zero_()
        torch.testing.assert_close(model.predict(X, X_cat=categories), before)

    def test_inspected_values_are_inference_tensors_like_every_query(self):
        model, X, categories = prediction_model()
        centroids = model.inspect("continuous_centroids")[model.recipe.blocks[0].name]
        for tensor in (centroids, model.predict(X, X_cat=categories)):
            assert tensor.is_inference()
            with pytest.raises(RuntimeError, match="Inplace update to inference tensor"):
                tensor.zero_()

    @pytest.mark.parametrize("name", ["unknown", "gamma", "log_Z_train", None, []])
    def test_unknown_requests_raise(self, name):
        model, _, _ = prediction_model()
        with pytest.raises(ValueError, match="inspection"):
            model.inspect(name)

    @pytest.mark.parametrize(
        ("kind", "depth", "name"),
        [
            ("continuous", 1, "categorical_centroids"),
            ("continuous", 1, "manifold_projectors"),
            ("manifold", 1, "feature_weights"),
            ("continuous", 0, "transition_matrices"),
        ],
    )
    def test_unsupported_requests_raise(self, kind, depth, name):
        model, _, _ = prediction_model(input_kind=kind, depth=depth)
        with pytest.raises(ValueError, match="unavailable"):
            model.inspect(name)

    def test_inspection_preserves_custom_block_and_connection_names(self):
        from dataclasses import replace

        from entlearn import Network, Recipe

        original, X, categories = prediction_model(input_kind="mixed")
        names = {"input": "measurements", "hidden_1": "clusters", "output": "labels"}
        recipe = Recipe(
            tuple(replace(block, name=names[block.name]) for block in original.recipe.blocks),
            tuple(
                replace(
                    connection,
                    name=f"path_{index}",
                    source=names[connection.source],
                    target=names[connection.target],
                )
                for index, connection in enumerate(original.recipe.connections)
            ),
        )
        labels = torch.arange(X.shape[0], device=DEVICE) % 3
        model = Network.fit(recipe, X, labels, X_cat=categories, max_iter=2)
        assert set(model.inspect("continuous_centroids")) == {"measurements"}
        assert set(model.inspect("head_parameters")) == {"labels"}
        assert set(model.inspect("transition_matrices")) == {"path_0"}
        assert set(model.inspect("training_affiliations")) == {"measurements", "clusters"}

    @pytest.mark.parametrize("saved", [False, True])
    @pytest.mark.parametrize(
        ("input_epsilon", "hidden_epsilon", "expected"),
        [
            (0.05, 0.0, {"input": "soft", "hidden_1": "hard"}),
            (torch.finfo(DTYPE).eps, 0.1, {"input": "hard", "hidden_1": "soft"}),
        ],
    )
    def test_affiliation_regimes_follow_the_fitted_temperature_gate(
        self, tmp_path, saved, input_epsilon, hidden_epsilon, expected
    ):
        from entlearn import ClassificationHead, Coupling, Hidden, Input, Network, Recipe

        X, y = blobs(2, 0.04, n_features=2)
        model = Network.fit(
            Recipe.chain(
                Input(K=2, epsilon=input_epsilon),
                Hidden(K=2, epsilon=hidden_epsilon),
                ClassificationHead(coupling=Coupling.M),
                coupling=Coupling.M,
            ),
            X,
            y,
            max_iter=2,
        )
        if saved:
            model.save(tmp_path / "model.safetensors")
            model = Network.load(tmp_path / "model.safetensors", device=DEVICE)
        assert model.inspect("affiliation_regimes") == expected

    @pytest.mark.parametrize(
        ("epsilon_T", "available"),
        [(0.5, True), (torch.finfo(DTYPE).eps, False), (float("inf"), False)],
    )
    @pytest.mark.parametrize("kind", ["continuous", "mixed", "manifold"])
    def test_scores_available_agrees_with_score_samples(self, epsilon_T, available, kind):
        from entlearn.network.queries import _scores_available

        model, X, categories = prediction_model(input_kind=kind, epsilon_T=epsilon_T)
        assert _scores_available(model) is available
        if available:
            assert model.score_samples(X, X_cat=categories).shape == (X.shape[0],)
        else:
            with pytest.raises(ValueError, match="instance-weight recovery"):
                model.score_samples(X, X_cat=categories)
