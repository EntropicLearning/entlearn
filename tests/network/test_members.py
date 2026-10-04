"""Original-state and fitted-member retention through the public lifecycle."""

import gc
import weakref

import pytest
import torch
from conftest import DEVICE, DTYPE
from network._fixtures import (
    blobs,
    classification_recipe,
    regression_blobs,
    regression_recipe,
    squared_error,
    validation_pairs,
)

from entlearn import ClassificationHead, Coupling, ManifoldInput, Network, PredictConfig, Recipe


class TestMemberRetention:
    @pytest.mark.parametrize("n_inits", [1, 3])
    @pytest.mark.parametrize("retain", ["winner", "states", "members"])
    def test_retention_matrix(self, n_inits, retain):
        X, y = blobs(2, 0.1)
        recipe = classification_recipe()
        network = Network.fit(
            recipe,
            X,
            y,
            n_inits=n_inits,
            seed=13,
            max_iter=3,
            retain=retain,
        )
        records = network.diagnostics.initialisation_outcomes
        assert len(records) == n_inits
        assert [record.index for record in records] == list(range(n_inits))
        best = min(records, key=lambda record: record.score)
        original = Network.initialise(recipe, X, y, seed=best.seed)
        assert torch.equal(
            network.initial_state.input_geometry.continuous_centroids,
            original.input_geometry.continuous_centroids,
        )
        replay = Network.fit(recipe, X, y, initial_state=network.initial_state, max_iter=3)
        assert torch.equal(replay.predict(X), network.predict(X))
        if n_inits == 1:
            assert records[0].seed == 13
        assert (network.initial_states is not None) == (retain != "winner")
        assert (network.members is not None) == (retain == "members")
        if retain != "winner":
            assert len(network.initial_states) == n_inits
            assert network.initial_state is network.initial_states[best.index]
            for record, state in zip(records, network.initial_states, strict=True):
                expected = Network.initialise(recipe, X, y, seed=record.seed)
                assert torch.equal(
                    state.input_geometry.continuous_centroids,
                    expected.input_geometry.continuous_centroids,
                )
        if retain == "members":
            assert len(network.members) == n_inits
            predictions = network.predict_all(X)
            assert predictions.shape == (n_inits, *network.predict(X).shape)
            for index, member in enumerate(network.members):
                assert member.members is None
                assert member.initial_states is None
                replay = Network.fit(recipe, X, y, initial_state=member.initial_state, max_iter=3)
                assert torch.equal(replay.predict(X), predictions[index])
                assert member.initial_state is network.initial_states[index]
                assert torch.equal(predictions[index], member.predict(X))
            assert network.initial_state is network.members[best.index].initial_state
            assert torch.equal(predictions[best.index], network.predict(X))
        else:
            with pytest.raises(ValueError, match='retain="members"'):
                network.predict_all(X)

    @pytest.mark.parametrize("retain", ["all", "Members", True, None])
    def test_retain_rejects_values_outside_the_three_modes(self, retain):
        X, y = blobs(2, 0.1)
        with pytest.raises(ValueError, match="retain must be 'winner', 'states' or 'members'"):
            Network.fit(classification_recipe(), X, y, retain=retain, max_iter=3)

    def test_later_winner_retains_its_original_state_without_optional_collections(self):
        X, y = blobs(3, 0.1)
        recipe = classification_recipe(K=3)
        scores = iter((2.0, 1.0, 3.0))

        def select_second(prediction, target, **weights):
            return next(scores)

        network = Network.fit(recipe, X, y, n_inits=3, selection_loss=select_second, max_iter=3)
        assert network.initial_states is None
        assert network.members is None
        winner = network.diagnostics.initialisation_outcomes[1]
        expected = Network.initialise(recipe, X, y, seed=winner.seed)
        assert torch.equal(
            network.initial_state.input_geometry.continuous_centroids,
            expected.input_geometry.continuous_centroids,
        )
        replay = Network.fit(recipe, X, y, initial_state=network.initial_state, max_iter=3)
        assert torch.equal(replay.predict(X), network.predict(X))

    @pytest.mark.parametrize("backend", ["threads", "processes"])
    def test_parallel_winner_replays_without_optional_collections(self, backend):
        X, y = blobs(3, 0.1)
        recipe = classification_recipe(K=3)
        network = Network.fit(
            recipe, X, y, n_inits=3, n_jobs=2, parallel_backend=backend, max_iter=3
        )
        assert network.initial_states is None
        assert network.members is None
        assert network.initial_state.input_geometry.continuous_centroids.is_inference()
        replay = Network.fit(recipe, X, y, initial_state=network.initial_state, max_iter=3)
        assert torch.equal(replay.predict(X), network.predict(X))

    def test_original_state_preserves_pre_pruning_geometry(self):
        X, y = blobs(2, 0.1)
        recipe = classification_recipe(K=3)
        state = Network.initialise(
            recipe,
            X,
            y,
            continuous_centroids=torch.zeros(3, X.shape[1], dtype=DTYPE, device=DEVICE),
        )
        network = Network.fit(recipe, X, y, initial_state=state, max_iter=3)
        assert dict(network.schema.K_active)["input"] == 1
        assert network.initial_state.input_geometry.K_active == 3
        assert not network.initial_state.input_geometry.captured
        assert torch.equal(
            network.initial_state.input_geometry.continuous_centroids,
            state.input_geometry.continuous_centroids,
        )
        replay = Network.fit(recipe, X, y, initial_state=network.initial_state, max_iter=3)
        assert torch.equal(replay.predict(X), network.predict(X))
        changed_data = Network.fit(
            recipe, X + 0.2, y, initial_state=network.initial_state, max_iter=3
        )
        assert torch.equal(
            changed_data.initial_state.input_geometry.continuous_centroids,
            state.input_geometry.continuous_centroids,
        )
        assert changed_data.initial_state.input_geometry.K_active == 3
        before = network.predict(X).clone()
        with torch.inference_mode():
            state.input_geometry.continuous_centroids.fill_(10)
            assert (
                torch.count_nonzero(network.initial_state.input_geometry.continuous_centroids) == 0
            )
            network.initial_state.input_geometry.continuous_centroids.fill_(20)
        assert torch.equal(before, network.predict(X))

    def test_single_member_prediction_and_complete_override(self):
        X, y = blobs(2, 0.1)
        network = Network.fit(classification_recipe(), X, y, retain="members", max_iter=3)
        assert torch.equal(network.predict_all(X)[0], network.predict(X))
        config = PredictConfig(output_mode="arithmetic")
        assert torch.equal(
            network.predict_all(X, predict_config=config)[0],
            network.predict(X, predict_config=config),
        )

    @pytest.mark.parametrize(
        "backend,n_jobs", [("threads", None), ("threads", 2), ("processes", 2)]
    )
    @pytest.mark.parametrize("coupling", [Coupling.M, Coupling.S])
    def test_manifold_members_replay_and_reconstruct(self, backend, n_jobs, coupling):
        X, y = blobs(2, 0.1)
        recipe = Recipe.chain(
            ManifoldInput(K=2, subspace_dimension=1, alpha=0.1),
            ClassificationHead(coupling=coupling),
        )
        network = Network.fit(
            recipe,
            X,
            y,
            n_inits=2,
            n_jobs=n_jobs,
            parallel_backend=backend,
            retain="members",
            max_iter=3,
        )
        assert network.members is not None
        query = X[:3]
        predictions = network.predict_all(query)
        for index, member in enumerate(network.members):
            state = member.initial_state
            assert state is network.initial_states[index]
            assert state.input_geometry.feature_weights is None
            assert state.input_geometry.manifold_projectors.is_inference()
            assert member.schema.computation_dtype == DTYPE
            replay = Network.fit(recipe, X, y, initial_state=state, max_iter=3)
            assert torch.equal(predictions[index], replay.predict(query))
            assert torch.equal(
                member.reconstruct(query).continuous, replay.reconstruct(query).continuous
            )
        before = network.predict_all(query).clone()
        with torch.inference_mode():
            network.initial_state.input_geometry.manifold_projectors.zero_()
        assert torch.equal(network.predict_all(query), before)

    @pytest.mark.parametrize("backend", ["threads", "processes"])
    def test_regression_members_imply_all_state_collection(self, backend):
        X, y, _ = regression_blobs(2)
        network = Network.fit(
            regression_recipe(),
            X,
            y,
            n_inits=3,
            n_jobs=2,
            parallel_backend=backend,
            retain="members",
            max_iter=3,
        )
        assert network.initial_states is not None
        predictions = network.predict_all(X[:4])
        assert predictions.shape == (3, 4, 2)
        assert predictions.device == DEVICE
        assert network.members is not None
        for index, member in enumerate(network.members):
            assert member.initial_states is None
            assert network.initial_states[index] is member.initial_state
            assert member.initial_state.input_geometry.continuous_centroids.is_inference()
            assert torch.equal(predictions[index], member.predict(X[:4]))
            replay = Network.fit(
                regression_recipe(), X, y, initial_state=member.initial_state, max_iter=3
            )
            assert torch.equal(predictions[index], replay.predict(X[:4]))

    def test_retained_original_state_does_not_keep_callers_tensor_storage(self):
        X, y = blobs(2, 0.1)
        recipe = classification_recipe()
        state = Network.initialise(recipe, X, y)
        reference = weakref.ref(state.input_geometry.continuous_centroids)
        network = Network.fit(recipe, X, y, initial_state=state, retain="members", max_iter=3)
        del state
        gc.collect()
        assert reference() is None
        assert network.initial_states[0] is network.initial_state
        replay = Network.fit(recipe, X, y, initial_state=network.initial_state, max_iter=3)
        assert torch.equal(replay.predict(X), network.predict(X))
        assert torch.equal(network.predict_all(X)[0], network.predict(X))

    def test_original_states_share_no_storage_across_folds_members_or_callers(self):
        X, y, _ = regression_blobs(2)
        recipe = regression_recipe()
        selected = Network.fit(
            recipe,
            X,
            y,
            n_inits=2,
            max_iter=2,
            validation_pairs=validation_pairs(),
            selection_loss=squared_error,
            retain="members",
        )
        supplied = Network.initialise(recipe, X, y, seed=3)
        refit = Network.fit(recipe, X, y, initial_state=supplied, max_iter=2)
        assert selected.initial_states is not None and selected.members is not None
        states = (*selected.initial_states, refit.initial_state, supplied)
        storages = {
            state.input_geometry.continuous_centroids.untyped_storage().data_ptr()
            for state in states
        }
        assert len(storages) == len(states)
        networks = (*selected.members, refit)
        before = [network.predict(X) for network in networks]
        with torch.inference_mode():
            for state in states:
                state.input_geometry.continuous_centroids.add_(1)
        for network, expected in zip(networks, before, strict=True):
            torch.testing.assert_close(network.predict(X), expected, rtol=0, atol=0)

    def test_categorical_queries_and_prediction_errors_use_member_contract(self):
        X, y = blobs(2, 0.1)
        network = Network.fit(
            classification_recipe(), X, y, X_cat=(y,), retain="members", max_iter=3
        )
        assert torch.equal(
            network.predict_all(X[:3], X_cat=(y[:3],))[0],
            network.predict(X[:3], X_cat=(y[:3],)),
        )
        with pytest.raises(ValueError):
            network.predict_all(X[:3])
        with pytest.raises(ValueError):
            network.predict_all(X[:3], X_cat=(y[:3],), predict_config=object())
