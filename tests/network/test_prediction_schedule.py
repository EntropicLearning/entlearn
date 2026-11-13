"""Independent equations pin the ordered prediction coordinates and loss."""

from dataclasses import replace

import pytest
import torch
from conftest import DEVICE, DTYPE, FIXED_BUDGET
from network._fixtures import prediction_model, skip_fit_iterations

from entlearn import ClassificationHead, Coupling, Input, Network, PredictConfig, Recipe

pytestmark = FIXED_BUDGET


class TestPredictionSchedule:
    @pytest.mark.parametrize("task", ["classification", "regression"])
    @pytest.mark.parametrize("depth", [0, 2])
    @pytest.mark.parametrize("seeded", [False, True])
    def test_two_cycles_match_explicit_head_to_input_equations(self, task, depth, seeded):
        model, train, _ = prediction_model(task, depth=depth, coupling=Coupling.S)
        X = train[:7]
        config = replace(model.predict_config, predict_mode="iterative", max_iter=2, tol=0)
        initial = model.predict_with_details(X, details=("affiliations", "instance_weights"))
        graph = model._graph
        inp, head = graph.input, graph.head
        gammas = {name: value.clone() for name, value in initial.affiliations.items()}
        prediction = initial.prediction.clone()
        start = prediction.flip(0).clone() if seeded else None
        if start is not None:
            prediction = start.clone()
        weights = initial.instance_weights.clone()
        distances = ((X[:, None, :] - inp.continuous_centroids).square() * inp.feature_weights).sum(
            2
        )
        eps = torch.finfo(X.dtype).eps

        def entropy(value):
            return -torch.special.entr(value).sum()

        def outgoing(name):
            (connection,) = graph.outgoing[name]
            if connection.target == graph.order[-1]:
                if task == "classification":
                    return -connection.delta * (prediction @ head.log_theta)
                residual = (prediction[:, None, :] - head.C_y.T).square()
                return connection.delta * (residual * head.W_M).sum(2)
            target = graph.blocks[connection.target]
            return -connection.delta * (
                gammas[connection.target] @ target.incoming[connection.name].log_theta
            )

        losses = []
        for iteration in range(2):
            if iteration or not seeded:
                source = gammas[graph.terminal.source]
                prediction = (
                    torch.softmax(
                        source @ head.log_theta.T * graph.terminal.delta / config.epsilon_P, 1
                    )
                    if task == "classification"
                    else source @ head.C_y.T
                )
            for name in reversed(graph.order[1:-1]):
                block = graph.blocks[name]
                (connection,) = graph.incoming[name]
                cost = outgoing(name) - connection.delta * (
                    gammas[connection.source] @ block.incoming[connection.name].log_theta.T
                )
                gammas[name] = torch.softmax(-cost / block.description.epsilon, 1)
            cost = distances * (weights * train.shape[0])[:, None] + outgoing(graph.order[0])
            gammas[graph.order[0]] = torch.softmax(-cost / inp.description.epsilon, 1)
            per_row = (distances * gammas[graph.order[0]]).sum(1)
            log_partition = inp.log_partition - inp.partition_shift / inp.description.epsilon_T
            weights = torch.exp(-per_row / inp.description.epsilon_T - log_partition).clamp(0, 1)
            loss = train.shape[0] * (weights * per_row).sum()
            loss += (
                train.shape[0]
                * inp.description.epsilon_T
                * (entropy(weights) + weights.sum() * (log_partition - 1))
            )
            for name in graph.order[:-1]:
                loss += graph.blocks[name].description.epsilon * entropy(gammas[name])
            loss += (gammas[graph.terminal.source] * outgoing(graph.terminal.source)).sum()
            if task == "classification":
                loss += config.epsilon_P * entropy(prediction)
            for connection in graph.connections[:-1]:
                target = graph.blocks[connection.target]
                loss -= (
                    connection.delta
                    * (
                        (gammas[connection.target] @ target.incoming[connection.name].log_theta)
                        * gammas[connection.source]
                    ).sum()
                )
            losses.append(float(loss / X.shape[0]))
        actual = model.predict_with_details(
            X,
            predict_config=config,
            predict_init=start,
            details=("affiliations", "instance_weights", "diagnostics"),
        )
        for observed, expected in [
            (actual.prediction, prediction),
            (actual.instance_weights, weights),
            *[(actual.affiliations[name], gamma) for name, gamma in gammas.items()],
            (
                torch.tensor(actual.loss_history, dtype=X.dtype, device=X.device),
                torch.tensor(losses, dtype=X.dtype, device=X.device),
            ),
        ]:
            # The oracle reorders the row and cluster sums, and each softmax at temperature 0.1
            # scales a cost's rounding by up to ten, over two cycles; 256 eps covers that with
            # margin (the measured difference is under 2 eps in either dtype).
            torch.testing.assert_close(observed, expected, atol=256 * eps, rtol=256 * eps)


class TestRetainedEntropy:
    @pytest.mark.parametrize("scale", [0.5, 1.5])
    def test_input_regime_and_loss_do_not_use_query_count_to_gate_entropy(self, monkeypatch, scale):
        epsilon = scale * torch.finfo(DTYPE).eps
        X = torch.tensor([[0.0], [epsilon**0.5]], dtype=DTYPE, device=DEVICE)
        y = torch.tensor([0, 1], device=DEVICE)
        recipe = Recipe.chain(
            Input(K=2, epsilon=epsilon),
            ClassificationHead(coupling=Coupling.M),
            delta=epsilon,
        )
        config = PredictConfig(epsilon_P=0)
        captured = Network.fit(
            recipe, X, y, max_iter=1, predict_config=config
        ).capture_current_state()
        # A neutral head isolates the input entropy coefficient from supervision.
        (head,) = captured.parameters
        state = replace(
            captured,
            input_geometry=replace(captured.input_geometry, continuous_centroids=X.clone()),
            parameters=(replace(head, theta=torch.full_like(head.theta, 0.5)),),
        )
        skip_fit_iterations(monkeypatch)
        model = Network.fit(recipe, X, y, initial_state=state, max_iter=1, predict_config=config)
        policy = PredictConfig(predict_mode="iterative", epsilon_P=0, max_iter=2, tol=0)
        query = X.mean(0, keepdim=True)
        alone = model.predict_with_details(
            query, predict_config=policy, details=("affiliations", "diagnostics")
        )
        batch = model.predict_with_details(
            query.repeat(16, 1), predict_config=policy, details=("affiliations", "diagnostics")
        )
        gamma = alone.affiliations["input"]
        assert torch.count_nonzero(gamma) == 2
        torch.testing.assert_close(batch.affiliations["input"], gamma.expand(16, -1))
        expected = epsilon / 4 + epsilon * torch.log(X.new_tensor(2.0))
        expected -= epsilon * torch.log(X.new_tensor(2.0))
        for result in (alone, batch):
            torch.testing.assert_close(
                X.new_tensor(result.loss_history),
                expected.expand(2),
                atol=epsilon * torch.finfo(DTYPE).eps * 64,
                rtol=torch.finfo(DTYPE).eps * 64,
            )
