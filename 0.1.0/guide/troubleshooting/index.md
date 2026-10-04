# Troubleshooting

This page explains how to follow a fit while it runs, what the warnings emitted during fitting and prediction mean, and where to look first when a result is unexpected.

## Where to look first

Check the state recorded during fitting. For a `Network`, that is `network.recipe`, `network.schema`, `network.diagnostics`, `network.predict_config` and `network.can_resume`. The diagnostics hold `loss_history`, `n_iter`, `converged`, retained `warnings` and, with several initialisations, `initialisation_outcomes`. For an estimator, the equivalent state is its constructor parameters, `feature_layout_`, `loss_curve_`, `n_iter_` and the underlying `network_`.

## Logging

Logging is controlled through the `verbose` argument of `Network.fit`, `resume` and `fine_tune`:

| `verbose`     | Logs                                                                                    |
| ------------- | --------------------------------------------------------------------------------------- |
| `0` (default) | Nothing                                                                                 |
| `1`           | A final summary: whether the fit converged, the number of iterations and the final loss |
| `2` or more   | Also the initial loss and the loss after each iteration                                 |

The estimators accept the same `verbose` values as a constructor parameter.

```
import torch
from entlearn import ClassificationHead, Input, Network, Recipe

X = torch.tensor([[0.0], [0.1], [0.9], [1.0]], dtype=torch.float64)
y = torch.tensor([0, 0, 1, 1])
recipe = Recipe.chain(Input(K=2), ClassificationHead())

network = Network.fit(recipe, X, y, seed=7, max_iter=20, verbose=1)
```

By default, the records are written to the terminal (standard error). To send them to your application's logging instead, pass your own `logging.Logger` as `logger`:

```
import logging

logger = logging.getLogger("my_application.eon")
network = Network.fit(recipe, X, y, seed=7, max_iter=20, logger=logger, verbose=2)
```

The summary is logged at `INFO` level and the loss of each iteration at `DEBUG` level.

With several initialisations (`n_inits` above one), `verbose=1` also reports the selected candidate and the scores of all candidates. When candidates are fitted in parallel, each candidate's records are reported once it has finished, in candidate order.

## Warnings

Warnings are independent of logging: they are emitted through Python's `warnings` module even with `verbose=0`. When candidates are fitted in parallel, their warnings are held back until every candidate has finished, then emitted in candidate order with their original category. A warning that your filters turn into an error is therefore raised after the candidates finish rather than inside a worker, and no model is returned. Python's warning filters are shared by every thread in a process. While candidates are fitted on threads, warnings raised meanwhile by your other threads are held back too.

| Warning                        | Meaning                                                                                                                                                                                |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `entlearn.ConvergenceWarning`  | The fit reached `max_iter` before converging                                                                                                                                           |
| `entlearn.LossIncreaseWarning` | The loss increased by more than the numerical noise expected at the computation dtype                                                                                                  |
| `UserWarning`                  | A problem with the data or the recipe, for example a hidden block with `K=1`, which passes on no information, or a requested input `K` larger than the number of eligible observations |

The convergence and loss-increase warnings, the warning about a hidden block with `K=1`, and the warnings about parts of a supplied `InitialState` that the fit does not use are also retained in `network.diagnostics.warnings`:

```
print(network.diagnostics.warnings)
```

Other warnings about the input data and the initialisation are emitted before fitting starts, and are not retained.

What should I do about a `ConvergenceWarning`?

The fit stopped at `max_iter` iterations, while the relative change of the loss was still larger than `tol`. The returned network is still usable, but you can increase `max_iter` or loosen `tol`. You can also [continue the fit](https://entropiclearning.github.io/entlearn/0.1.0/guide/networks/#continue-fitting) with `resume`.

With several initialisations, a single `ConvergenceWarning` lists every candidate that did not converge, identified by its index and seed, even when the selected candidate converged.

`LossIncreaseWarning` is not expected: each update step minimises the loss exactly, so the loss should never increase beyond the numerical noise. This is likely a bug.

What should I do about a `LossIncreaseWarning`?

Check the data for extreme values, and consider fitting in float64.

If the warning remains, please report it to the maintainers as an [issue on the entlearn repository](https://github.com/EntropicLearning/entlearn/issues).

Include the smallest example that reproduces the warning, using synthetic data if yours is private. Also include the entlearn version, the platform, the dtype and device, the seed, the recipe and the call, the complete warning, and `network.diagnostics.loss_history`.

## Schema and target mismatches

Categorical feature order, fitted cardinalities, and the meaning of each class code or target column must stay stable between fitting and later use. An unseen label or category, or a column whose meaning has changed, is rejected as a **schema error** before fitting or prediction. See [Expected input shapes](https://entropiclearning.github.io/entlearn/0.1.0/guide/networks/#expected-input-shapes) and [Categorical features](https://entropiclearning.github.io/entlearn/0.1.0/guide/estimators/#categorical-features) for the exact representations expected at each boundary.

## Unstable or disappointing predictions

Before concluding that the model is not performing well on your problem, try more independent initialisations and potentially automatic hyperparameter optimization on held-out data. See [Model selection](https://entropiclearning.github.io/entlearn/0.1.0/guide/model_selection/index.md) for examples.
