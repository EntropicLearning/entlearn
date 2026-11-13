# AGENTS.md

entlearn is a pure-tensor PyTorch implementation of EON, Entropy-Optimal Networks
([Bassetti et al. 2025](https://arxiv.org/abs/2506.17940)), with a scikit-learn interface.
An EON model stacks entropy-regularised clusterings as blocks, links them by transition
matrices and fits them by exact block-coordinate descent.

The doc site describes what the package does. Read the page before changing the code it
describes, and update it in the same change:

- [Concepts](docs_site/concepts/index.md): the loss, the six objects and their exact update
  steps. [Notation](docs_site/concepts/notation.md) holds the symbols.
- [Glossary](docs_site/concepts/glossary.md): the names to use in code, docstrings and pages.
- [Guide](docs_site/guide/introduction.md): the public interface. The API reference is
  rendered from the docstrings.
- [tests/README.md](tests/README.md): how the tests are organised, and the tolerances they use.

## Layout

```
src/entlearn/
  recipe.py        Recipe and its descriptions: Input, ManifoldInput, Hidden,
                   ClassificationHead, RegressionHead, Connection, Coupling
  network/         Network (model.py). build.py compiles a Recipe, data.py stages data,
                   fit.py runs the iterations, predict.py the single forward pass;
                   blocks/ and connections.py hold the fitted state; session.py holds the
                   compiled graph; initialisation/, persistence/, calibration.py
  primitives/      stateless tensor operations the blocks are built on
  helpers/         optional, state-free tools
  scikit_adapter/  EONClassifier and EONRegressor
  plotting/        optional Plotly figures; Network.plot builds on them
tests/             mirrors src/; conftest.py selects the device and dtype
scripts/           documentation build tools and on-demand package and doc-site checks
examples/          marimo notebooks: the tutorial series
docs_site/         the MkDocs site
```

## How the code is built, and why

- **Exact block-coordinate descent.** Every update step sets one group of parameters to the
  exact minimiser of the loss with the rest fixed, so there is no learning rate and the loss
  never increases within an iteration ([why](docs_site/concepts/index.md#one-iteration)).
  Entropy is exact (`torch.special.entr`), so each softmax is the true minimiser of its
  step. A new update step is exact too. `tests/network/test_monotonicity.py` checks every
  step, and pytest turns a `LossIncreaseWarning` into an error, so a failure there points at
  the update, with the tolerance left as it is.
- **Description apart from runtime.** An immutable `Recipe` describes the model and holds no
  tensors; `Network.fit` compiles it into the runtime blocks that hold the fitted state. A
  search replaces descriptions (`replace_block`, `replace_connection`, `set_params`) and
  rebuilds, which keeps the scikit-learn adapters a thin layer over `Network`, and a bad
  configuration fails when the `Recipe` is built, with a named message.
  [Package and Recipes](docs_site/guide/package.md#what-shapes-can-a-model-have) lists the
  shapes a model can take; `network/build.py` checks them.
- **One public fitted type.** `Network` is the fitted object users see; runtime blocks,
  connection state, workspaces and fit sessions stay private. `scripts/test_public_surface.py`
  pins every exported name, so a new export gets a line there.
- **A closed set of blocks.** The runtime has five concrete block classes, grouped by type
  unions such as `_InputBlock = _StandardInputBlock | _ManifoldInputBlock`, and code that
  branches over them uses `match` with `assert_never`. A new block kind extends each union,
  and the type checker points at every branch that needs it.
- **State lives with what outlives the fit.** A block owns its parameters, affiliations and
  instance weights. A transition matrix belongs to the block it leads into (its `incoming`
  table), so each block and its incoming state are compacted together when a cluster is
  pruned (`_CompiledGraph.prune_` in `network/session.py`).
- **Pruning is the one empty-cluster policy.** A cluster whose unweighted affiliation mass is
  at or below `_tiny(dtype)` is removed ([Pruning](docs_site/concepts/learning.md#pruning)).
- **Three numerical thresholds**, each with one role and a helper in `primitives/normalise.py`:
  - Zero selects a setting. `_is_soft(temp)` is `temp > 0` in every dtype, and a zero weight
    removes its term.
  - `_tiny(dtype)` marks a computed value as numerically zero. It is the empty-mass threshold,
    the floor of `floored_log_` for every probability logarithm, and the clamp on a divisor.
  - `_eps(dtype)` compares a quantity with another on the same scale: round-off tolerances,
    the stopping and loss-increase tests, the relative manifold ridge and the noise floor on
    the regression targets' spread in label-aware seeding.

  A softmax subtracts the minimum cost before scaling through `_scale_ratio_`, and a value
  that could overflow, such as the instance-weight log-normaliser, is kept as a shift and a
  shifted part.
- **Update steps allocate nothing.** Workspaces are built when the data is staged, and block
  buffers are sized at the starting cluster count and read through contiguous views of the
  active count (`_scratch_view`), which keeps fits fast and reproducible run to run.
  `assert_zero_alloc` in `tests/_alloc.py` checks it. Staging, initialisation, workspace
  construction and pruning allocate.
- **Inference mode throughout.** Every public `Network` entry point runs under
  `torch.inference_mode()`, so every fitted tensor is an inference tensor, and code that
  writes into one (resuming, fine-tuning, loading) wraps the write in inference mode too.
- **Data are checked, then used as given.** Every operation stages its data through
  `_stage_data` in `network/data.py`, and a bad input raises. Callers convert dtypes, devices
  and label codes themselves; the estimators do it for tabular data.
- **Checks raise, from one owner.** `python -O` strips `assert`, so a correctness check raises
  `ValueError` for a bad public input and `RuntimeError` for a broken invariant, and `assert`
  serves type narrowing. A valid input that cannot be honoured in full, or that another option
  overrides, warns instead and says what was done. Each check lives in one function that every
  path calls, which keeps the rules from drifting apart.
- **One saved representation.** The snapshot in `network/persistence/snapshot.py` serves the
  `.safetensors` file, pickling and process-worker transport, so every route runs the same
  validation.
- **A light core.** `import entlearn` loads only the standard library, `torch` and
  `safetensors`. NumPy, pandas, SciPy, scikit-learn and the plotting libraries belong to
  `helpers/`, `scikit_adapter/` and `plotting/`, and load when those are used; joblib
  loads only when `network/dispatch.py` fits candidates in parallel.
  `scripts/test_import_isolation.py` and `scripts/test_source_policy.py` check the boundaries.
- **In-place functions end in an underscore**, as in PyTorch.

## Commands

```bash
uv sync --all-extras                # development and documentation groups included
scripts/gate                        # every check below except the docs build; run before a PR
uv run pytest                       # the suite; EON_TEST_* variables select the lane
uv run ruff check . && uv run ruff format --check . && uv run ty check
uv run poe docs-build               # strict MkDocs build; `uv run poe docs` previews
uv run pytest scripts/test_release_docs.py
```

- CI runs the cpu lane in `float64` and in `float32`, and every device and dtype cell stays
  green: a tolerance that holds only in `float64` is a bug. [tests/README.md](tests/README.md)
  lists the lane variables.
- Continuous feature data in the tests stays in or near `[0, 1]`, the regime the fixtures and
  tolerances assume.
- The `scripts/` checks stay out of CI and run in `scripts/gate`:
  [scripts/README.md](scripts/README.md) says which one checks what.
- `git config core.hooksPath scripts/hooks` installs a pre-commit hook that runs Ruff.
