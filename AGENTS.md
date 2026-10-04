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
scripts/           checks of the package boundary and the doc site, run when needed
examples/          marimo notebooks: the tutorial series
docs_site/         the MkDocs site
```

## How the code is built, and why

- **Exact block-coordinate descent.** Every update step sets one group of parameters to the
  exact minimiser of the loss with the rest fixed, so there is no learning rate and the loss
  never increases within an iteration
  ([why](docs_site/concepts/loss.md#why-the-loss-never-increases)). A new update step is exact
  too. `tests/network/test_monotonicity.py` checks every step, and pytest turns a
  `LossIncreaseWarning` into an error, so a failure there points at the update, with the
  tolerance left as it is.
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
  at most machine epsilon is removed; this is the case in which removing it leaves the loss
  unchanged ([Empty clusters](docs_site/concepts/affiliations.md#empty-clusters)).
- **One precision floor.** `_eps(dtype)` in `primitives/normalise.py` is the safe-log floor,
  the hard-or-soft temperature gate, the empty-cluster threshold and the relative-change
  guard, so the loss the solver minimises is the loss the monitor reports. The gate compares
  a temperature setting such as `epsilon` with it, never a per-row temperature such as
  `epsilon / T`, so no regime depends on the number of rows. A clamp that only turns `0/0`
  into `0` and must leave every positive value alone uses `torch.finfo(dtype).tiny` instead:
  the covariance rescaling in `primitives/manifold.py` and the plotting helpers.
- **Update steps allocate nothing.** Workspaces are built when the data is staged, and block
  buffers are sized at the starting cluster count and read through `[:K]` views, which keeps
  fits fast and reproducible run to run. `assert_zero_alloc` in `tests/_alloc.py` checks it.
  Staging, initialisation, workspace construction and pruning allocate.
- **Inference mode throughout.** Every public `Network` entry point runs under
  `torch.inference_mode()`, so every fitted tensor is an inference tensor, and code that
  writes into one (resuming, fine-tuning, loading) wraps the write in inference mode too.
- **Data are checked, then used as given.** Every operation stages its data through
  `_stage_data` in `network/data.py`, and a bad input raises. Callers convert dtypes, devices
  and label codes themselves; the estimators do it for tabular data.
- **Checks raise, from one owner.** `python -O` strips `assert`, so a correctness check raises
  `ValueError` for a bad public input and `RuntimeError` for a broken invariant, and `assert`
  serves type narrowing. Each check lives in one function that every path calls, which keeps
  the rules from drifting apart.
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
- The `scripts/` checks run on demand: [scripts/README.md](scripts/README.md) says which one
  checks what.
