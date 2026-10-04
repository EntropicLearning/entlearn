# Contributing

Thanks for considering a contribution to `entlearn`.

## Set up

```bash
git clone https://github.com/EntropicLearning/entlearn.git
cd entlearn
uv sync --all-extras
```

`uv sync` installs the development and documentation dependency groups by default.

## Run the checks

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run poe docs-build
```

## How we like changes made

- Write British English, and Google-style docstrings in plain language.
- Name things with the [glossary](docs_site/concepts/glossary.md)'s terms.
- Identifiers are snake_case. Mathematical symbols such as `X`, `C`, `K`, `T` and `Wd` keep
  their case standing alone or leading a name (`C_cat`, `X_sq_wd_sum`) and are lower-case as a
  later qualifier (`sqdist_wd`). Name a transient buffer `scratch_*`, and a reused buffer by
  its content (`centroid_numerator`, `log_theta`).
- Group tests as methods of `Test...` classes, take `DTYPE` and `DEVICE` from `conftest`, and
  derive tolerances from the dtype. [tests/README.md](tests/README.md) is the full test
  policy.
- Declare a new dependency in `pyproject.toml`, and allow it for the subpackage that uses it in
  `scripts/test_source_policy.py`.

## Open a pull request

Open it against `main` on the public repository, and fill in the pull request template. A
marimo notebook under `examples/` that shows a new feature is welcome but not required, and
the maintainers update the docstrings and doc-site pages when they merge a change.

[AGENTS.md](AGENTS.md) explains how the codebase is built and why.
