# Scripts

This directory holds the gate, documentation build tools and checks outside the ordinary
`tests/` collection. CI does not run these checks; `scripts/gate` does. Run one alone when a
change touches what it guards, for example `uv run pytest scripts/test_public_surface.py`.

- `gate`: Ruff's format and lint checks, `ty`, every check below and the suite. Run it before
  opening a pull request; extra arguments go to the suite's pytest.
- `hooks/pre-commit`: Ruff's format and lint checks before each commit. Install it with
  `git config core.hooksPath scripts/hooks`.

- `test_public_surface.py`: the public import surface of the package. Run it after adding,
  removing or renaming an export.
- `test_import_isolation.py`: ordinary imports do not load call-time optional dependencies.
  Run it after changing imports in `entlearn` or `entlearn.helpers`.
- `test_source_policy.py`: dependency boundaries, class-based test organisation, and the
  prose rules for the comments and docstrings under `src/` and `tests/` and for doc-site
  pages. Run it after moving code between modules, adding a dependency, or editing a page or
  the comments and docstrings of a source or test file.
- `test_release_docs.py`: the Python examples on every guide page run, in page order, against
  the installed package. CI also runs this one.
- `test_doc_math.py`: the site's Markdown extensions mark up TeX for MathJax and leave code
  examples alone. Run it after changing `markdown_extensions` in `mkdocs.yml`.

## Tutorial exports

`uv run poe docs-export` runs `export_tutorials.py` to export every tutorial in `examples/`
as HTML in `docs_site/tutorials/notebooks/`. It uses the project's environment, not a
sandbox, so the rendered pages use the checkout's entlearn. A failed notebook stops the
build. These HTML files and marimo session snapshots are generated output, not source.

`uv run poe docs-build` exports the tutorials before the strict MkDocs build;
for a local tutorial preview, run `uv run poe docs-export` before `uv run poe docs`.
After changing a notebook during a preview, rerun the export to refresh its output.
Versioned publishing with `mike` also needs this export step before deployment.
