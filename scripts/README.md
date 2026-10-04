# Scripts

These checks sit outside the ordinary `tests/` collection. Run one when a change touches what
it guards, for example `uv run pytest scripts/test_public_surface.py`; `uv run pytest scripts/`
runs them all.

- `test_public_surface.py`: the public import surface of the package. Run it after adding,
  removing or renaming an export.
- `test_import_isolation.py`: ordinary imports do not load call-time optional dependencies.
  Run it after changing imports in `entlearn` or `entlearn.helpers`.
- `test_source_policy.py`: dependency boundaries, class-based test organisation and
  source-prose rules. Run it after moving code between modules or adding a dependency.
- `test_release_docs.py`: mkdocstrings targets and documented example imports resolve
  through public `entlearn` namespaces. CI also runs this one.
- `test_doc_math.py`: the site's Markdown extensions mark up TeX for MathJax and leave code
  examples alone. Run it after changing `markdown_extensions` in `mkdocs.yml`.
