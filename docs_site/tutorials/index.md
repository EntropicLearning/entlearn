# Tutorials

The [tutorial series](series.md) walks through fitting, inspecting and selecting a
network. Each tutorial has a rendered page with its outputs, a Python source link,
and a molab link for running it in server mode.

## How to run the notebooks
The notebooks are included in the source code, so you can run them locally with ease.
From a [development checkout](../guide/installation.md#installation-for-developers):

```bash
uv sync --all-extras
uv run marimo edit --no-sandbox examples/01_first_network.py
```

For a read-only interactive app:

```bash
uv run marimo run --include-code --no-sandbox examples/01_first_network.py
```

## Share a snapshot

```bash
uv run marimo export html --no-sandbox examples/01_first_network.py -o first-network.html
```

An HTML snapshot preserves figures but does not rerun Python when a control changes.

For more information, see the [marimo documentation](https://marimo.io/).
