# Tutorials

In this section, you can find a set of tutorial notebooks meant to provide a starting point for learning how to use the package. [Tutorial series](series.md) lists the available notebooks, with source and molab links.

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

For more information, see the [marimo documentation](https://marimo.io/)
