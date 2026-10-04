# Installation

Entlearn uses PyTorch and runs on CPU or supported accelerators.
See `pyproject.toml` in the GitHub repository for the dependency requirements.

!!! warn "This is a prerelease!"

    The package is not yet on PyPI, so please install it from GitHub with `pip install git+https://github.com/EntropicLearning/entlearn` (or `uv add git+https://github.com/EntropicLearning/entlearn`).


To install the package from PyPI run:

```bash
pip install entlearn
```

## Optional features

Install the corresponding extras to use the scikit-learn estimators and plotting tools:

```bash
pip install "entlearn[sklearn,plotting]"
```

These are optional dependencies, so you can install only what you need:

| Extra | Adds |
| --- | --- |
| `sklearn` | Scikit-learn estimators, tabular input support and model selection |
| `plotting` | Plotly figures of fitted networks |
| `rich` | Formatted terminal reports |

Marimo is used for the interactive notebooks and can be installed separately with `pip install marimo`.

## Check the installation

Run the following command to check that the package can be imported:

```bash
python -c "from entlearn import Network, Recipe; print('entlearn is ready')"
```

You can now [fit your first network](../tutorials/series.md#tutorial-1).

## Accelerators

By default, everything is run on the CPU. 
If you have a GPU that supports CUDA (meaning `torch.cuda.is_available()` returns `True`), you can use it by simply specifying it to the `device` argument of a `Network`.

Choose the device and floating-point precision explicitly:

```python
import torch

requested = "cpu"  # Change to "cuda" to use an available CUDA device.
if requested == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable; choose CPU.")
device = torch.device(requested)
dtype = torch.float32
```

For a `Network`, place inputs and targets on that device, with compatible dtypes. Query tensors must match the fitted network's device and computation dtype.
Estimators instead accept `device=device` and `dtype=dtype` and convert their tabular inputs for you. See [Networks](networks.md#expected-input-shapes) and [Estimators](estimators.md).

On Apple hardware, check `torch.backends.mps.is_available()` before requesting `"mps"`.
MPS requires float32: loading a saved float64 model on MPS raises an error.
Some fitting operations, such as manifold eigensolvers, may require CPU fallback. Set `PYTORCH_ENABLE_MPS_FALLBACK=1` before starting Python if needed. 
This permits CPU work and may be slower. Alternatively, fit on CPU.

[Loading a saved network](networks.md#save-the-network) defaults to CPU and preserves its dtype.
To request another device, pass it using the `device` argument.

For how the device affects repeated fits, see [Reproducible fits](networks.md#reproducible-fits).

## Installation for developers

To work on the package itself, follow the [contributing guide](https://github.com/EntropicLearning/entlearn/blob/main/CONTRIBUTING.md) to set up a development environment and run the checks.
