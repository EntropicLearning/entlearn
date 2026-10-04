# Entlearn

> [!NOTE]
> This is a pre-release. `entlearn` is not on PyPI yet, so until it is, install it from
> GitHub with `pip install git+https://github.com/EntropicLearning/entlearn` (or
> `uv add git+https://github.com/EntropicLearning/entlearn`).

[![PyPI version](https://img.shields.io/pypi/v/entlearn.svg)](https://pypi.org/project/entlearn/) [![Python versions](https://img.shields.io/pypi/pyversions/entlearn.svg)](https://pypi.org/project/entlearn/) [![CI](https://github.com/EntropicLearning/entlearn/actions/workflows/ci.yml/badge.svg)](https://github.com/EntropicLearning/entlearn/actions/workflows/ci.yml) [![Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://entropiclearning.github.io/entlearn/) [![Licence](https://img.shields.io/github/license/EntropicLearning/entlearn.svg)](https://github.com/EntropicLearning/entlearn/blob/main/LICENSE)

`entlearn` is a pure-tensor PyTorch implementation of Entropy-Optimal Networks (EON):
entropy-regularised clusterings stacked as blocks and fitted by exact block-coordinate descent.

[Documentation](https://entropiclearning.github.io/entlearn/) · [Examples](https://github.com/EntropicLearning/entlearn/tree/main/examples) · [Skills](https://github.com/EntropicLearning/entlearn_skills) · [Contributing](https://github.com/EntropicLearning/entlearn/blob/main/CONTRIBUTING.md)

## Quick start

```bash
pip install entlearn  # or: uv add entlearn
```

```python
import torch
from entlearn import ClassificationHead, Input, Network, Recipe

X = torch.rand(200, 4, dtype=torch.float64)
y = (X[:, 0] > 0.5).long()
recipe = Recipe.chain(Input(K=8, epsilon=0.01), ClassificationHead())
network = Network.fit(recipe, X, y)
predictions = network.predict(X).argmax(dim=-1)
```

## Licence

`entlearn` is free software under the
[GNU AGPL-3.0](https://github.com/EntropicLearning/entlearn/blob/main/LICENSE). You may use,
study, change and share it; if you distribute it, or run a changed version as a service for
others, you share your source code under the same licence.

## Citation

If you use `entlearn`, please cite Bassetti et al. (2025),
[An entropy-optimal path to humble AI](https://arxiv.org/abs/2506.17940).
[CITATION.cff](https://github.com/EntropicLearning/entlearn/blob/main/CITATION.cff) also gives the software citation.

## Funding

This work has received support from the AI4LUNGS project, funded by the European Union's
Horizon 2020 research and innovation programme. AI4LUNGS applies AI to medical data, but EON
is a general-purpose method that applies to many domains beyond healthcare.
