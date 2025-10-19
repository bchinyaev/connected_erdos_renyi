# Connected ER Generators — Experiments

This repository accompanies the paper:

Chinyaev, “A Method for Generating Connected Erdős–Rényi Random Graphs”

- arXiv abs: https://arxiv.org/abs/2504.05907

It contains a small Python module with functions for generating connected random graphs and a demo notebook:
- `connected_generation.py` — reference implementations
- `connected_generation.ipynb` — examples and experiments

Abstract (for context):

We propose a novel exact algorithm for generating connected Erd\H{o}s--R\'enyi random graphs $G(n,p)$. The method couples the graph exploration process to an inhomogeneous Poisson random walk, which yields an exact sampler that runs in $O(n)$ time in the sparse regime $p=c/n$. We also show how the method extends to the $G(n,M)$ model via an additional acceptance--rejection step.

Provided functions

- `connected_gnp_generation(N, p, seed=None)` — exact sampler for connected $G(n,p)$ (sparse regime recommended: $p=c/n$).
- `connected_gnp_generation_fast(N, p, seed=None)` — optimized variant with the same output law.
- `connected_gnm_generation_fast(N, M, seed=None)` — exact connected $G(n,M)$ via an extra acceptance–rejection on the number of additional edges.

Quick start

```bash
pip install numpy networkx  # minimal requirements
```

```python
from connected_generation import (
    connected_gnp_generation_fast,
    connected_gnm_generation_fast,
    visualize_G
)
# Connected G(n,p) in the sparse regime p = c/n
N = 100
c = 1.0
G = connected_gnp_generation_fast(N=N, p=c/N, seed=42)
visualize_G(G)

# Connected G(n,M) with exactly M edges
H = connected_gnm_generation_fast(N=100, M=105, seed=123)
visualize_G(H)
```

Notes

- The notebook `connected_generation.ipynb` reproduces small-scale experiments and usage examples.
- The implementations are distilled from research code used for the paper and aim for clarity over micro-optimizations.
