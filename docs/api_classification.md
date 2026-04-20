# API Functional Classification

SMOLFuzz places every API into one of **11 functional groups** derived from the official
PyTorch and TensorFlow documentation. The groups are used by the multi-armed-roulette
selector (`selector.py → MultiRouletteSelector`) to ensure each generated model contains
a **long and diverse API dependency chain** rather than sampling uniformly from the full
pool (which would be dominated by the mathematics group).

## The 11 Groups

| # | Group | Description |
|---|-------|-------------|
| 1 | `creation_conversion` | Tensor creation and dtype/device conversion — ops described in the docs as creating a new tensor or changing its dtype or device (e.g., `torch.zeros`, `tf.constant`, `torch.Tensor.to`, `tf.cast`) |
| 2 | `mathematics` | Arithmetic, linear algebra, trigonometry, reduction, cumulative, and spectral ops (e.g., `torch.matmul`, `tf.math.*`, `torch.linalg.*`, `tf.signal.*`) |
| 3 | `reshaping` | Reshape, view, slice, transpose, concatenate, broadcast, scatter, and gather ops (e.g., `torch.reshape`, `tf.concat`, `torch.gather`, `tf.pad`) |
| 4 | `logical` | Comparison, boolean, and conditional ops (e.g., `torch.eq`, `tf.where`, `torch.logical_and`, `tf.cond`) |
| 5 | `distributions` | Parametric probability distribution families — ops whose documentation describes them as sampling from or computing properties of a named statistical distribution (e.g., `torch.distributions.*`, `tf.random.gamma`, `tf.random.poisson`) |
| 6 | `forward_layers` | Neural-network layers, activations, losses, metrics, regularizers, constraints, and initializers (e.g., `torch.nn.Linear`, `tf.keras.layers.*`, `tf.nn.*`, `torch.nn.functional.*`) |
| 7 | `gradients_optim` | Autograd, optimizers, and gradient utility ops (e.g., `torch.autograd.*`, `tf.GradientTape`, `torch.optim.*`, `tf.keras.optimizers.*`) |
| 8 | `storage_serial` | Save, load, and serialization ops (e.g., `torch.save`, `torch.load`, `tf.train.Checkpoint`) |
| 9 | `random_generation` | Seeded random tensor generation — **excluded from the selectable pool** (see §Exclusion Policy below) |
| 10 | `model_io` | Model hub, checkpoint, scripting, and application helpers (e.g., `torch.hub.*`, `tf.keras.models.*`, `tf.keras.applications.*`) |
| 11 | `misc` | Everything not matched by the ten groups above |

## API Counts

Counts include all 11 groups (`drop_excluded=False`).  
Random generation is classified but removed from the selectable pool at runtime.

| # | Group | PyTorch | TensorFlow |
|---|-------|--------:|-----------:|
| 1 | Tensor creation / conversion | 127 | 38 |
| 2 | Tensor mathematics | 577 | 306 |
| 3 | Tensor reshaping | 115 | 44 |
| 4 | Logical operations | 86 | 18 |
| 5 | Probability distributions | 13 | 9 |
| 6 | Forward layers | 278 | 566 |
| 7 | Gradients and optimization | 28 | 44 |
| 8 | Storage / serialization | 11 | 3 |
| 9 | Random generation† | 23 | 37 |
| 10 | Model I/O | 16 | 125 |
| 11 | Miscellaneous utilities | 55 | 1,025 |
| | **Total** | **1,329** | **2,215** |

†Excluded from the selectable pool at runtime.

## Exclusion Policy

The following API categories are removed **before** classification and are never
sampled by the selector:

| Category | Reason |
|----------|--------|
| `random_generation` | Destroys CPU/GPU determinism — every differential test would diverge by construction |
| Experimental / private (`torch._*`, `tf.compat.v1.*`, `tf.experimental.*`) | Unstable interfaces not part of the stable tensor runtime |
| Compiler front-ends (`torch.jit`, `torch.compile`, `tf.function`, `tf.autograph`, …) | Compilation stack, not numerical operators |
| Distributed / data / profiler infra (`torch.distributed`, `tf.data.*`, `tf.io.*`, …) | Non-numerical infrastructure |

After exclusion the **selectable pool** is:

| Framework | Selectable APIs |
|-----------|---------------:|
| PyTorch | 1,226 |
| TensorFlow | 1,135 |

## Implementation

| File | Role |
|------|------|
| `api_loader.py` | Defines `_RULES` (PyTorch) and `_TF_RULES` (TensorFlow), applies exclusion via `is_excluded()`, and exposes `load_and_classify()` |
| `selector.py → MultiRouletteSelector` | Accepts the group dict from `load_and_classify()` and allocates the per-group API budget using largest-remainder proportional allocation (Eq. 3), then samples within each group by roulette wheel |
| `main.py` | Calls `load_and_classify(api_file)` → passes groups to `MultiRouletteSelector` |
| `run_tf.py` | Calls `load_and_classify(api_path, framework="tf")` → passes groups to `MultiRouletteSelector` |

The selector draws `N` APIs per model, distributing them across groups in proportion to
group size so that no single group (e.g., `mathematics`) dominates the selection.
