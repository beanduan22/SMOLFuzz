# SMOLFuzz

**S**ynthesizing **M**odels with **O**pen **L**LMs for **Fuzz**ing Deep Learning Libraries.

SMOLFuzz is a differential fuzzer that uses a local LLM to generate PyTorch/TensorFlow models, then compares CPU vs GPU numerical output to find divergence bugs in deep learning library kernels.

---

## How It Works

```
API pool (1,226 PT / 1,135 TF selectable)
      │
      ▼
Multi-Roulette Selector          ← 11 functional groups, proportional budget
      │  api_set (20 APIs)
      ▼
LLM Synthesizer (Ollama)         ← generates deterministic Model + make_inputs()
      │  Python code
      ▼
Executor (subprocess, 30s limit) ← CPU run, GPU run (×2 for non-det check)
      │  outputs
      ▼
Differential Oracle              ← rel_err > 100× dtype tolerance → bug
      │
      ▼
Mutation loop (5 strategies)     ← add_noise · scale_small · mask · uniform · scale_large
      │  per-strategy adaptive counter, sweep+reset, 10-model early stop
      ▼
Bug report (.json + .repro.py + tensors)
```

### API Classification — 11 Functional Groups

APIs are placed into 11 groups so the selector builds long, diverse dependency chains instead of sampling dominated by the mathematics group.

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
| | **Selectable pool** | **1,226** | **1,135** |

†Excluded at runtime — random ops destroy CPU/GPU determinism.

See [`docs/api_classification.md`](docs/api_classification.md) for full group definitions and exclusion policy.

---

## Installation

```bash
# Python ≥ 3.10, PyTorch ≥ 2.1 with CUDA, TensorFlow ≥ 2.13 with GPU
pip install -r requirements.txt

# Local LLM via Ollama (default: qwen2.5-coder:32b, llama3.3:70b)
ollama pull qwen2.5-coder:32b
```

## Usage

```bash
# Quick validation (5 models)
python3 -m smolfuzz.main --mode subset

# Full PyTorch run (300 models)
python3 -m smolfuzz.main --mode full --models 300 --budget 60

# TensorFlow run
python3 -m smolfuzz.run_tf --models 300

# Custom LLM
python3 -m smolfuzz.main --llm-models "qwen2.5-coder:32b,llama3.3:70b"
```

Results are written to `results/` (gitignored).

---

## Confirmed Bugs

### Fuzzer-generated bugs — April 2026 run

`bugs/github/new/pytorch/` contains **4 confirmed PyTorch CPU/GPU divergence bugs** found by a 300-model fuzzer run (102 PT models started, 60 s mutation budget each).

All 4 are `INCONSISTENT` type: the same model with identical weights produces numerically different outputs on CPU vs GPU under large-scale input mutation (`scale_large` ×3, `mask` ×1).

| Model | APIs (key) | Mutation | rel\_err | Shape |
|-------|-----------|---------|---------|-------|
| m0028 | Linear · BatchNorm1d · sin · cummax | scale\_large | 838× | [4, 4] |
| m0046 | Linear · BatchNorm1d · Mish · sin · fmod | scale\_large | 163× | [4, 4] |
| m0050 | Linear · BatchNorm1d · silu · sin · arctanh · autograd.grad | scale\_large | 259× | [64] |
| m0088 | Linear · BatchNorm1d · ELU · addcdiv · linalg.matrix\_norm | mask | 4459× | [4, 8] |

Root cause for all four: `BatchNorm1d` running statistics (Welford online algorithm on CPU vs cuDNN on GPU) diverge under extreme-magnitude or sparse inputs, amplified by subsequent trig / norm / element-wise ops.

**Sample output (m0028 — CPU vs GPU, first 4 values):**

```
CPU: [0.2614, 0.2614, 0.2614, 0.2614, ...]
GPU: [0.2311, 0.2311, 0.2311, 0.2311, ...]
rel_err = 838×   (BUG_MARGIN threshold: 100×)
```

**Sample output (m0088 — CPU vs GPU, first 8 values):**

```
CPU: [-0.1289, -0.1289, -0.1289, -0.1289, -0.1289, -0.1289, -0.1289, -0.1289]
GPU: [ 0.0000,  0.0000,  0.0000, -0.1289,  0.0000,  0.0000, -0.1289,  0.0000]
rel_err = 4459×
```

Reproducers: `bugs/github/new/pytorch/bug_inconsistent_m00{28,46,50,88}_*.repro.py`

---

### Manually crafted bugs — 50 CPU/GPU divergence scripts

`high_confidence_bugs/new_bugs/` contains **50 standalone scripts** that directly reproduce known CPU vs GPU divergence bugs in PyTorch and TensorFlow:

```bash
$ python3 high_confidence_bugs/new_bugs/pt_cumprod_f32_large.py
cpu_max_err = 5.9604e-08  (CPU promotes f32→f64 internally)
gpu_max_err = 5.4827e-03  (GPU accumulates in f32)
GPU/CPU error ratio: 91986x
BUG CONFIRMED: PT cumprod f32 N=1M GPU is ~97000x less accurate than CPU

$ python3 high_confidence_bugs/new_bugs/tf_mean_f16_wrong.py
ref = -0.003780
cpu = -0.000000  (WRONG: float16 overflow → 0.0)
gpu = -0.003780
BUG CONFIRMED: TF CPU reduce_mean returns 0.0 for float16 N=65536

$ python3 high_confidence_bugs/new_bugs/pt_lstsq_rankdef.py
NumPy: [ 0.8333  0.3333 -0.1667]
CPU:   [ 0.8333  0.3333 -0.1667]
GPU:   [ 0.5814  0.7631  0.0000]   <-- BUG
BUG CONFIRMED: PT lstsq GPU gives wrong answer for rank-deficient matrix
```

| Category | PyTorch | TensorFlow |
|----------|---------|------------|
| Cumulative ops (cumsum/cumprod) | 8 | 8 |
| Linear algebra (SVD/eigh/norm/pinv) | 8 | 6 |
| Wrong value (NaN/zero/Inf) | 3 | 6 |
| NaN/Inf casting to int | 2 | 3 |
| Matmul TF32 | 1 | 1 |
| Linear solver (lstsq) | 2 | 1 |
| **Total** | **25** | **25** |

---

### Minimal fuzzer bugs — 27 PT + 2 TF

`high_confidence_bugs/minimal/` contains **29 minimal reproducer scripts** found by the SMOLFuzz fuzzer in earlier runs.

| File | Bug type | Key APIs | Signal |
|------|----------|---------|--------|
| pt080.py | Inconsistent | sin/cos large input + weight_norm | L2 `5.9e-03` |
| pt147.py | Inconsistent | BatchNorm1d Welford vs cuDNN | L2 `8.35e+01` |
| pt192.py | Asymmetric NaN | logsumexp: CPU NaN, GPU finite | 1 position |
| pt295.py | Inconsistent | BatchNorm1d train mode twice; near-zero vs exact zero | L2 `2.83e+00` |
| pt346.py | Inconsistent | sin/cos + hann_window + cumprod + mse_loss | L2 `3.28e+04` |
| pt396.py | Asymmetric Inf | logdet: CPU=-inf, GPU=-108.34 | — |
| pt441.py | CPU crash / GPU ok | corrcoef → cholesky on near-singular matrix | CPU RuntimeError |
| pt486.py | Inconsistent (output + grad) | sin/cos + autograd.grad | output+grad both diverge |
| tf067.py | Inconsistent | BatchNorm + DCT + min_max_norm | L2 `1.92e-03` |
| tf106.py | Inconsistent | BatchNorm + digamma(float64) | L2 `4.91e-01` |
| *(+19 more)* | | | |

```bash
$ python3 high_confidence_bugs/minimal/pt295.py
CPU: tensor([[4.2480e-12, 1.0000e+00, 1.8019e-12, 1.0000e+00], ...])
GPU: tensor([[0., 0., 0., 0.], ...])
L2: 2.8284e+00

$ python3 high_confidence_bugs/minimal/pt441.py
GPU: tensor([1.0000, 1.0006, 1.0015, 1.0018])
CPU crash: cholesky: The factorization could not be completed...
```

---

## Project Structure

```
smolfuzz/
├── main.py           # PyTorch fuzzing entry point
├── run_tf.py         # TensorFlow fuzzing entry point
├── api_loader.py     # API list loader + 11-group classifier
├── selector.py       # Multi-roulette selector (per-group proportional budget)
├── synthesizer.py    # LLM model synthesis + self-repair loop
├── executor.py       # Subprocess executor + 5 mutation strategies
├── oracle.py         # Differential oracle (CPU vs GPU comparison)
├── llm_client.py     # Ollama LLM client
├── prompts.py        # LLM prompt templates
├── run_both.py       # Run PT + TF in parallel
├── torch_valid_apis.txt  # PyTorch API pool (1,329 classified)
├── tf_valid_apis.txt     # TensorFlow API pool (2,215 classified)
├── docs/
│   └── api_classification.md  # 11-group definitions + counts
├── bugs/github/new/
│   ├── pytorch/      # 4 real bugs from April 2026 run
│   └── SUMMARY.md    # Triage notes
├── high_confidence_bugs/
│   ├── new_bugs/     # 50 manually crafted CPU/GPU divergence scripts
│   └── minimal/      # 27 PT + 2 TF fuzzer-generated minimal reproducers
└── Fuzzing/          # Original prototype (GPT-4 generated models)
```

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.1 with CUDA
- TensorFlow ≥ 2.13 with GPU support
- [Ollama](https://ollama.ai) with at least one code-capable model
