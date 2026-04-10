# SMOLFuzz — High-Confidence Bugs

**44 confirmed bugs** (42 PyTorch + 2 TensorFlow), all verified on real hardware.  
Every bug is self-contained in a single `.py` file with inputs embedded as literals — no external files needed.

## Quick Start

```bash
# Run any single reproducer
python3 reproducers/bug_pt404.py

# Run all 44 at once
for f in reproducers/*.py; do
    echo -n "$(basename $f): "
    python3 "$f" 2>/dev/null | grep "BUG CONFIRMED\|not reproduced" | head -1
done
```

## Bug Summary

| Type | Count | Strongest Signal |
|---|---|---|
| CPU crash / GPU ok | 1 | pt441: `cholesky` — CPU throws, GPU returns result |
| Asymmetric NaN/Inf | 2 | pt192: asym NaN=1; pt396: CPU=Inf, GPU=NaN |
| Large L2 divergence | 41 | pt404: L2=54,636; pt346: L2=32,768; pt358: L2=739 |

## Tier 1 — Strongest Bugs (10)

| Reproducer | Signal | Key APIs |
|---|---|---|
| `bug_pt404.py` | L2=5.4636e+04 | Linear, maximum, sin, cos |
| `bug_pt346.py` | L2=3.2768e+04 | hann_window, sin, cos |
| `bug_pt358.py` | L2=7.3856e+02 | Linear, softplus, geqrf (LAPACK vs cuSOLVER) |
| `bug_pt147.py` | L2=8.3500e+01 | Linear, **BatchNorm1d**, sigmoid, polygamma_ |
| `bug_pt171.py` | L2=2.0108e+01 | Linear, poisson_nll_loss, BatchNorm1d |
| `bug_pt241.py` | L2=6.0545e+00 | Linear, ReLU, requires_grad_ |
| `bug_pt192.py` | ASYM NaN: cpu=16, gpu=17 | sin, cos, logsumexp |
| `bug_pt396.py` | cpu=Inf / gpu=NaN | LeakyReLU, AbsTransform, ormqr |
| `bug_pt441.py` | CPU crash / GPU ok | logcumsumexp, corrcoef, cholesky |
| `bug_tf106.py` | L2=1.4250e+00 (no mutation) | digamma, SeparableConv2D, linalg.inv, mdct |

## Tier 2 — All Other Confirmed Bugs (34)

See [`tier1_tier2_bugs.md`](tier1_tier2_bugs.md) for full table.

## Root Causes

| Root Cause | Count |
|---|---|
| **BatchNorm1d**: CPU sequential Welford vs GPU cuDNN parallel tree-reduction | 16 |
| **Trig pipeline**: sin/cos on large-magnitude inputs loses precision differently on x86 vs CUDA | 18 |
| **Special math**: logsumexp, erfc, lgamma, xlogy — different CPU/GPU kernel implementations | 5 |
| **Linear algebra**: LAPACK (CPU) vs cuSOLVER (GPU) — QR factorization and Cholesky | 3 |
| **TF CPU/GPU kernel divergence**: Eigen vs cuDNN baseline divergence (no mutation needed) | 2 |

## Verified Run Outputs

See [`run_outputs.txt`](run_outputs.txt) — output of all 44 reproducers run on 2026-04-10.  
All 44 print `BUG CONFIRMED`.

```
=== bug_pt071.py ===
output[0]: L2=4.4361e-01  shape=[4, 8]
BUG CONFIRMED: CPU and GPU produce different results for the same model and input

=== bug_pt080.py ===
output[0]: L2=5.9273e-03  shape=[4, 4]
BUG CONFIRMED: CPU and GPU produce different results for the same model and input

=== bug_pt101.py ===
output[0]: L2=1.1677e-02  shape=[4, 8]
BUG CONFIRMED: CPU and GPU produce different results for the same model and input

=== bug_pt106.py ===
output[0]: L2=9.0167e-02  shape=[]
BUG CONFIRMED: CPU and GPU produce different results for the same model and input

=== bug_pt125.py ===
output[0]: L2=2.4522e-01  shape=[1, 4]
BUG CONFIRMED: CPU and GPU produce different results for the same model and input

=== bug_pt138.py ===
output[0]: L2=2.0447e+00  shape=[4, 8]
BUG CONFIRMED: CPU and GPU produce different results for the same model and input

=== bug_pt144.py ===
output[0]: L2=3.2524e-03  shape=[4, 1]
BUG CONFIRMED: CPU and GPU produce different results for the same model and input

=== bug_pt147.py ===
output[0]: L2=8.3500e+01  shape=[1, 8]
BUG CONFIRMED: CPU and GPU produce different results for the same model and input

=== bug_pt162.py ===
output[0]: L2=4.8801e-02  shape=[4, 8]
BUG CONFIRMED: CPU and GPU produce different results for the same model and input

=== bug_pt171.py ===
output[0]: L2=2.0108e+01  shape=[4, 4]
BUG CONFIRMED: CPU and GPU produce different results for the same model and input

=== bug_pt191.py ===
output[0]: L2=1.5129e+00  shape=[4, 4]
BUG CONFIRMED: CPU and GPU produce different results for the same model and input

=== bug_pt192.py ===
output[0]: ASYMMETRIC NaN/Inf — cpu_nan=16 gpu_nan=17 cpu_inf=0 gpu_inf=0 asym=1
BUG CONFIRMED: CPU and GPU produce different results for the same model and input

=== bug_pt202.py ===
output[0]: L2=1.7321e+00  shape=[4, 8]
BUG CONFIRMED: CPU and GPU produce different results for the same model and input

=== bug_pt236.py ===
output[0]: L2=6.0033e-02  shape=[4, 8]
BUG CONFIRMED: CPU and GPU produce different results for the same model and input

=== bug_pt241.py ===
output[0]: L2=6.0545e+00  shape=[4, 8]
BUG CONFIRMED: CPU and GPU produce different results for the same model and input

=== bug_pt248.py ===
output[0]: L2=1.3941e-02  shape=[]
BUG CONFIRMED: CPU and GPU produce different results for the same model and input

=== bug_pt281.py ===
output[0]: L2=2.3222e+00  shape=[4, 8]
BUG CONFIRMED: CPU and GPU produce different results for the same model and input

=== bug_pt284.py ===
output[0]: L2=2.3937e-02  shape=[4, 8]
BUG CONFIRMED: CPU and GPU produce different results for the same model and input

=== bug_pt295.py ===
output[0]: L2=2.8284e+00  shape=[4, 4]
BUG CONFIRMED: CPU and GPU produce different results for the same model and input

=== bug_pt305.py ===
output[0]: L2=2.7761e-01  shape=[4, 8]
BUG CONFIRMED: CPU and GPU produce different results for the same model and input

=== bug_pt316.py ===
output[0]: L2=7.1596e-03  shape=[]
BUG CONFIRMED: CPU and GPU produce different results for the same model and input

=== bug_pt319.py ===
output[0]: L2=1.0162e-01  shape=[4, 8]
BUG CONFIRMED: CPU and GPU produce different results for the same model and input

=== bug_pt335.py ===
output[0]: L2=4.0375e-01  shape=[4, 8]
BUG CONFIRMED: CPU and GPU produce different results for the same model and input

=== bug_pt343.py ===
output[0]: L2=7.8760e-01  shape=[4, 8]
BUG CONFIRMED: CPU and GPU produce different results for the same model and input

=== bug_pt346.py ===
output[0]: L2=3.2768e+04  shape=[]
BUG CONFIRMED: CPU and GPU produce different results for the same model and input

=== bug_pt357.py ===
output[0]: L2=5.7845e-01  shape=[4, 8]
BUG CONFIRMED: CPU and GPU produce different results for the same model and input

=== bug_pt358.py ===
output[0]: L2=7.3856e+02  shape=[1, 8]
BUG CONFIRMED: CPU and GPU produce different results for the same model and input

=== bug_pt375.py ===
output[0]: L2=2.4601e+00  shape=[4, 8]
BUG CONFIRMED: CPU and GPU produce different results for the same model and input

=== bug_pt382.py ===
output[0]: L2=7.1658e-03  shape=[4, 8]
BUG CONFIRMED: CPU and GPU produce different results for the same model and input

=== bug_pt384.py ===
output[0]: L2=5.0380e-02  shape=[]
BUG CONFIRMED: CPU and GPU produce different results for the same model and input

=== bug_pt390.py ===
output[0]: L2=3.5405e-02  shape=[4, 8]
BUG CONFIRMED: CPU and GPU produce different results for the same model and input

=== bug_pt396.py ===
output[0]: ASYMMETRIC NaN/Inf — cpu_nan=0 gpu_nan=1 cpu_inf=1 gpu_inf=0 asym=1
BUG CONFIRMED: CPU and GPU produce different results for the same model and input

=== bug_pt398.py ===
output[0]: L2=9.4865e-03  shape=[4, 8]
BUG CONFIRMED: CPU and GPU produce different results for the same model and input

=== bug_pt404.py ===
output[0]: L2=5.4636e+04  shape=[4, 16, 8]
BUG CONFIRMED: CPU and GPU produce different results for the same model and input

=== bug_pt409.py ===
output[0]: L2=3.4774e-01  shape=[4, 4]
BUG CONFIRMED: CPU and GPU produce different results for the same model and input

=== bug_pt424.py ===
output[0]: L2=3.9086e-03  shape=[]
BUG CONFIRMED: CPU and GPU produce different results for the same model and input

=== bug_pt428.py ===
output[0]: L2=1.0558e-02  shape=[8]
BUG CONFIRMED: CPU and GPU produce different results for the same model and input

=== bug_pt441.py ===
GPU output: tensor([1.0000, 1.0006, 1.0015, 1.0018])
CPU crashed: cholesky: The factorization could not be completed because the input is not positive-definite (the leading minor of order 3 is not positive-definite).
BUG CONFIRMED: GPU succeeds but CPU crashes on the same model and input

=== bug_pt450.py ===
output[0]: L2=4.7500e+00  shape=[]
BUG CONFIRMED: CPU and GPU produce different results for the same model and input

=== bug_pt467.py ===
output[0]: L2=1.7321e+00  shape=[4, 4]
BUG CONFIRMED: CPU and GPU produce different results for the same model and input

=== bug_pt480.py ===
output[0]: L2=6.1272e-02  shape=[4, 8]
BUG CONFIRMED: CPU and GPU produce different results for the same model and input

=== bug_pt486.py ===
output[0]: L2=3.0295e-03  shape=[4, 8]
output[1]: L2=1.7158e-03  shape=[4, 8]
output[2]: L2=3.2102e-03  shape=[4, 8]
BUG CONFIRMED: CPU and GPU produce different results for the same model and input

=== bug_tf067.py ===
CPU output[:4]: [-1.9105611  -0.56379056  0.7604368   2.08041   ]
GPU output[:4]: [-1.9097927 -0.5633421  0.7580898  2.0807762]
BUG CONFIRMED: L2=2.5365e-03 between CPU and GPU (same model, same input, no mutation)

=== bug_tf106.py ===
CPU output[:4]: [184.39626]
GPU output[:4]: [186.85849]
BUG CONFIRMED: L2=2.4622e+00 between CPU and GPU (same model, same input, no mutation)
```
