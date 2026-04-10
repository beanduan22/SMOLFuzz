# SMOLFuzz — Tier 1 & Tier 2 Confirmed Bugs

**Total: 44 bugs** (10 Tier 1 + 34 Tier 2)  
**Frameworks: PyTorch (42) + TensorFlow (2)**  
**Verified:** 2026-04-10 — all bugs confirmed stable across 5 consecutive runs  
**All reproducers:** `high_confidence_bugs/reproducers/`  
**Inputs:** Embedded as literals in every reproducer — no external files needed

**Tier 1:** crash / asymmetric NaN-Inf / L2 > 5.0 (PT) / any confirmed TF baseline  
**Tier 2:** all other confirmed reproducible bugs (L2 > 1e-3, stable across runs)

---

## Tier 1 — Strongest Unambiguous Bugs (10)

| # | Reproducer | Bug Type | Confirmed Signal | Key APIs | Mutation |
|---|---|---|---|---|---|
| 1 | `bug_pt404.py` | inconsistent | **L2=5.4636e+04** (512 elems) | Linear, maximum, sin, cos | scale_large |
| 2 | `bug_pt346.py` | inconsistent | **L2=3.2768e+04** (scalar) | hann_window, Linear, sin, cos | scale_large |
| 3 | `bug_pt358.py` | inconsistent | **L2=7.3856e+02** (8 elems) | Linear, softplus, sin, cos, geqrf | scale_large |
| 4 | `bug_pt147.py` | inconsistent | **L2=8.3500e+01** (8 elems) | Linear, **BatchNorm1d**, sigmoid, expm1, polygamma_ | mask |
| 5 | `bug_pt171.py` | inconsistent | **L2=2.0108e+01** (16 elems) | Linear, poisson_nll_loss, cumprod, BatchNorm1d | mask |
| 6 | `bug_pt241.py` | inconsistent | **L2=6.0545e+00** (32 elems) | Linear, ReLU, relu, requires_grad_ | uniform |
| 7 | `bug_pt192.py` | **NaN** | **ASYM NaN: cpu=16, gpu=17, asym=1** | Linear, sin, cos, logsumexp | scale_large |
| 8 | `bug_pt396.py` | **NaN** | **cpu=Inf / gpu=NaN (scalar)** | Linear, LeakyReLU, AbsTransform, ormqr | uniform |
| 9 | `bug_pt441.py` | **crash** | **CPU crashes / GPU ok** | Linear, logcumsumexp, corrcoef, cholesky | scale_small |
| 10 | `bug_tf106.py` | baseline incon. | **L2=1.4250e+00** (no mutation) | digamma, SeparableConv2D, linalg.inv, mdct | baseline |

---

## Tier 2 — All Other Confirmed Bugs (34)

### PyTorch — 33 bugs

| # | Reproducer | Confirmed L2 | Key APIs | Mutation |
|---|---|---|---|---|
| 11 | `bug_pt450.py` | L2=4.7500e+00 | distributions.transforms, Linear, SiLU, sin | uniform |
| 12 | `bug_pt295.py` | L2=2.8284e+00 | Linear, sin, Softshrink, BatchNorm1d | uniform |
| 13 | `bug_pt375.py` | L2=2.4601e+00 | Linear, **BatchNorm1d**, dropout, tanh, sinc_ | add_noise |
| 14 | `bug_pt281.py` | L2=2.3222e+00 | Linear, sin, multilabel_soft_margin_loss, mul | mask |
| 15 | `bug_pt138.py` | L2=2.0447e+00 | Linear, sin, cos, tanh | scale_small |
| 16 | `bug_pt467.py` | L2=1.7321e+00 | Linear, **BatchNorm1d**, dropout, arctan, matrix_exp | scale_large |
| 17 | `bug_pt202.py` | L2=1.7321e+00 | Linear, special.log_softmax, sin, cos | scale_large |
| 18 | `bug_pt191.py` | L2=1.5129e+00 | Linear, **BatchNorm1d**, sin, arccos, clamp | add_noise |
| 19 | `bug_pt343.py` | L2=7.8760e-01 | Linear, **BatchNorm1d**, sin, tanh, Dropout | uniform |
| 20 | `bug_pt357.py` | L2=5.7845e-01 | affine_grid, index_select, ge_, **BatchNorm1d**, Dropout | mask |
| 21 | `bug_pt071.py` | L2=4.4361e-01 | Linear, **BatchNorm1d**, Dropout, sin | add_noise |
| 22 | `bug_pt335.py` | L2=4.0375e-01 | Linear, **BatchNorm1d**, Dropout, pad | add_noise |
| 23 | `bug_pt409.py` | L2=3.4774e-01 | Linear, expm1, sin, BatchNorm1d | scale_small |
| 24 | `bug_pt305.py` | L2=2.7761e-01 | Linear, **BatchNorm1d**, sin, Dropout, special.i1 | scale_small |
| 25 | `bug_pt125.py` | L2=2.4522e-01 | Linear, sin, enable_grad, **BatchNorm1d**, Dropout | scale_large |
| 26 | `bug_pt319.py` | L2=1.0162e-01 | Linear, ReLU, Unflatten, **Upsample** | scale_large |
| 27 | `bug_pt106.py` | L2=9.0167e-02 | Linear, sin, cos, hardshrink | scale_large |
| 28 | `bug_pt480.py` | L2=6.1272e-02 | Linear, GELU, sin, cos | scale_large |
| 29 | `bug_pt236.py` | L2=6.0033e-02 | Linear, sin, cos, mean | scale_large |
| 30 | `bug_pt384.py` | L2=5.0380e-02 | Linear, stft, sin, cos | scale_large |
| 31 | `bug_pt162.py` | L2=4.8801e-02 | Linear, sin, cos, special.erfc | scale_large |
| 32 | `bug_pt390.py` | L2=3.5405e-02 | Linear, sin, cos, clip | scale_large |
| 33 | `bug_pt284.py` | L2=2.3937e-02 | Linear, sin, cos, Tensor.hardshrink | scale_large |
| 34 | `bug_pt248.py` | L2=1.3941e-02 | Linear, sin, cos, special.xlogy | scale_large |
| 35 | `bug_pt101.py` | L2=1.1677e-02 | Linear, BatchNorm1d, sin, cos | scale_large |
| 36 | `bug_pt428.py` | L2=1.0558e-02 | Linear, sin, cos, nn.GLU | scale_large |
| 37 | `bug_pt398.py` | L2=9.4865e-03 | Linear, sin, cos, special.gammaln | scale_large |
| 38 | `bug_pt382.py` | L2=7.1658e-03 | Linear, sin, cos, nn.Hardswish | scale_large |
| 39 | `bug_pt316.py` | L2=7.1596e-03 | Linear, sin, cos, lgamma | scale_large |
| 40 | `bug_pt080.py` | L2=5.9273e-03 | Linear, WeightNorm, BatchNorm1d | uniform |
| 41 | `bug_pt424.py` | L2=3.9086e-03 | Linear, sin, cos, nn.Mish | scale_large |
| 42 | `bug_pt144.py` | L2=3.2524e-03 | Linear, BatchNorm1d, sin, cos | uniform |
| 43 | `bug_pt486.py` | L2=3.0295e-03 | Linear, sin, cos (multi-output) | scale_large |

### TensorFlow — 1 bug

| # | Reproducer | Confirmed L2 | Key APIs |
|---|---|---|---|
| 44 | `bug_tf067.py` | L2=6.2642e-03 | linalg.diag, LocallyConnected1D, math.angle |

---

## Root Cause Summary

| Root Cause | Count | Bugs |
|---|---|---|
| BatchNorm1d: CPU Welford vs GPU cuDNN reduction | 16 | pt071, pt080, pt101, pt125, pt144, pt147, pt171, pt191, pt295, pt305, pt335, pt343, pt357, pt375, pt409, pt467 |
| Trigonometric pipeline (sin/cos on large inputs) | 18 | pt106, pt138, pt162, pt202, pt236, pt248, pt284, pt316, pt346, pt382, pt384, pt390, pt398, pt404, pt424, pt428, pt480, pt486 |
| Special math (logsumexp, erfc, lgamma, xlogy…) | 5 | pt192, pt241, pt281, pt319, pt450 |
| Linear algebra backends (LAPACK vs cuSOLVER) | 3 | pt358, pt396, pt441 |
| TF CPU Eigen vs GPU cuDNN kernel divergence | 2 | tf067, tf106 |

## How to Run

```bash
# Single bug
python3 high_confidence_bugs/reproducers/bug_pt404.py

# All bugs (should all print BUG CONFIRMED)
for f in high_confidence_bugs/reproducers/*.py; do
    echo -n "$(basename $f): "
    python3 "$f" 2>/dev/null | grep -E "BUG CONFIRMED|not reproduced" | head -1
done
```
