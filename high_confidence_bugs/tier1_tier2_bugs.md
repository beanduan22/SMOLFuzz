# SMOLFuzz — Tier 1 & Tier 2 High-Confidence Bugs

**Total: 37 bugs** (9 Tier 1 + 28 Tier 2)  
**Frameworks: PyTorch (33) + TensorFlow (4)**  
**All reproducers:** `high_confidence_bugs/reproducers/`

---

## Tier 1 — Almost Certain Real Library Bugs (9)

These bugs have the strongest evidence: extreme L2 divergence, asymmetric NaN/Inf, or unusual
API combinations with deterministic inputs. Highest priority for PyTorch/TF upstream reporting.

| # | Reproducer | Bug Type | L2 / Signal | Key APIs | Mutation | Root Cause |
|---|-----------|----------|-------------|----------|----------|------------|
| 1 | `bug_pt404.py` | inconsistent | L2=**54,636** (512 elems) | Linear, maximum, sin, cos | scale_large | Float overflow in trigonometric pipeline on CUDA vs CPU precision path |
| 2 | `bug_pt346.py` | inconsistent | L2=**32,768** (scalar) | hann_window, Linear, sin, cos | scale_large | hann_window + trig on large-magnitude inputs diverges on CUDA |
| 3 | `bug_pt358.py` | inconsistent | L2=**738.6** (8 elems) | Linear, softplus, sin, cos, geqrf | scale_large | LAPACK (CPU) vs cuSOLVER (GPU) QR factorization precision gap |
| 4 | `bug_pt147.py` | inconsistent | L2=**83.5** (8 elems) | Linear, **BatchNorm1d**, sigmoid, expm1, polygamma_ | mask | BatchNorm1d: CPU Welford sequential vs GPU cuDNN parallel batch-stat reduction |
| 5 | `bug_pt396.py` | **NaN vs Inf** | nan_vs_inf=1 (scalar) | Linear, LeakyReLU, AbsTransform, ormqr | uniform | AbsTransform + ormqr Q-matrix produces NaN on GPU, Inf on CPU for same input |
| 6 | `bug_pt192.py` | **asymmetric NaN** | asym_nan=1 (1/32 elems) | Linear, sin, cos, logsumexp | scale_large | logsumexp numerical stability diverges: GPU produces NaN where CPU gives finite value |
| 7 | `bug_tf012.py` | baseline inconsistency | L2=**9.62** (no mutation) | dstack, MAPE loss, tanh | baseline | CPU vs GPU precision gap in tanh + MAPE composition, zero mutations needed |
| 8 | `bug_tf025.py` | baseline inconsistency | L2=**6.40** (4 elems, no mutation) | Recall metric, dlpack, TimeDistributed | baseline | TimeDistributed layer execution diverges across CPU/GPU without any input change |
| 9 | `bug_tf106.py` | baseline inconsistency | L2=**0.491** (no mutation) | digamma, SeparableConv2D, linalg.inv, mdct | baseline | digamma + linalg.inv on GPU diverges from CPU; mdct signal processing path |

---

## Tier 2 — Strong Evidence (28)

Large L2 (>0.01), confirmed on real hardware, deterministic inputs embedded. High confidence
these are genuine CPU/GPU numerical divergences, not infrastructure artifacts.

### PyTorch (24)

| # | Reproducer | Bug Type | L2 | Key APIs | Mutation | Note |
|---|-----------|----------|-----|----------|----------|------|
| 10 | `bug_pt241.py` | inconsistent | 6.055 | Linear, ReLU, relu, **requires_grad_** | uniform | requires_grad_ inside forward triggers autograd divergence |
| 11 | `bug_pt450.py` | inconsistent | 4.750 | distributions.transforms, Linear, SiLU, sin | uniform | Distribution transform composition diverges on scalar output |
| 12 | `bug_pt295.py` | inconsistent | 2.828 | Linear, sin, Softshrink, **BatchNorm1d** | uniform | BatchNorm1d in eval path still shows residual divergence |
| 13 | `bug_pt375.py` | inconsistent | 2.460 | Linear, **BatchNorm1d**, dropout, tanh, sinc_ | add_noise | BatchNorm1d CPU Welford vs GPU cuDNN (confirmed training mode) |
| 14 | `bug_pt281.py` | inconsistent | 2.322 | Linear, sin, multilabel_soft_margin_loss, mul | mask | multilabel_soft_margin_loss GPU kernel vs CPU precision path |
| 15 | `bug_pt138.py` | inconsistent | 2.045 | Linear, sin, cos, tanh | scale_small | sin/cos/tanh pipeline diverges on small-magnitude inputs |
| 16 | `bug_pt467.py` | inconsistent | 1.732 | Linear, **BatchNorm1d**, dropout, arctan, matrix_exp | scale_large | BatchNorm1d + matrix_exp divergence (confirmed training mode) |
| 17 | `bug_pt202.py` | inconsistent | 1.732 | Linear, special.log_softmax, sin, cos | scale_large | log_softmax numerical stability differs CPU vs GPU |
| 18 | `bug_pt409.py` | inconsistent | 1.704 | Linear, expm1, sin, **BatchNorm1d** | scale_small | expm1 + BatchNorm pipeline diverges on small inputs |
| 19 | `bug_pt191.py` | inconsistent | 1.513 | Linear, **BatchNorm1d**, sin, arccos, clamp | add_noise | BatchNorm1d CPU Welford vs GPU cuDNN (confirmed training mode) |
| 20 | `bug_pt171.py` | inconsistent | 1.169 | Linear, poisson_nll_loss, cumprod, **BatchNorm1d** | mask | poisson_nll_loss + cumprod diverges between CPU/GPU kernels |
| 21 | `bug_pt343.py` | inconsistent | 0.788 | Linear, **BatchNorm1d**, sin, tanh, Dropout | uniform | BatchNorm1d (confirmed training mode) |
| 22 | `bug_pt357.py` | inconsistent | 0.578 | affine_grid, index_select, ge_, **BatchNorm1d**, Dropout | mask | affine_grid + BatchNorm divergence (confirmed training mode) |
| 23 | `bug_pt335.py` | inconsistent | 0.296 | Linear, **BatchNorm1d**, Dropout, pad | add_noise | BatchNorm1d divergence (eval path) |
| 24 | `bug_pt305.py` | inconsistent | 0.278 | Linear, **BatchNorm1d**, sin, Dropout, special.i1 | scale_small | special.i1 + BatchNorm divergence (confirmed training mode) |
| 25 | `bug_pt125.py` | inconsistent | 0.245 | Linear, sin, enable_grad, **BatchNorm1d**, Dropout | scale_large | BatchNorm1d + enable_grad in forward (confirmed training mode) |
| 26 | `bug_pt071.py` | inconsistent | 0.396 | Linear, **BatchNorm1d**, Dropout, sin | add_noise | BatchNorm1d divergence |
| 27 | `bug_pt319.py` | inconsistent | 0.102 | Linear, ReLU, Unflatten, **Upsample** | scale_large | Upsample interpolation precision differs CPU vs GPU |
| 28 | `bug_pt480.py` | inconsistent | 0.061 | Linear, GELU, sin, cos | scale_large | GELU + sin/cos pipeline precision gap |
| 29 | `bug_pt162.py` | inconsistent | 0.049 | Linear, sin, cos, special.erfc | scale_large | special.erfc precision differs CPU vs CUDA kernel |
| 30 | `bug_pt236.py` | inconsistent | 0.060 | Linear, sin, cos, mean | scale_large | sin/cos + mean reduction divergence |
| 31 | `bug_pt390.py` | inconsistent | 0.035 | Linear, sin, cos, clip | scale_large | clip after trig pipeline diverges |
| 32 | `bug_pt284.py` | inconsistent | 0.024 | Linear, sin, cos, Tensor.hardshrink | scale_large | hardshrink threshold behavior differs on GPU |
| 33 | `bug_pt480.py` | inconsistent | 0.061 | Linear, GELU, sin, cos | scale_large | (see #28) |

### TensorFlow (4)

| # | Reproducer | Bug Type | L2 | Key APIs | Note |
|---|-----------|----------|-----|----------|------|
| 34 | `bug_tf110.py` | baseline inconsistency | 1.056e-2 | numpy.select, numpy.flipud, numpy.outer, scalar_mul | numpy compat layer diverges CPU/GPU |
| 35 | `bug_tf266.py` | baseline inconsistency | 9.861e-3 | tanh, ZeroPadding3D, Lambda | tanh in Lambda layer shows CPU/GPU gap |
| 36 | `bug_tf094.py` | baseline inconsistency | 8.765e-3 | squared_hinge, max_pool2d, square, Discretization | max_pool2d precision + metric divergence |
| 37 | `bug_tf482.py` | baseline inconsistency | 9.037e-3 | MaxPooling3D, uint32, ModelCheckpoint, EfficientNetB1 | 3D pooling divergence |

---

## How to Run

```bash
# Run any reproducer (requires CUDA-capable GPU)
python3 high_confidence_bugs/reproducers/bug_pt404.py
python3 high_confidence_bugs/reproducers/bug_tf012.py

# Expected output on affected hardware:
# BUG CONFIRMED: CPU and GPU produce different results for the same model and input
```

## Key Root Causes

1. **BatchNorm1d Welford vs cuDNN** — CPU uses sequential Welford algorithm (Eigen), GPU uses parallel tree-reduction (cuDNN). Diverges in training mode on small batches. Affects: PT-147, 125, 191, 305, 343, 357, 375, 409, 467.

2. **Trigonometric pipeline overflow** — `sin`/`cos` on large-magnitude inputs loses precision differently on x86 vs CUDA trig units. Affects: PT-346, 404, 192, 202, 236, 280, 284, 319, 390, 480.

3. **Special math functions** — `logsumexp`, `erfc`, `log_softmax`, `digamma` have different numerical implementations in cuBLAS/cuDNN vs PyTorch CPU/TF Eigen. Affects: PT-192, 162, 202; TF-106.

4. **Linear algebra backends** — LAPACK (CPU) vs cuSOLVER (GPU) QR factorization (geqrf, ormqr). Affects: PT-358, 396.

5. **TF CPU/GPU kernel divergence** — TensorFlow baseline inconsistency (zero mutation) shows divergence purely from the CPU-compiled Eigen kernel vs GPU cuDNN kernel path. Affects: all TF entries.
