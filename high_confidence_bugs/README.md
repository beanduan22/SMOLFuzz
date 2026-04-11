# High-Confidence Bugs — Minimal Reproducers

8 genuine CPU/GPU divergence bugs found by SMOLFuzz (6 PyTorch + 2 TensorFlow).  
All bugs are caused by platform-specific differences in numeric algorithms, not by floating-point sensitivity of trigonometric functions on large inputs.

Each script in `minimal/` is self-contained. Run with:

```
python3 minimal/<file>.py
```

Requires: PyTorch ≥ 2.1 with CUDA / TensorFlow ≥ 2.13 with GPU.

---

## PyTorch

### pt147.py — BatchNorm1d variance algorithm divergence
**Root cause:** CPU uses Welford's online algorithm; GPU uses cuDNN. For certain inputs the accumulated variance differs.

```
CPU: tensor([[3.8083e+00, 2.7262e+00, 2.3084e+02, 3.7534e+00, 1.7628e+01, 1.2490e+00,
         1.6753e+00, 6.2550e+05]])
GPU: tensor([[3.8083e+00, 2.7262e+00, 2.3084e+02, 3.7534e+00, 1.7628e+01, 1.2490e+00,
         1.6753e+00, 6.2542e+05]])
L2: 8.3500e+01
```

---

### pt192.py — Asymmetric NaN in logsumexp
**Root cause:** CPU and GPU disagree on whether one boundary value should be NaN. CPU produces 16 NaNs, GPU produces 17.

```
CPU nan count: 16
GPU nan count: 17
Asymmetric NaN positions: 1
```

---

### pt295.py — BatchNorm1d applied twice in training mode
**Root cause:** Applying BatchNorm twice in the same forward pass (once with train stats, once with updated running stats) exposes a divergence between CPU (Welford) and GPU (cuDNN) normalization paths. CPU produces near-zero residuals (~4e-12); GPU rounds to exactly zero.

```
CPU: tensor([[4.2480e-12, 1.0000e+00, 1.8019e-12, 1.0000e+00],
        [4.2480e-12, 1.0000e+00, 1.8019e-12, 1.0000e+00],
        [4.2480e-12, 1.0000e+00, 1.8019e-12, 1.0000e+00],
        [4.2480e-12, 1.0000e+00, 1.8019e-12, 1.0000e+00]])
GPU: tensor([[0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.]])
L2: 2.8284e+00
```

---

### pt396.py — logdet: CPU returns -inf, GPU returns finite
**Root cause:** Uniform input creates a rank-1 matrix. LAPACK (CPU) returns `-inf` for its determinant; cuSOLVER (GPU) returns a finite value (`-108.34`).

```
CPU logdet: -inf
GPU logdet: -108.34196472167969
CPU is inf: True GPU is nan: False
```

---

### pt441.py — CPU crashes, GPU succeeds (Cholesky)
**Root cause:** Small inputs produce a correlation matrix that is not positive-definite. LAPACK on CPU raises a `RuntimeError`; cuSOLVER on GPU silently returns a result.

```
GPU: tensor([1.0000, 1.0006, 1.0015, 1.0018])
CPU crash: cholesky: The factorization could not be completed because the input is not positive-definite (the leading minor of order 3 is not positive-definite).
```

---

### pt450.py — cholesky_inverse divergence
**Root cause:** `torch.outer(w[0], w[0])` produces a rank-1 matrix. `cholesky_inverse` on a singular/near-singular matrix diverges between LAPACK (CPU) and cuSOLVER (GPU).

```
CPU: 3377650.0
GPU: 3377654.75
L2: 4.7500e+00
```

---

## TensorFlow

> These bugs are hardware-dependent; they were confirmed on the original test hardware.

### tf067.py — BatchNormalization + DCT divergence
**Root cause:** CPU and GPU BatchNormalization accumulate running statistics differently under a `min_max_norm` kernel constraint, causing downstream DCT outputs to diverge.

```
CPU: [-0.22901464  0.45488286 -2.4911814  -0.22179246]
GPU: [-0.22896935  0.45494384 -2.4912076  -0.22175983]
L2: 1.9166e-03
```

---

### tf106.py — BatchNormalization + digamma(float64) divergence
**Root cause:** BatchNormalization running statistics diverge between CPU and GPU. After casting to float64 and applying digamma, the difference amplifies.

```
CPU: 814.2219...
GPU: 814.7134...
L2: 4.9146e-01
```
