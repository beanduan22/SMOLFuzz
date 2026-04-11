# New Bugs — Confirmed CPU/GPU Divergence

3 real bugs found by searching PyTorch and TensorFlow issue trackers and reproducing on current versions.  
These are **not floating-point noise** — they are wrong algorithms or missing IEEE 754 handling.

Run with: `python3 <file>.py`  
Requires: PyTorch ≥ 2.1 with CUDA / TensorFlow ≥ 2.13 with GPU.

---

## pt_lstsq_rankdef.py — `torch.linalg.lstsq` wrong result on GPU for rank-deficient input

**Root cause:** CPU uses LAPACK `gelsd` (SVD-based, handles rank deficiency). GPU uses cuBLAS `gels` (QR-based, assumes full rank). For a rank-deficient matrix, GPU silently returns a completely wrong solution instead of raising an error or falling back to SVD.

```
NumPy (reference): [ 0.8333333   0.33333334 -0.16666667]
CPU:               [ 0.83333385  0.33333337 -0.1666665 ]
GPU:               [ 0.58141476  0.76307154 -0.        ]
CPU vs NumPy L2: 7.5424e-07
GPU vs NumPy L2: 1.2867e+00
```

Related: https://github.com/pytorch/pytorch/issues/88101

---

## pt_eigh_float32.py — `torch.linalg.eigh` GPU is 52× less accurate than CPU for float32

**Root cause:** CPU uses LAPACK `ssyevd` (Divide & Conquer). GPU uses cuSOLVER's float32 eigensolver, which for the same matrix produces eigenvalues that deviate from the float64 reference by 52× more than LAPACK does.

```
CPU eigenvalue error vs float64 reference: 8.2712e-05
GPU eigenvalue error vs float64 reference: 4.2929e-03
GPU is 52x less accurate than CPU
CPU vs GPU L2: 4.2694e-03

First 5 eigenvalues:
  CPU: [0.001586, 0.051802, 0.261722, 0.427910, 0.761837]
  GPU: [0.001603, 0.051811, 0.261725, 0.427923, 0.761844]
  Ref: [0.001605, 0.051807, 0.261724, 0.427918, 0.761839]
```

---

## tf_abs_complex_nan.py — `tf.math.abs` returns `nan` on GPU where CPU returns `inf` for complex128

**Root cause:** For `complex(inf, nan)` and `complex(nan, inf)`, IEEE 754 mandates that `|inf + nan*j| = inf` because inf dominates. CPU (C++ `hypot`) follows this rule. GPU CUDA kernel does not, returning `nan` instead.

```
abs((inf+nanj))
  CPU: inf
  GPU: nan   <-- BUG

abs((nan+infj))
  CPU: inf
  GPU: nan   <-- BUG
```

Related: https://github.com/tensorflow/tensorflow/issues/98410
