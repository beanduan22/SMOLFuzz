# New Bugs — Confirmed CPU/GPU Divergence

15 real bugs confirmed across PyTorch and TensorFlow.
Sources: PyTorch/TensorFlow issue trackers + cross-verified from JAX (#22557), PaddlePaddle (#72779), and MXNet (#9555).
These are **not floating-point noise** — they are wrong algorithms, undefined-behavior exploitation, or missing IEEE 754 handling.

Run with: `python3 <file>.py`
Requires: PyTorch ≥ 2.1 with CUDA / TensorFlow ≥ 2.13 with GPU.

---

## PyTorch Bugs

### pt_lstsq_rankdef.py — `torch.linalg.lstsq` wrong result on GPU for rank-deficient input

**Root cause:** CPU uses LAPACK `gelsd` (SVD-based, handles rank deficiency). GPU uses cuBLAS `gels` (QR-based, assumes full rank). For a rank-deficient matrix, GPU silently returns a completely wrong solution.

```
NumPy (reference): [ 0.8333333   0.33333334 -0.16666667]
CPU:               [ 0.83333385  0.33333337 -0.1666665 ]
GPU:               [ 0.58141476  0.76307154 -0.        ]
CPU vs NumPy L2: 7.5424e-07
GPU vs NumPy L2: 1.2867e+00
```

Related: https://github.com/pytorch/pytorch/issues/88101

---

### pt_eigh_float32.py — `torch.linalg.eigh` GPU is 52× less accurate than CPU for float32

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

### pt_std_overflow.py — `torch.std` returns `inf` on GPU for large float32 values

**Root cause:** GPU variance kernel computes `sum(x²) − n·mean²` in a single pass. For values ~1e20, `x²` overflows float32 to `inf`. CPU uses a two-pass or compensated algorithm that avoids this overflow.

```
Reference (float64): 1.0287e+19
CPU (float32):       1.0287e+19
GPU (float32):       inf   <-- BUG
```

---

### pt_svdvals_accuracy.py — `torch.linalg.svdvals` GPU is 81× less accurate than CPU

**Root cause:** CPU uses LAPACK `dgesdd` (internally float64). GPU uses cuSOLVER float32 SVD, accumulating significantly more rounding error across 200 singular values.

```
CPU singular value error vs float64 reference: 5.8592e-05
GPU singular value error vs float64 reference: 4.7265e-03
GPU is 81x less accurate than CPU
CPU vs GPU L2: 4.7092e-03

First 5 singular values:
  CPU: [28.238087, 27.561943, 26.981163, 26.831314, 26.499115]
  GPU: [28.238829, 27.562660, 26.981678, 26.831942, 26.499847]
  Ref: [28.238094, 27.561947, 26.981161, 26.831326, 26.499111]
```

---

### pt_matrix_norm_nuc.py — `torch.linalg.matrix_norm` nuclear norm GPU is 224× less accurate

**Root cause:** Nuclear norm = sum of singular values. GPU uses cuSOLVER float32 SVD; accumulated singular value errors make the nuclear norm significantly less accurate than CPU (LAPACK, internally float64).

```
Reference (NumPy):   2400.332520
CPU (float32):       2400.332764
GPU (float32):       2400.387207
CPU error vs NumPy: 2.4414e-04
GPU error vs NumPy: 5.4688e-02
GPU is 224x less accurate than CPU
```

---

### pt_cumsum_f16.py — `torch.cumsum` GPU is 5× less accurate than CPU for float16

**Root cause:** CPU promotes float16 to float32 during accumulation. GPU performs cumulative sum natively in float16, accumulating 10,000 half-precision additions without compensation, causing significant numerical drift.

```
CPU error vs float64 reference: 2.0503e+00
GPU error vs float64 reference: 1.0641e+01
GPU is 5.2x less accurate than CPU
CPU vs GPU L2: 1.0917e+01
```

---

## TensorFlow Bugs

### tf_abs_complex_nan.py — `tf.math.abs` returns `nan` on GPU for complex128 inputs

**Root cause:** For `complex(inf, nan)` and `complex(nan, inf)`, IEEE 754 mandates that `|inf + nan·j| = inf`. CPU (C++ `hypot`) follows this rule. GPU CUDA kernel does not, returning `nan` instead.

```
abs((inf+nanj))
  CPU: inf
  GPU: nan   <-- BUG

abs((nan+infj))
  CPU: inf
  GPU: nan   <-- BUG
```

Related: https://github.com/tensorflow/tensorflow/issues/98410

---

### tf_abs_complex64.py — `tf.math.abs` returns `nan` on GPU for complex64 inputs

**Root cause:** Same IEEE 754 violation as the complex128 bug above, but affecting the complex64 (single-precision) CUDA kernel. Both dtypes fail to propagate `inf` correctly when the other component is `nan`.

```
abs((inf+nanj))
  CPU: inf
  GPU: nan   <-- BUG

abs((nan+infj))
  CPU: inf
  GPU: nan   <-- BUG
```

---

### tf_eigh_float32.py — `tf.linalg.eigh` CPU is 23× less accurate than GPU for float32

**Root cause:** TF's CPU eigensolver calls LAPACK `ssyevd` in single precision. TF's GPU path uses cuSOLVER which internally promotes to float64, producing eigenvalues much closer to the float64 reference. This is the **opposite** of PyTorch's eigh bug.

```
CPU eigenvalue error vs float64 reference: 1.1751e-03
GPU eigenvalue error vs float64 reference: 5.2058e-05
CPU is 23x less accurate than GPU
CPU vs GPU L2: 1.1689e-03

First 5 eigenvalues:
  CPU: [0.002436, 0.119596, 0.439590, 0.589958, 0.833545]
  GPU: [0.002414, 0.119597, 0.439585, 0.589948, 0.833544]
  Ref: [0.002431, 0.119597, 0.439587, 0.589948, 0.833544]
```

---

### tf_svd_accuracy.py — `tf.linalg.svd` CPU is 17× less accurate than GPU for float32

**Root cause:** TF's CPU SVD calls LAPACK `sgesdd` in single precision. TF's GPU cuSOLVER path internally upconverts to double precision, producing singular values much closer to the float64 reference.

```
CPU singular value error vs float64 reference: 1.5531e-04
GPU singular value error vs float64 reference: 8.8921e-06
CPU is 17x less accurate than GPU
CPU vs GPU L2: 1.5351e-04

First 5 singular values:
  CPU: [28.081991, 27.322769, 27.039207, 26.456455, 26.270422]
  GPU: [28.081966, 27.322744, 27.039177, 26.456444, 26.270397]
  Ref: [28.081966, 27.322744, 27.039177, 26.456444, 26.270397]
```

---

### tf_cumsum_f16.py — `tf.math.cumsum` CPU is 12× less accurate than GPU for float16

**Root cause:** TF's CPU cumsum kernel operates natively in float16 without internal precision promotion. TF's GPU CUDA kernel promotes to float32 for accumulation, greatly reducing rounding error. This is the **opposite** of PyTorch's cumsum float16 bug.

```
CPU error vs float64 reference: 1.0686e+02
GPU error vs float64 reference: 9.1100e+00
CPU is 11.7x less accurate than GPU
CPU vs GPU L2: 1.0696e+02
```

---

## Cross-Framework Bugs (found in JAX/PaddlePaddle/MXNet, verified in PyTorch/TensorFlow)

### pt_cast_nan_inf.py — `float32(nan/inf)` → `int32/int64` CPU vs GPU different results

**Origin:** PaddlePaddle issue #72779 (float32(NaN)->int32 gives CPU=INT32_MIN, GPU=0). Same bug confirmed in PyTorch.

**Root cause:** C and CUDA standards both leave float-to-int conversion for NaN/Inf as **undefined behavior**. x86 `CVTTSS2SI` saturates to `INT_MIN` for any out-of-range value. CUDA's float-to-int returns 0 for NaN and `INT_MAX` for +Inf — opposite saturation semantics.

```
float32(nan)  -> int32:  CPU=-2147483648  GPU=0               -> BUG
float32(inf)  -> int32:  CPU=-2147483648  GPU=2147483647       -> BUG
float32(inf)  -> int64:  CPU=-9223372036854775808  GPU=9223372036854775807  -> BUG
```

---

### tf_cast_nan_inf.py — Same bug in TensorFlow

**Origin:** Same PaddlePaddle #72779. Confirmed in TF.

```
float32(nan)  -> int32:  CPU=-2147483648  GPU=0               -> BUG
float32(inf)  -> int32:  CPU=-2147483648  GPU=2147483647       -> BUG
float32(inf)  -> int64:  CPU=-9223372036854775808  GPU=9223372036854775807  -> BUG
```

---

### pt_matmul_tf32.py — `torch.matmul` with TF32 is 1208× less accurate than CPU

**Origin:** JAX issue #22557 (jnp.einsum GPU 70x less accurate due to reduced matmul precision). PyTorch has the same bug with a larger gap.

**Root cause:** NVIDIA Ampere+ GPUs support TF32 which reduces float32 mantissa from 23 bits to 10 bits in matrix multiplications. `torch.backends.cuda.matmul.allow_tf32 = True` is the **default** on Ampere+. CPU always uses full float32, creating a silent 1208x accuracy gap.

```
CPU  error vs float64 reference: 2.8160e-03
GPU  error (TF32=off):           2.4148e-03
GPU  error (TF32=on):            3.4008e+00   <-- BUG
TF32 makes GPU 1208x less accurate than CPU

First 4 output values:
  ref:      [34.167099, -0.626520, 2.158956, -7.296951]
  cpu:      [34.167099, -0.626529, 2.158953, -7.296947]
  gpu_tf32: [34.161766, -0.628163, 2.151218, -7.299671]
```

---

### tf_matmul_tf32.py — `tf.matmul` GPU is 1011× less accurate than CPU

**Origin:** Same JAX issue #22557. TF uses cuBLAS TF32 on Ampere+ with no user-accessible flag to disable it.

```
CPU error vs float64 reference: 3.3549e-03
GPU error vs float64 reference: 3.3903e+00   <-- BUG
GPU is 1011x less accurate than CPU (TF32 TensorCore)

First 4 output values:
  ref: [8.810464, 19.309502, -2.176206, -2.222901]
  cpu: [8.810467, 19.309511, -2.176223, -2.222898]
  gpu: [8.810652, 19.299034, -2.163798, -2.212879]
```
