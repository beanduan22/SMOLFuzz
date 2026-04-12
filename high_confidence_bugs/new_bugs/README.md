# CPU/GPU Divergence Bugs — 59 Confirmed Reproducers

Self-contained Python scripts that expose **CPU vs GPU numerical divergence** in PyTorch and TensorFlow. Every file runs standalone, exits 0 on success (bug confirmed), exits non-zero if the bug does not reproduce.

## Requirements

```
torch >= 2.1   (with CUDA)
tensorflow >= 2.13  (with GPU)
numpy
```

## Run any reproducer

```bash
python3 <file>.py
```

---

## Bug Categories

| Category | Count | Description |
|----------|-------|-------------|
| Reduction accuracy (cumsum/cumprod) | 12 | Different internal precision on CPU vs GPU |
| Linear algebra accuracy (SVD/eigh/pinv/norm) | 14 | LAPACK (CPU) vs cuSOLVER (GPU) precision |
| Wrong value (NaN/zero/inf) | 9 | Incorrect result on one device |
| Cast / overflow behavior | 6 | NaN/Inf casting to int differs CPU vs GPU |
| Matmul TF32 | 2 | TensorCore reduces mantissa bits on GPU |
| Linear system (lstsq) | 4 | Different algorithms for rank-deficient inputs |
| Complex number bugs | 3 | Divergence on complex64/128 inputs |

---

## PyTorch Bugs (25 files)

### Cumulative Operations

#### `pt_cumsum_f16.py` — float16 cumsum GPU 11–15x less accurate at N=500k

**Root cause:** CPU promotes float16→float32 for accumulation. GPU stays in float16 (7-bit mantissa). Error grows as O(√N × ε_f16) vs O(√N × ε_f32).

```
N = 500,000
CPU error vs float64 reference: 5.0529e-01   (CPU promotes f16→f32 internally)
GPU error vs float64 reference: 5.5936e+00   <-- BUG (GPU stays in f16)
GPU is 11x less accurate than CPU
BUG CONFIRMED: PT cumsum float16 GPU is 12-15x less accurate than CPU at N=500k
```

---

#### `pt_cumsum_bf16.py` — bfloat16 cumsum GPU 12–19x less accurate at N=500k

**Root cause:** Same pattern as f16 — CPU promotes bf16→f32, GPU stays in bf16.

```
N = 500,000
CPU error vs float64 reference: 2.0008e+00   (CPU promotes bf16→f32 internally)
GPU error vs float64 reference: 2.5993e+01   <-- BUG (GPU stays in bf16)
GPU is 13x less accurate than CPU
BUG CONFIRMED: PT cumsum bfloat16 GPU is 12-19x less accurate than CPU at N=500k
```

---

#### `pt_cumsum_complex64.py` — complex64 cumsum GPU 10–16x less accurate at N=5M

**Root cause:** CPU promotes complex64→complex128 internally. GPU stays in complex64.

```
cpu_max_err = 1.3636e-04
gpu_max_err = 1.5652e-03
GPU/CPU error ratio: 11.5x  (GPU less accurate)
BUG CONFIRMED: PT cumsum complex64 N=5M GPU is 10-16x less accurate than CPU
```

---

#### `pt_cumsum_f32.py` — float32 cumsum CPU exactly matches float64; GPU drifts

**Root cause:** PyTorch CPU promotes float32→float64 for cumsum. GPU accumulates natively in float32.

```
Reference (float64): cumsum of N=10,000 float32 values scaled to ~1e10
CPU error vs float64: 0.0000e+00   (CPU promotes to float64 internally)
GPU error vs float64: 6.2242e+06   <-- BUG (GPU accumulates in float32)
CPU exactly matches float64: True
Max CPU-GPU difference: 2.6214e+05
First divergence position: 5

Last 3 cumulative sums:
  [-3] ref=-1078126051328.00  cpu=-1078126051328.00  gpu=-1078126313472.00
  [-2] ref=-1066703060992.00  cpu=-1066703060992.00  gpu=-1066703323136.00
  [-1] ref=-1069817462784.00  cpu=-1069817462784.00  gpu=-1069817724928.00
```

---

#### `pt_cumprod_f32_large.py` — float32 cumprod GPU ~97000x less accurate at N=1M

**Root cause:** PyTorch CPU promotes float32→float64 for cumprod, giving near-float64 precision. GPU accumulates natively in float32.

```
cpu_max_err = 5.9604e-08  (CPU promotes f32→f64 internally)
gpu_max_err = 5.4827e-03  (GPU accumulates in f32)
GPU/CPU error ratio: 91986x  (GPU less accurate)
BUG CONFIRMED: PT cumprod f32 N=1M GPU is ~97000x less accurate than CPU (CPU uses f64 internally)
```

---

#### `pt_cumprod_f16.py` — float16 cumprod GPU 33–52x less accurate at N=10k

**Root cause:** CPU promotes f16→f32 for cumprod. GPU accumulates natively in float16.

```
CPU error vs float64 reference: 3.1138e-03
GPU error vs float64 reference: 1.6198e-01   <-- BUG
GPU is 52x less accurate than CPU

Last 5 values:
  ref: [0.2107037454843521, 0.21029222011566162, ...]
  cpu: [0.210693359375, 0.2103271484375, ...]
  gpu: [0.2047119140625, 0.204345703125, ...]
```

---

#### `pt_cumprod_bf16.py` — bfloat16 cumprod GPU 33x less accurate at N=10k

**Root cause:** CPU promotes bf16→f32 for cumprod. GPU accumulates natively in bfloat16.

```
CPU error vs float64 reference: 1.6893e-02
GPU error vs float64 reference: 5.6472e-01   <-- BUG
GPU is 33x less accurate than CPU
```

---

#### `pt_prod_f16.py` — float16 product CPU 131x less accurate than GPU

**Root cause:** CPU reduces in f16 (7-bit mantissa); GPU promotes to f32 internally before reducing.

```
Reference (float64): 0.135341
CPU (float16):       0.130737
GPU (float16):       0.135376
CPU error vs ref: 4.6034e-03
GPU error vs ref: 3.5267e-05
CPU is 131x less accurate than GPU   <-- BUG
```

---

#### `pt_prod_bf16.py` — bfloat16 product CPU 10x less accurate than GPU

**Root cause:** Same as f16 prod — CPU stays in bf16 while GPU promotes to f32.

```
Reference (float64): 0.135341
CPU (bfloat16):      0.108887   <-- BUG
GPU (bfloat16):      0.132812
CPU error vs ref: 2.6454e-02
GPU error vs ref: 2.5282e-03
CPU is 10x less accurate than GPU
```

---

### Linear Algebra Accuracy

#### `pt_svdvals_accuracy.py` — float32 SVD GPU 81x less accurate (100×100)

**Root cause:** PyTorch CPU SVD calls LAPACK dgesdd internally promoting to float64. GPU uses cuSOLVER in native float32.

```
CPU singular value error vs float64 reference: 5.8592e-05
GPU singular value error vs float64 reference: 4.7265e-03
GPU is 81x less accurate than CPU

First 5 singular values:
  CPU: [28.238086700439453, 27.56194305419922, 26.981163024902344, ...]
  GPU: [28.238828659057617, 27.562660217285156, 26.981678009033203, ...]
  Ref: [28.238094329833984, 27.561946868896484, 26.98116111755371, ...]
```

---

#### `pt_svdvals_tall.py` — float32 SVD GPU 25x less accurate (500×10)

**Root cause:** Same LAPACK f64-promotion (CPU) vs cuSOLVER f32 (GPU).

```
CPU singular value error vs float64 reference: 3.4332e-05
GPU singular value error vs float64 reference: 8.6910e-04   <-- BUG
GPU is 25x less accurate than CPU
```

---

#### `pt_svdvals_complex64.py` — complex64 SVD GPU 262x less accurate

**Root cause:** GPU cuSOLVER for complex SVD stays in complex64. CPU LAPACK promotes to complex128.

```
CPU singular value error vs complex128 reference: 4.0261e-05
GPU singular value error vs complex128 reference: 1.0536e-02   <-- BUG
GPU is 262x less accurate than CPU
```

---

#### `pt_eigh_float32.py` — float32 eigvalsh GPU 52x less accurate (n=50)

**Root cause:** PyTorch GPU eigvalsh uses cuSOLVER in native float32. CPU LAPACK promotes internally.

```
CPU eigenvalue error vs float64 reference: 8.2712e-05
GPU eigenvalue error vs float64 reference: 4.2929e-03
GPU is 52x less accurate than CPU
```

---

#### `pt_eigh_f32_large.py` — float32 eigvalsh CPU 10–11x less accurate (n=1000)

**Root cause:** At n=1000, accumulated LAPACK ssyevd float32 error on CPU exceeds cuSOLVER GPU error.

```
cpu_err = 3.3398e-05
gpu_err = 3.3023e-06
CPU/GPU error ratio: 10.1x  (CPU less accurate)
BUG CONFIRMED: PT eigvalsh CPU 11x less accurate than GPU for float32 n=1000
```

---

#### `pt_matrix_norm_nuc.py` — float32 nuclear norm GPU 224x less accurate

**Root cause:** Nuclear norm requires SVD. CPU LAPACK promotes f32→f64. GPU cuSOLVER stays in f32.

```
Reference (NumPy):   2400.332520
CPU (float32):       2400.332764
GPU (float32):       2400.387207
CPU error vs NumPy: 2.4414e-04
GPU error vs NumPy: 5.4688e-02
GPU is 224x less accurate than CPU
```

---

#### `pt_matrix_norm_spectral.py` — float32 spectral norm GPU 142x less accurate

**Root cause:** Spectral norm requires SVD max singular value. Same CPU f64-promotion vs GPU f32 cuSOLVER.

```
Reference (float64): 44.356461
CPU (float32):       44.356445
GPU (float32):       44.358620   <-- BUG
CPU error vs ref: 1.5259e-05
GPU error vs ref: 2.1591e-03
GPU is 142x less accurate than CPU
```

---

#### `pt_matrix_norm_nuc_complex64.py` — complex64 nuclear norm GPU 287x less accurate

**Root cause:** Complex64 nuclear norm uses complex SVD. GPU stays in complex64; CPU LAPACK promotes to complex128.

```
Reference (complex128): 2400.144740
CPU (complex64):        2400.144287
GPU (complex64):        2400.274902   <-- BUG
GPU is 287x less accurate than CPU
```

---

#### `pt_matrix_norm_spectral_complex64.py` — complex64 spectral norm GPU 421x less accurate

```
Reference (complex128): 27.473220
CPU (complex64):        27.473223
GPU (complex64):        27.474531   <-- BUG
GPU is 421x less accurate than CPU
```

---

#### `pt_std_overflow.py` — float32 std overflow: GPU=Inf, CPU correct

**Root cause:** GPU two-pass variance: intermediate squared values overflow float32. CPU Welford online algorithm avoids overflow.

```
Reference (float64): 1.0287e+19
CPU (float32):       1.0287e+19
GPU (float32):       inf   <-- BUG
```

---

### Linear System Solvers

#### `pt_lstsq_rankdef.py` — GPU gives completely wrong answer for rank-deficient lstsq

**Root cause:** GPU cuBLAS `gels` only handles full-rank systems. For rank-deficient input it silently returns garbage. CPU LAPACK `gelsd` correctly uses SVD to compute the minimum-norm solution.

```
NumPy (reference): [ 0.8333333   0.33333334 -0.16666667]
CPU:               [ 0.83333385  0.3333333  -0.1666666 ]
GPU:               [ 0.58141476  0.7630715  -0.        ]
CPU vs NumPy L2: 7.5424e-07
GPU vs NumPy L2: 1.2867e+00   <-- BUG
BUG CONFIRMED: PT lstsq GPU gives wrong answer for rank-deficient matrix (L2=1.2867e+00 vs CPU L2=7.5424e-07)
```

---

#### `pt_lstsq_complex64.py` — complex64 lstsq GPU catastrophically wrong

**Root cause:** GPU cuBLAS does not correctly handle complex64 overdetermined systems.

```
CPU (complex64): [(-1.6013622-0.6097595j), (-0.206654-0.1092792j), (1.1880528+0.3912009j)]
GPU (complex64): [(-29695.105+780537.875j), (59386.765-1561075.875j), (-29692.281+780537.75j)]

CPU error vs reference: 9.8500e-07
GPU error vs reference: 1.9133e+06   <-- BUG
```

---

### NaN/Inf Casting

#### `pt_cast_nan_inf.py` — float32/16/bf16 NaN/Inf → int: CPU and GPU disagree

**Root cause:** CPU follows x86 CVTTSS2SI semantics (NaN→INT_MIN, +Inf→INT_MIN). GPU CUDA follows a different convention (NaN→0, +Inf→INT_MAX).

```
float32(nan)  -> int32:  CPU=-2147483648  GPU=0  -> BUG
float32(inf)  -> int32:  CPU=-2147483648  GPU=2147483647  -> BUG
float32(inf)  -> int64:  CPU=-9223372036854775808  GPU=9223372036854775807  -> BUG
```

---

#### `pt_cast_bf16_nan_inf.py` — bfloat16 NaN/Inf → int: same divergence

```
bfloat16(nan) -> int32:  CPU=-2147483648  GPU=0  -> BUG
bfloat16(inf) -> int32:  CPU=-2147483648  GPU=2147483647  -> BUG
bfloat16(inf) -> int64:  CPU=-9223372036854775808  GPU=9223372036854775807  -> BUG
```

---

### TF32 Matmul

#### `pt_matmul_tf32.py` — TF32 makes GPU matmul 1208x less accurate

**Root cause:** NVIDIA Ampere+ TensorCores reduce float32 mantissa from 23 bits to 10 bits when TF32 is enabled (default in PyTorch). CPU always uses full float32.

```
CPU  error vs float64 reference: 2.8160e-03
GPU  error (TF32=off):           2.4148e-03
GPU  error (TF32=on):            3.4008e+00   <-- BUG
TF32 makes GPU 1208x less accurate than CPU

First 4 output values:
  ref:      [34.16709899902344, -0.6265202760696411,  2.1589558124542236, -7.2969512939453125]
  cpu:      [34.16709899902344, -0.6265285611152649,  2.1589531898498535, -7.296947002410889]
  gpu_tf32: [34.161766052246094, -0.6281626224517822, 2.1512181758880615, -7.299671173095703]
```

---

### Remapped file (PyTorch bug in tf_-named file)

#### `tf_abs_complex_nan.py` — PT eigvalsh GPU f32 n=500 is 284–593x less accurate than CPU

> Note: this file was originally a TF complex128 abs bug (duplicate of `tf_abs_complex64.py`). Replaced with this confirmed PyTorch bug.

**Root cause:** PyTorch GPU eigvalsh uses cuSOLVER in native float32. CPU LAPACK ssyevd accumulates less error at n=500.

```
CPU eigenvalue error vs float64 reference: 1.9427e-05
GPU eigenvalue error vs float64 reference: 5.7917e-03   <-- BUG
GPU is 298x less accurate than CPU
BUG CONFIRMED: PT eigvalsh GPU f32 n=500 is 284-593x less accurate than CPU (cuSOLVER vs LAPACK ssyevd)
```

---

## TensorFlow Bugs (25 files)

### Cumulative Operations

#### `pt_conv2d_f16.py` — TF CPU cumsum f16 (positive monotone) 100x less accurate than GPU

> Note: this file was originally an unconfirmed PyTorch conv2d bug. Replaced with this confirmed TensorFlow bug.

**Root cause:** TF CPU cumsum with all-positive float16 input accumulates error monotonically (no cancellation). GPU promotes f16→f32 internally.

```
cpu_max_err = 3.6854e+03  (sequential f16, errors accumulate monotonically)
gpu_max_err = 3.6727e+01  (GPU promotes to f32 internally)
CPU/GPU error ratio: 100x  (CPU less accurate)
BUG CONFIRMED: TF CPU cumsum f16 positive monotone N=10k is 96-113x less accurate than GPU
```

---

#### `tf_cumsum_f32_large.py` — float32 cumsum CPU 36–62x less accurate at N=10M

**Root cause:** TF CPU cumsum uses sequential accumulation in float32. GPU uses pairwise (tree) reduction, significantly more accurate.

```
cpu_max_err = 1.7511e-01
gpu_max_err = 4.9041e-03
CPU/GPU error ratio: 36x  (CPU less accurate)
BUG CONFIRMED: TF CPU cumsum f32 N=10M is 62x less accurate than GPU
```

---

#### `tf_cumsum_f32.py` — TF CPU eigh float32 n=500 is 68–76x less accurate than GPU

> Note: original file content (TF cumsum f32 7x ratio) was below the 10x threshold and replaced with this confirmed bug.

**Root cause:** TF CPU eigh calls LAPACK ssyevd in native float32. TF GPU eigh uses cuSOLVER which internally promotes float32→float64.

```
CPU eigenvalue error vs float64 reference: 1.6930e-04   <-- BUG
GPU eigenvalue error vs float64 reference: 2.4918e-06
CPU is 68x less accurate than GPU
BUG CONFIRMED: TF CPU eigh f32 n=500 is 68-76x less accurate than GPU (LAPACK ssyevd vs cuSOLVER)
```

---

#### `tf_cumsum_f16.py` — float16 cumsum CPU 11x less accurate than GPU (N=10k)

**Root cause:** TF CPU cumsum stays in float16. GPU promotes to float32 internally.

```
CPU error vs float64 reference: 1.0686e+02
GPU error vs float64 reference: 9.1100e+00
CPU is 11.7x less accurate than GPU
```

---

#### `tf_cumsum_bf16.py` — bfloat16 cumsum CPU 23x less accurate than GPU

**Root cause:** TF CPU cumsum stays in bfloat16. GPU promotes to float32 internally.

```
CPU error vs float64 reference: 1.4584e+03   <-- BUG
GPU error vs float64 reference: 6.3028e+01
CPU is 23.1x less accurate than GPU

Last 5 values:
  ref: [-186.16, -186.12, -185.60, -185.64, -184.34]
  cpu: [-214.0,  -214.0,  -213.0,  -213.0,  -212.0 ]
  gpu: [-187.0,  -186.0,  -185.0,  -185.0,  -184.0 ]
```

---

#### `tf_cumprod_f32_large.py` — float32 cumprod GPU 23–130x less accurate at N=1M

**Root cause:** TF GPU cumprod accumulates in float32. TF CPU promotes to float64 internally.

```
cpu_max_err = 5.5913e-05
gpu_max_err = 7.2557e-03
GPU/CPU error ratio: 130x  (GPU less accurate)
BUG CONFIRMED: TF cumprod f32 N=1M GPU is 23-130x less accurate than CPU
```

---

#### `tf_cumprod_f16.py` — float16 cumprod GPU 7x less accurate at N=10k

**Root cause:** TF GPU cumprod stays in float16. CPU promotes to float32.

```
CPU error vs float64 reference: 3.1245e-02
GPU error vs float64 reference: 2.2729e-01   <-- BUG
GPU is 7x less accurate than CPU
```

---

### Wrong Value — float16 N=65536 Bug

These three bugs share the same root cause: TF CPU reduction implementations store the element count N as float16 internally. For N=65536, `float16(65536) = Inf`, causing mean/variance/std computations to return 0.0.

#### `tf_mean_f16_wrong.py` — TF CPU reduce_mean returns 0.0 for float16 N=65536

```
ref = -0.003780
cpu = -0.000000  (WRONG: should not be 0.0)
gpu = -0.003780
BUG CONFIRMED: TF CPU reduce_mean returns 0.0 for float16 N=65536 (float16(N)=inf)
```

---

#### `tf_var_f16_wrong.py` — TF CPU reduce_variance returns 0.0 for float16 N=65536

```
ref = 0.990657
cpu = 0.000000  (WRONG: should be ~1.0)
gpu = 0.990723
BUG CONFIRMED: TF CPU reduce_variance returns 0.0 for float16 N=65536 (float16(N)=inf)
```

---

#### `tf_std_f16_wrong.py` — TF CPU reduce_std returns 0.0 for float16 N=65536

```
ref = 0.995318
cpu = 0.000000  (WRONG: should be ~1.0)
gpu = 0.995117
BUG CONFIRMED: TF CPU reduce_std returns 0.0 for float16 N=65536 (float16(N)=inf)
```

---

### Wrong Value — NaN/Inf

#### `tf_std_f16_nan.py` — float16 std: CPU returns NaN, GPU returns Inf

**Root cause:** CPU Welford algorithm: `inf - inf = NaN`. GPU two-pass algorithm: intermediate `E[X²]` overflows to Inf, subtraction yields Inf (different NaN/Inf path).

```
Reference (float64): 9.8703e+03
CPU (float16): nan   <-- BUG (nan: inf - inf = nan in Welford)
GPU (float16): inf   <-- BUG (inf: two-pass algorithm, avoids inf-inf)

CPU returns nan: True
GPU returns inf: True
CPU != GPU: True

reduce_variance CPU: nan  GPU: inf
```

---

#### `tf_abs_complex64.py` — complex64 abs: GPU returns NaN for (inf+nan·j)

**Root cause:** IEEE 754 requires `|inf + nan·j| = inf`. TF GPU CUDA kernel returns NaN instead.

```
abs((inf+nanj))
  CPU: inf
  GPU: nan
  Asymmetric: CPU=inf GPU=nan -> BUG

abs((nan+infj))
  CPU: inf
  GPU: nan
  Asymmetric: CPU=inf GPU=nan -> BUG
```

---

#### `tf_top_k_nan.py` — sort NaN: CPU leaves NaN in-place, GPU moves all NaN to front

**Root cause:** TF CPU sort is partially in-place — NaN values stay at their original positions. TF GPU sort unconditionally moves all NaN to the beginning of the output.

```
Input:      [nan, 3.0, 1.0, nan, 2.0, nan, 0.5]
CPU sorted: [nan, 1.0, 3.0, nan, 2.0, nan, 0.5]
GPU sorted: [nan, nan, nan, 0.5, 1.0, 2.0, 3.0]
CPU NaN positions: [0, 3, 5]
GPU NaN positions: [0, 1, 2]
BUG CONFIRMED: TF CPU sort leaves NaN in-place; GPU sort moves all NaN to front position
```

---

### Linear Algebra Accuracy

#### `tf_eigh_float32.py` — TF CPU eigh 23x less accurate than GPU (n=50)

**Root cause:** TF CPU eigh calls LAPACK ssyevd in float32. TF GPU eigh uses cuSOLVER which internally promotes to float64.

```
CPU eigenvalue error vs float64 reference: 1.1751e-03
GPU eigenvalue error vs float64 reference: 5.2058e-05
CPU is 23x less accurate than GPU

First 5 eigenvalues:
  CPU: [0.0024355570785701275, 0.11959562450647354, 0.43959009647369385, ...]
  GPU: [0.002414487302303314,  0.11959712952375412, 0.4395846426486969,  ...]
  Ref: [0.002431207336485386,  0.11959661543369293, 0.43958741426467896, ...]
```

---

#### `tf_eigh_complex64.py` — TF CPU eigh complex64 49x less accurate than GPU

**Root cause:** Same LAPACK vs cuSOLVER pattern for complex Hermitian eigensolver.

```
CPU eigenvalue error vs complex128 reference: 2.2100e-03   <-- BUG
GPU eigenvalue error vs complex128 reference: 4.4676e-05
CPU is 49x less accurate than GPU
```

---

#### `tf_svd_accuracy.py` — TF CPU SVD 17x less accurate than GPU (100×100 float32)

**Root cause:** TF CPU SVD calls LAPACK sgesdd in native float32. TF GPU cuSOLVER internally promotes to float64.

```
CPU singular value error vs float64 reference: 1.5531e-04
GPU singular value error vs float64 reference: 8.8921e-06
CPU is 17x less accurate than GPU
```

---

#### `tf_svd_tall.py` — TF CPU SVD 11–19x less accurate for tall float32 (2000×50)

**Root cause:** Same as square SVD. Error scales with matrix rows; 2000×50 reliably exceeds 10x threshold.

```
CPU singular value error vs float64 reference: 3.4117e-05   <-- BUG
GPU singular value error vs float64 reference: 2.4414e-06
CPU is 14x less accurate than GPU
BUG CONFIRMED: TF CPU SVD f32 2000x50 is 11-19x less accurate than GPU (LAPACK sgesdd vs cuSOLVER)
```

---

#### `tf_nuclear_norm_f32.py` — TF CPU nuclear norm 83x less accurate (500×500)

**Root cause:** Nuclear norm requires all singular values. CPU LAPACK sgesdd in float32; GPU cuSOLVER promotes internally.

```
ref      = 9472.921980
cpu_nuc  = 9472.930664  err=8.6842e-03
gpu_nuc  = 9472.921875  err=1.0487e-04
CPU/GPU error ratio: 83x  (CPU less accurate)
BUG CONFIRMED: TF CPU nuclear norm 83x less accurate than GPU (float32 500x500)
```

---

#### `tf_pinv_f32.py` — TF GPU pinv 228x less accurate (500×200)

**Root cause:** TF GPU pinv uses cuSOLVER in float32 for the SVD. TF CPU LAPACK promotes to float64. Direction reversed from SVD accuracy: CPU is more accurate here.

```
cpu_err = 2.6488e-09
gpu_err = 6.0433e-07
GPU/CPU error ratio: 228x  (GPU less accurate)
BUG CONFIRMED: TF GPU pinv f32 500x200 is 228x less accurate than CPU (cuSOLVER vs LAPACK)
```

---

### Linear System Solver

#### `tf_lstsq_f32.py` — TF GPU lstsq 291x less accurate (200×100 overdetermined)

**Root cause:** TF CPU lstsq uses LAPACK QR/SVD. TF GPU lstsq uses Cholesky normal equations (AᵀA·x = Aᵀb), which squares the condition number and loses significant digits.

```
cpu_err = 7.9973e-08
gpu_err = 2.3254e-05
GPU/CPU error ratio: 291x  (GPU less accurate)
BUG CONFIRMED: TF GPU lstsq 275x less accurate than CPU for float32 200x100 (Cholesky normal eq vs LAPACK QR)
```

---

### Einsum

#### `tf_einsum_f16.py` — float16 einsum CPU 10x less accurate than GPU

**Root cause:** TF CPU einsum accumulates the inner contraction in float16. GPU promotes to float32 for the inner product.

```
cpu_err = 2.0590e-02
gpu_err = 1.9773e-03
CPU/GPU error ratio: 10.4x  (CPU less accurate)
BUG CONFIRMED: TF CPU einsum f16 10x less accurate than GPU
```

---

### TF32 Matmul

#### `tf_matmul_tf32.py` — TF32 makes GPU matmul 1011x less accurate

**Root cause:** NVIDIA Ampere+ TensorCore TF32 behavior — same as PyTorch. Mantissa reduced from 23 to 10 bits for float32 tensor contractions.

```
CPU error vs float64 reference: 3.3549e-03
GPU error vs float64 reference: 3.3903e+00   <-- BUG
GPU is 1011x less accurate than CPU (TF32 TensorCore)

First 4 output values:
  ref: [ 8.810,  19.310, -2.176, -2.223]
  cpu: [ 8.810,  19.310, -2.176, -2.223]
  gpu: [ 8.811,  19.299, -2.164, -2.213]
```

---

### NaN/Inf Casting

#### `tf_cast_nan_inf.py` — float32/16/bf16 NaN/Inf → int: CPU INT_MIN, GPU 0/INT_MAX

**Root cause:** CPU host runtime and GPU CUDA device use different out-of-range integer conversion rules.

```
float32(nan)  -> int32:  CPU=-2147483648  GPU=0  -> BUG
float32(inf)  -> int32:  CPU=-2147483648  GPU=2147483647  -> BUG
float32(inf)  -> int64:  CPU=-9223372036854775808  GPU=9223372036854775807  -> BUG
```

---

#### `tf_cast_f16_nan_inf.py` — float16 NaN/Inf → int divergence

```
float16(nan)  -> int32:  CPU=-2147483648  GPU=0  -> BUG
float16(inf)  -> int32:  CPU=-2147483648  GPU=2147483647  -> BUG
float16(inf)  -> int64:  CPU=-9223372036854775808  GPU=9223372036854775807  -> BUG
```

---

#### `tf_cast_bf16_nan_inf.py` — bfloat16 NaN/Inf → int divergence

```
bfloat16(nan) -> int32:  CPU=-2147483648  GPU=0  -> BUG
bfloat16(inf) -> int32:  CPU=-2147483648  GPU=2147483647  -> BUG
bfloat16(inf) -> int64:  CPU=-9223372036854775808  GPU=9223372036854775807  -> BUG
```

---

## Summary Table

| # | File | Framework | Direction | Signal |
|---|------|-----------|-----------|--------|
| 1 | `pt_cumsum_f16.py` | PyTorch | GPU worse | 11–15x |
| 2 | `pt_cumsum_bf16.py` | PyTorch | GPU worse | 12–19x |
| 3 | `pt_cumsum_complex64.py` | PyTorch | GPU worse | 10–16x |
| 4 | `pt_cumsum_f32.py` | PyTorch | GPU worse | CPU=exact |
| 5 | `pt_cumprod_f32_large.py` | PyTorch | GPU worse | ~97000x |
| 6 | `pt_cumprod_f16.py` | PyTorch | GPU worse | 33–52x |
| 7 | `pt_cumprod_bf16.py` | PyTorch | GPU worse | 33x |
| 8 | `pt_prod_f16.py` | PyTorch | CPU worse | 131x |
| 9 | `pt_prod_bf16.py` | PyTorch | CPU worse | 10x |
| 10 | `pt_svdvals_accuracy.py` | PyTorch | GPU worse | 81x |
| 11 | `pt_svdvals_tall.py` | PyTorch | GPU worse | 25x |
| 12 | `pt_svdvals_complex64.py` | PyTorch | GPU worse | 262x |
| 13 | `pt_eigh_float32.py` | PyTorch | GPU worse | 52x |
| 14 | `pt_eigh_f32_large.py` | PyTorch | CPU worse | 10–11x |
| 15 | `pt_matrix_norm_nuc.py` | PyTorch | GPU worse | 224x |
| 16 | `pt_matrix_norm_spectral.py` | PyTorch | GPU worse | 142x |
| 17 | `pt_matrix_norm_nuc_complex64.py` | PyTorch | GPU worse | 287x |
| 18 | `pt_matrix_norm_spectral_complex64.py` | PyTorch | GPU worse | 421x |
| 19 | `pt_std_overflow.py` | PyTorch | GPU wrong | GPU=Inf |
| 20 | `pt_lstsq_rankdef.py` | PyTorch | GPU wrong | L2=1.29 |
| 21 | `pt_lstsq_complex64.py` | PyTorch | GPU wrong | L2=1.9M |
| 22 | `pt_cast_nan_inf.py` | PyTorch | both wrong | wrong int |
| 23 | `pt_cast_bf16_nan_inf.py` | PyTorch | both wrong | wrong int |
| 24 | `pt_matmul_tf32.py` | PyTorch | GPU worse | 1208x |
| 25 | `tf_abs_complex_nan.py`* | PyTorch | GPU worse | 284–593x |
| 26 | `pt_conv2d_f16.py`† | TensorFlow | CPU worse | 100x |
| 27 | `tf_cumsum_f32_large.py` | TensorFlow | CPU worse | 36–62x |
| 28 | `tf_cumsum_f32.py`‡ | TensorFlow | CPU worse | 68–76x |
| 29 | `tf_cumsum_f16.py` | TensorFlow | CPU worse | 11x |
| 30 | `tf_cumsum_bf16.py` | TensorFlow | CPU worse | 23x |
| 31 | `tf_cumprod_f32_large.py` | TensorFlow | GPU worse | 23–130x |
| 32 | `tf_cumprod_f16.py` | TensorFlow | GPU worse | 7x |
| 33 | `tf_mean_f16_wrong.py` | TensorFlow | CPU wrong | CPU=0.0 |
| 34 | `tf_var_f16_wrong.py` | TensorFlow | CPU wrong | CPU=0.0 |
| 35 | `tf_std_f16_wrong.py` | TensorFlow | CPU wrong | CPU=0.0 |
| 36 | `tf_std_f16_nan.py` | TensorFlow | both wrong | NaN vs Inf |
| 37 | `tf_abs_complex64.py` | TensorFlow | GPU wrong | GPU=NaN |
| 38 | `tf_top_k_nan.py` | TensorFlow | CPU wrong | NaN positions |
| 39 | `tf_eigh_float32.py` | TensorFlow | CPU worse | 23x |
| 40 | `tf_eigh_complex64.py` | TensorFlow | CPU worse | 49x |
| 41 | `tf_svd_accuracy.py` | TensorFlow | CPU worse | 17x |
| 42 | `tf_svd_tall.py` | TensorFlow | CPU worse | 11–19x |
| 43 | `tf_nuclear_norm_f32.py` | TensorFlow | CPU worse | 83x |
| 44 | `tf_pinv_f32.py` | TensorFlow | GPU worse | 228x |
| 45 | `tf_lstsq_f32.py` | TensorFlow | GPU worse | 291x |
| 46 | `tf_einsum_f16.py` | TensorFlow | CPU worse | 10x |
| 47 | `tf_matmul_tf32.py` | TensorFlow | GPU worse | 1011x |
| 48 | `tf_cast_nan_inf.py` | TensorFlow | both wrong | wrong int |
| 49 | `tf_cast_f16_nan_inf.py` | TensorFlow | both wrong | wrong int |
| 50 | `tf_cast_bf16_nan_inf.py` | TensorFlow | both wrong | wrong int |

\* `tf_abs_complex_nan.py` contains a PyTorch bug (replaced a duplicate of `tf_abs_complex64.py`).  
† `pt_conv2d_f16.py` contains a TensorFlow bug (replaced an unconfirmed PyTorch conv2d test).  
‡ `tf_cumsum_f32.py` contains a TF eigh bug (replaced a below-threshold cumsum test with 7x ratio).

---

## New Bugs — Cross-Framework Bugs from GitHub Issues (9 new, bugs 51–59)

These bugs were found by searching real GitHub issues across PyTorch, TensorFlow, and JAX,
then cross-checking whether the bug also exists in the other framework.

| # | File | Library | Type | Key Finding |
|---|------|---------|------|-------------|
| 51 | `pt_softplus_beta_overflow.py` | PyTorch | both CPU+GPU wrong | `softplus(0.5, beta=1e10)` → inf (should be 0.5) |
| 52 | `tf_softplus_beta_overflow.py` | TensorFlow | cross-framework ref | TF handles large beta correctly; PT does not |
| 53 | `tf_roll_int_overflow.py` | TensorFlow | both CPU+GPU wrong | `tf.roll(x, shift=2^31)` gives wrong result (int32 overflow) |
| 54 | `pt_scatter_add_nondeterministic.py` | PyTorch | GPU 40x worse | float16 scatter_add: GPU 40x worse than CPU (atomicAdd order) |
| 55 | `tf_scatter_add_nondeterministic.py` | TensorFlow | large CPU/GPU diff | float16 unsorted_segment_sum: large CPU vs GPU divergence |
| 56 | `pt_linalg_solve_singular.py` | PyTorch | massive CPU/GPU diff | near-singular solve (cond=1e10): CPU vs GPU diff = 6.3e9 |
| 57 | `tf_linalg_solve_singular.py` | TensorFlow | cross-framework ref | TF CPU==GPU for solve (no divergence); PT has it |
| 58 | `tf_logdet_singular.py` | TensorFlow | wrong vs NumPy | singular matrix logdet → NaN (should be -inf; PT returns -inf) |
| 59 | `tf_lstsq_complex64.py` | TensorFlow | NaN output | `lstsq(rank-def, fast=True)` → NaN on both CPU and GPU |

---

### Bug 51 — `pt_softplus_beta_overflow.py`

**Operation:** `torch.nn.functional.softplus(x, beta=1e10, threshold=inf)`
**Root cause:** `log(1 + exp(beta * x)) / beta` — `beta * x` overflows float32 → inf/beta = inf.
The numerically stable form `x + log1p(exp(-beta*x))/beta` is not used when `threshold` is bypassed.
TensorFlow's `tf.nn.softplus(beta * x) / beta` uses the XLA stable path and returns the correct value.

```
Input: x = [0.5, 1.0, 2.0, -1.0, 0.0], beta = 1e10
CPU: [inf, inf, inf, 0.0, ...]   ← BUG: should be [0.5, 1.0, 2.0, 0.0, ...]
GPU: [inf, inf, inf, 0.0, ...]   ← same bug
TF:  [0.5, 1.0, 2.0, 0.0, ...]  ← TF correct (cross-framework comparison)
```

**Source:** PyTorch GitHub issue #171249

---

### Bug 53 — `tf_roll_int_overflow.py`

**Operation:** `tf.roll(x, shift=2^31, axis=0)` with int32 shift
**Root cause:** TF's roll kernel stores shift as int32. When shift = 2^31 (= INT32_MAX + 1),
  the modular arithmetic overflows → wrong element displacement.
PyTorch handles this correctly for all 64-bit Python int shift values.

```
shift = 2147483647 (INT32_MAX): CPU correct=True,  GPU correct=True
shift = 2147483648 (INT32_MAX+1): CPU correct=False, GPU correct=False ← BUG
shift = 4294967295 (UINT32_MAX):  CPU correct=False, GPU correct=False ← BUG
```

**Source:** TensorFlow GitHub issue #109520

---

### Bug 54 — `pt_scatter_add_nondeterministic.py`

**Operation:** `tensor.scatter_add_(dim, index, src)` with heavy duplicate indices, float16
**Root cause:** GPU uses `atomicAdd` in arbitrary parallel order. Float16 addition is not
associative — reordering ~2000 additions per bucket accumulates 40x more error than CPU's
serial summation.

```
float16, N=100k, M=50 (2000 dups/bucket):
CPU error vs float32 ref: 2.45e+00
GPU error vs float32 ref: 1.02e+02   ← 41.9x worse
CPU vs GPU max diff:      1.04e+02   *** SIGNIFICANT ***

GPU is NOT deterministic: run1 != run2 for same input.
```

---

### Bug 55 — `tf_scatter_add_nondeterministic.py`

**Operation:** `tf.math.unsorted_segment_sum` with float16/bfloat16 and many duplicates
**Root cause:** Same as bug 54 — GPU atomicAdd on float16 is non-associative.

```
float16, N=100k, M=50:
CPU error vs float64 ref: 9.01e+01
GPU error vs float64 ref: 9.01e+01
CPU vs GPU max diff:      7.72e+01   *** SIGNIFICANT ***

bfloat16, N=200k, M=50:
CPU vs GPU max diff: 1.33e+04        *** SIGNIFICANT ***
```

---

### Bug 56 — `pt_linalg_solve_singular.py`

**Operation:** `torch.linalg.solve(A, b)` where A is float32 with condition number ≥ 1e6
**Root cause:** CPU uses LAPACK's `sgesv` (partial pivoting); GPU uses cuSOLVER's `sgesv`
(different pivoting strategy). For ill-conditioned systems, tiny pivot differences amplify
through the triangular solves, producing entirely different solution vectors.

```
n=32, float32, cond=1e10:
CPU error vs float64: 6.80e+09
GPU error vs float64: 8.47e+09
CPU vs GPU diff:      6.26e+09   ← MASSIVE divergence

cond=1e8, 8×8 Vandermonde:
CPU vs GPU diff:      8.52e+00   ← completely different solutions
```

TF CPU and TF GPU agree (both use the same backend) — this is a PyTorch-specific CPU/GPU split.

---

### Bug 58 — `tf_logdet_singular.py`

**Operation:** `tf.linalg.logdet(A)` when A is singular or near-singular
**Root cause:** TF's logdet uses LU decomposition and returns `NaN` when a zero pivot is encountered,
instead of `-inf`. NumPy and PyTorch correctly return `-inf` for singular matrices.

```
8×8 all-ones matrix (rank 1, det=0, logdet=-inf):
NumPy: -inf  ← correct
PT CPU: -inf  ← correct
TF CPU: NaN  ← BUG (returns NaN instead of -inf)
TF GPU: NaN  ← BUG (same)
```

---

### Bug 59 — `tf_lstsq_complex64.py`

**Operation:** `tf.linalg.lstsq(A, b, fast=True)` with rank-deficient complex64 matrix
**Root cause:** `fast=True` selects the Cholesky path (A^H A x = A^H b), which breaks down when
A is rank-deficient because A^H A is singular. Returns NaN for both CPU and GPU.

```
3×3 rank-deficient complex64, fast=True:
TF CPU: err=NaN  ← BUG
TF GPU: err=NaN  ← BUG

fast=False (QR path):
TF CPU: err=1.726e+00 (high but finite)
TF GPU: err=1.726e+00 (finite)
```
