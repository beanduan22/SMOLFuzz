# Open / Unfixed Bug Reproducers

Minimal repros for **open and unfixed** bugs in PyTorch / TensorFlow.
Each `.py` file is a single bare bug — no helpers, no harness — and runs in a few lines.

Reproducer outputs below are captured on:
- PyTorch **2.11.0+cu128**, CUDA **12.8**
- TensorFlow **2.21.0**
- GPU: **NVIDIA RTX 6000 Ada Generation** (compute 8.9)

> Bug status (open / unfixed) is upstream truth at the time the catalog was mined.
> Some hardware/version combinations may mask numerical bugs (notably `tf_86350` below).

---

## CPU vs GPU divergence

### `cpu_gpu/pt_162235_neg_zero.py` — pytorch#162235

```
cpu maximum : tensor([-0.])
gpu maximum : tensor([0.], device='cuda:0')
cpu relu    : tensor([-0.])
gpu relu    : tensor([0.], device='cuda:0')
cpu argsort : tensor([0, 1])
gpu argsort : tensor([1, 0], device='cuda:0')
cpu amin    : tensor(-0.)
gpu amin    : tensor(0., device='cuda:0')
cpu max/torch.float64: tensor([-0.], dtype=torch.float64)
gpu max/torch.float64: tensor([0.], device='cuda:0', dtype=torch.float64)
cpu max/torch.float32: tensor([-0.])
gpu max/torch.float32: tensor([0.], device='cuda:0')
cpu max/torch.float16: tensor([-0.], dtype=torch.float16)
gpu max/torch.float16: tensor([0.], device='cuda:0', dtype=torch.float16)
```

→ **bug reproduces.** CPU preserves the `-0.0` sign bit; CUDA normalises to `+0.0`. argsort permutation flips. All three floating dtypes affected.

### `cpu_gpu/pt_52241_ctcloss_grad.py` — pytorch#52241

```
torch.autograd.gradcheck.GradcheckError: Jacobian mismatch for output 0 with respect to input 0,
numerical:tensor([[-0.8953],
                  [-0.1047],
                  [ 0.0000],
                  [-0.7510],
                  [-0.2490],
                  [ 0.0000],
                  [-0.1682],
                  [-0.8318],
                  [ 0.0000]], dtype=torch.float64)
```

→ **bug reproduces.** `gradcheck` fails on the minimal `V=2,T=3,N=1` case — `nn.CTCLoss`'s analytic gradient does not match the finite-difference gradient.

### `cpu_gpu/tf_86378_biasaddgrad.py` — tensorflow#86378

```
NCHW/bf16  diff: 0.03125
NHWC/bf16  diff: 0.0625
NCHW/f16   diff: 0.03125
```

→ **bug reproduces** across NCHW/NHWC and bf16/f16. Numeric divergence between CPU and GPU is well above bf16 noise floor.

### `cpu_gpu/tf_86256_adjust_hue.py` — tensorflow#86256

```
original  : cpu= [-0.3937407  -0.25841027 -0.0503466 ] gpu= [-0.0503466 -0.0503466 -0.0503466]
max|diff| : 0.3433941
delta=-0.99 max|diff|=2.235e-07
delta=-0.74 max|diff|=4.172e-07
delta=-0.50 max|diff|=4.172e-07
delta=-0.10 max|diff|=3.576e-07
delta=+0.50 max|diff|=4.172e-07
```

→ **bug reproduces** on the issue's exact inputs (`max|diff|=0.34`). Generic random inputs only show f32 rounding noise (~4e-7) — the bug is value-specific.

### `cpu_gpu/tf_96180_reciprocal_complex_inf.py` — tensorflow#96180

```
cpu pure-inf   : [nan+nanj nan+nanj  0. +0.j]
gpu pure-inf   : [0.+0.j 0.+0.j 0.+0.j]
cpu inf+nan    : [nan+nanj nan+nanj nan+nanj nan+nanj]
gpu inf+nan    : [ 0. +0.j  0. +0.j  0. +0.j nan+nanj]
cpu c64        : [0.+0.j 0.-0.j 0.-0.j]
gpu c64        : [0.+0.j 0.-0.j 0.-0.j]
```

→ **bug reproduces** for complex128. CPU returns NaN where GPU correctly returns 0. NaN poisoning visible: in the `inf+nan` row, CPU's NaN poisons all earlier rows. complex64 path is consistent.

### `cpu_gpu/tf_97204_notequal_nonbroadcast.py` — tensorflow#97204

```
cpu (4,1)x(1,28,2,3,2): tf.Tensor(True, shape=(), dtype=bool)
gpu: InvalidArgumentError: {{function_node __wrapped__NotEqual_device_/...
cpu (2,3,4)x(1,5,6,7,4): tf.Tensor(True, shape=(), dtype=bool)
gpu: InvalidArgumentError: {{function_node __wrapped__NotEqual_device_/...
```

→ **bug reproduces.** CPU silently returns scalar `True` for non-broadcastable shape pairs; GPU raises `InvalidArgumentError`. Both shape combos exhibit the divergence.

### `cpu_gpu/tf_86350_batchmatmulv2.py` — tensorflow#86350

```
original: max|diff|= 0.0
4D/bf16: max|diff|= 0.0
5D/bf16: max|diff|= 0.0
```

→ **does not reproduce on this hardware.** Issue was reported on Tesla T4; here on RTX 6000 Ada (compute 8.9) the bf16 BMM kernel matches CPU bit-exactly. Issue remains open upstream — likely driver/cuDNN-version dependent.

---

## Crashes

For each file: `python3 <file>` → process dies with the listed signal.

### `crashes/pt_177829_lu_unpack_a_empty_pivots.py` — pytorch#177829

```
Segmentation fault (core dumped)
```

→ **SIGSEGV reproduces.** Empty `int32` `LU_pivots` against 3×3 `LU_data`.

### `crashes/pt_177829_lu_unpack_b_pivots_1x0.py` — pytorch#177829 (variant)

```
Segmentation fault (core dumped)
```

→ **SIGSEGV reproduces.** Shape `(1, 0)` pivots crash; note `(0, 0)` does **not** crash — exposes a shape-rank-specific path.

### `crashes/pt_177829_lu_unpack_c_batched.py` — pytorch#177829 (variant)

```
Segmentation fault (core dumped)
```

→ **SIGSEGV reproduces.** Batched 2×3×3 `LU_data` with empty pivots.

### `crashes/pt_173574_arange_a_int64_float_step.py` — pytorch#173574

```
Floating point exception (core dumped)
```

→ **SIGFPE reproduces.** Float step with int64 `out=` tensor.

### `crashes/pt_173574_arange_b_neg_range.py` — pytorch#173574 (variant)

```
Floating point exception (core dumped)
```

→ **SIGFPE reproduces.** Negative range with fractional step into int64 `out=`.

### `crashes/pt_173574_arange_c_step_0_25.py` — pytorch#173574 (variant)

```
Floating point exception (core dumped)
```

→ **SIGFPE reproduces.** Smaller fractional step (0.25) — still SIGFPE.

### `crashes/tf_76726_encode_png_a_tile_zero.py` — tensorflow#76726

```
F0000 png_io.cc:357] 'image' Must be non NULL
*** Check failure stack trace: ***
    @ absl::log_internal::LogMessage::SendToLog()
    @ tensorflow::png::WriteImageToBuffer<>()
    @ tensorflow::EncodePngOp::Compute()
Aborted (core dumped)
```

→ **SIGABRT reproduces** via `LOG(FATAL)` in `png_io.cc:357`. `tf.tile(..., [0,0,1])` produces an empty image that bypasses validation.

### `crashes/tf_76726_encode_png_b_zero_height.py` — tensorflow#76726 (variant)

```
F0000 png_io.cc:357] 'image' Must be non NULL
Aborted (core dumped)
```

→ **SIGABRT reproduces.** Explicit `tf.zeros((0,4,3))` triggers the same fatal check.

### `crashes/tf_76726_encode_png_c_zero_width.py` — tensorflow#76726 (variant)

```
F0000 png_io.cc:357] 'image' Must be non NULL
Aborted (core dumped)
```

→ **SIGABRT reproduces.** Explicit `tf.zeros((4,0,3))` — confirms any zero-extent in the image triggers the fatal.

---

## Reproduction summary

| Bug | Variants reproduced |
|---|---|
| pytorch#162235 (`-0.0`) | 4/4 |
| pytorch#52241 (`CTCLoss`) | 1/1 |
| tensorflow#86378 (`BiasAddGrad`) | 3/3 |
| tensorflow#86256 (`adjust_hue`) | 1/6 (issue inputs only; bug is value-specific) |
| tensorflow#96180 (`reciprocal`) | 2/3 (complex128 paths) |
| tensorflow#97204 (`NotEqual`) | 2/2 |
| tensorflow#86350 (`BatchMatMulV2`) | 0/3 (hardware/version-dependent — RTX 6000 Ada matches CPU) |
| pytorch#177829 (`lu_unpack` SIGSEGV) | 3/3 |
| pytorch#173574 (`arange` SIGFPE) | 3/3 |
| tensorflow#76726 (`encode_png` SIGABRT) | 3/3 |
