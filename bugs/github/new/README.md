# SMOLFuzz — Confirmed CPU/GPU Divergence Bugs (April 2026)

4 real bugs found after full re-triage on 2026-04-18.  
All bugs are `INCONSISTENT` type: same model, same weights, same inputs — different outputs on CPU vs GPU.  
23 initially flagged bugs were rejected as false positives (see [SUMMARY.md](SUMMARY.md) for root causes).

## Environment

| Component | Version |
|-----------|---------|
| Python | 3.13.5 |
| PyTorch | 2.9.1+cu128 |
| CUDA | 12.8 |
| cuDNN | 9.10.2 |
| GPU | NVIDIA RTX 6000 Ada Generation (48 GB) |
| GPU Driver | 570.211.01 |
| CPU | AMD Ryzen Threadripper PRO 7985WX (64 cores) |

## Run Stats

| | PyTorch |
|--|---------|
| Models tested | 102 |
| Initially flagged | 5 |
| Confirmed real | **4** |
| False positives removed | 1 (m0035: `kaiming_normal_` in forward loop) |
| Fuzzing budget | 60s/model |
| LLM | qwen2.5-coder:32b |

TensorFlow: 200 models tested, 22 initially flagged, **all 22 rejected as FP** — root cause was different random weights on CPU vs GPU due to TF global op counter not resetting between device model initializations.

---

## PyTorch Bugs (4 confirmed)

### PT-1 · m0028 · scale_large · rel_err = 838×

**APIs:** `Linear, ELU, BatchNorm1d, sin, clip, cummax, Tensor.max, row_stack`  
**Output shape:** `[4, 4]`

```
CPU: [0.2614, 0.2614, 0.2614, 0.2614, 0.5000, 0.5000, 0.5000, 0.5000,
      0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000]
GPU: [0.2311, 0.2311, 0.2311, 0.2311, 0.5000, 0.5000, 0.5000, 0.5000,
      0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000]
```

Repro: `python3 pytorch/bug_inconsistent_m0028_mut5_20260416_183107.repro.py`

---

### PT-2 · m0046 · scale_large · rel_err = 163×

**APIs:** `Linear, BatchNorm1d, Mish, hardswish, threshold_, fmod, exp, sin`  
**Output shape:** `[4, 4]`

```
CPU: [-0.3346, -0.4632, -0.3013, -0.7158, -0.1079, -0.5654, -0.4068, -0.7080,
     -0.8391, -0.2875, -0.4747, -1.2912, -0.3692, -0.0015, -0.6657, -1.1881]
GPU: [-0.3417, -0.4571, -0.2973, -0.7106, -0.1081, -0.5653, -0.4068, -0.7081,
     -0.8430, -0.2839, -0.4743, -1.2959, -0.3704, -0.0019, -0.6647, -1.1879]
```

Repro: `python3 pytorch/bug_inconsistent_m0046_mut5_20260416_185555.repro.py`

---

### PT-3 · m0050 · scale_large · rel_err = 259×

**APIs:** `Linear, silu, BatchNorm1d, sin, arctanh, sigmoid, autograd.grad, concat`  
**Output shape:** `[64]`

```
CPU: [ 0.0000,  0.1228,  0.0000,  0.1959,  0.0000,  0.0000,  0.0000, -0.8478,
       0.2078, -0.4086, -0.1328, -0.4223, -0.2601, -0.0456, -0.0185, -0.3582,
       0.8692,  0.0000, -0.8007,  0.0000,  0.0000,  0.0000,  0.0000,  0.3690,
       0.2698, -0.4407, -0.2124, -0.4504, -0.3167, -0.0450,  0.0111, -0.2892,
       0.9990,  0.8293, -0.9791,  0.0000,  0.6678,  0.5675, -0.4889,  0.0000,
       0.2453, -0.3893, -0.1713, -0.4194, -0.2711, -0.0076, -0.0312, -0.2984,
       0.4360,  0.7491,  0.0161,  0.0000,  0.3402,  0.9999,  0.0000,  0.0000,
       0.2208, -0.3621, -0.1191, -0.3916, -0.2557,  0.0161, -0.0265, -0.3049]
GPU: [ 0.0000,  0.1189,  0.0000,  0.2036,  0.0000,  0.0000,  0.0000, -0.8436,
       0.2078, -0.4085, -0.1328, -0.4220, -0.2601, -0.0456, -0.0184, -0.3580,
       0.8687,  0.0000, -0.8007,  0.0000,  0.0000,  0.0000,  0.0000,  0.3704,
       0.2698, -0.4407, -0.2124, -0.4504, -0.3167, -0.0449,  0.0111, -0.2891,
       0.9987,  0.8282, -0.9783,  0.0000,  0.6649,  0.5675, -0.4752,  0.0000,
       0.2454, -0.3895, -0.1715, -0.4196, -0.2714, -0.0078, -0.0307, -0.2982,
       0.4360,  0.7497,  0.0161,  0.0000,  0.3439,  0.9999,  0.0000,  0.0000,
       0.2207, -0.3620, -0.1190, -0.3915, -0.2555,  0.0161, -0.0266, -0.3049]
```

Repro: `python3 pytorch/bug_inconsistent_m0050_mut5_20260416_190108.repro.py`

---

### PT-4 · m0088 · mask · rel_err = 4459×

**APIs:** `Linear, BatchNorm1d, ELU, addcdiv, linalg.matrix_norm, full_like, unsqueeze_, fmod`  
**Output shape:** `[4, 8]`

```
CPU: [-0.0520, -0.0549, -0.0619, -0.0689, -0.0718, -0.0689, -0.0619, -0.0549,
     -0.0105, -0.0270, -0.0307, -0.0194,  0.0003,  0.0168,  0.0205,  0.0092,
      0.1237,  0.0888,  0.0047, -0.0795, -0.1143, -0.0795,  0.0047,  0.0888,
     -0.0105,  0.0092,  0.0205,  0.0168,  0.0003, -0.0194, -0.0307, -0.0270]
GPU: [-0.0602, -0.0607, -0.0619, -0.0631, -0.0636, -0.0631, -0.0619, -0.0607,
      0.0218, -0.0300, -0.0672, -0.0681, -0.0320,  0.0198,  0.0570,  0.0579,
      0.0850,  0.0615,  0.0047, -0.0521, -0.0757, -0.0521,  0.0047,  0.0615,
      0.0218,  0.0579,  0.0570,  0.0198, -0.0320, -0.0681, -0.0672, -0.0300]
```

Repro: `python3 pytorch/bug_inconsistent_m0088_mut3_20260416_200741.repro.py`

---

## Observations

- All 4 bugs involve **BatchNorm1d** — normalization statistics (mean/variance) are computed differently on CPU (Welford algorithm) vs GPU (cuDNN kernel), causing outputs to diverge under large-magnitude inputs.
- **scale_large** mutation (3 of 4 bugs) and **mask** mutation (1 bug) are the triggers. Large or sparse inputs amplify the BatchNorm statistics difference through subsequent nonlinear layers.
- PT-4 (m0088) is the strongest divergence (rel_err=4459×): `linalg.matrix_norm` + `fmod` compound the initial BatchNorm difference into a qualitatively different output distribution (CPU shows more structured variation; GPU converges to near-uniform values).
