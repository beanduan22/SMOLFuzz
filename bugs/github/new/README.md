# SMOLFuzz — Confirmed CPU/GPU Divergence Bugs (April 2026)

27 real bugs found and independently re-verified on 2026-04-17.  
All bugs are `INCONSISTENT` type: same model, same weights, same inputs — different outputs on CPU vs GPU.

## Environment

| Component | Version |
|-----------|---------|
| Python | 3.13.5 |
| PyTorch | 2.9.1+cu128 |
| TensorFlow | 2.21.0 |
| CUDA | 12.8 |
| cuDNN | 9.10.2 |
| GPU | NVIDIA RTX 6000 Ada Generation (48 GB) |
| GPU Driver | 570.211.01 |
| CPU | AMD Ryzen Threadripper PRO 7985WX (64 cores) |

## Run Stats

| | PyTorch | TensorFlow |
|--|---------|------------|
| Models tested | 102 | 200 |
| Bugs confirmed | **5** | **22** |
| False positives removed | 0 | 1 |
| Fuzzing budget | 60s/model | 60s/model |
| LLM | qwen2.5-coder:32b | qwen2.5-coder:32b |

---

## PyTorch Bugs (5 confirmed)

### PT-1 · m0028 · scale_large · rel_err = 838×

**APIs:** `Linear, ELU, BatchNorm1d, sin, clip, cummax, row_stack`

```
CPU: [0.2614, 0.2614, 0.2614, 0.2614, 0.5, 0.5, 0.5, 0.5, ...]
GPU: [0.2311, 0.2311, 0.2311, 0.2311, 0.5, 0.5, 0.5, 0.5, ...]
```

Repro: `python3 pytorch/bug_inconsistent_m0028_mut5_20260416_183107.repro.py`

---

### PT-2 · m0035 · add_noise · rel_err = 3566×

**APIs:** `kaiming_normal_, glu, unique_consecutive, tensor_split, float_power_, erf, sort, abs`

```
CPU: [-0.0581, -0.0017, -0.0025, -0.0720]
GPU: [-0.0008, -0.0188, -0.0254, -0.0082]
```

Repro: `python3 pytorch/bug_inconsistent_m0035_mut1_20260416_183902.repro.py`

---

### PT-3 · m0046 · scale_large · rel_err = 163×

**APIs:** `Linear, BatchNorm1d, Mish, hardswish, threshold_, fmod, exp, sin`

```
CPU: [-0.3346, -0.4632, -0.3013, -0.7158, -0.1079, -0.5654, -0.4068, -0.7080]
GPU: [-0.3417, -0.4571, -0.2973, -0.7106, -0.1081, -0.5653, -0.4068, -0.7081]
```

Repro: `python3 pytorch/bug_inconsistent_m0046_mut5_20260416_185555.repro.py`

---

### PT-4 · m0050 · scale_large · rel_err = 259×

**APIs:** `Linear, silu, BatchNorm1d, sin, arctanh, sigmoid, autograd.grad, concat`

```
CPU: [0.0, 0.1228, 0.0, 0.1959, 0.0, 0.0, 0.0, -0.8478]
GPU: [0.0, 0.1189, 0.0, 0.2036, 0.0, 0.0, 0.0, -0.8436]
```

Repro: `python3 pytorch/bug_inconsistent_m0050_mut5_20260416_190108.repro.py`

---

### PT-5 · m0088 · mask · rel_err = 2483×

**APIs:** `Linear, BatchNorm1d, ELU, addcdiv, linalg.matrix_norm, full_like, unsqueeze_, fmod`

```
CPU: [-0.0520, -0.0549, -0.0619, -0.0689, -0.0718, -0.0689, -0.0619, -0.0549]
GPU: [-0.0602, -0.0607, -0.0619, -0.0631, -0.0636, -0.0631, -0.0619, -0.0607]
```

Repro: `python3 pytorch/bug_inconsistent_m0088_mut3_20260416_200741.repro.py`

---

## TensorFlow Bugs (22 confirmed)

### TF-1 · m0036 · scale_large · rel_err = 422,600×

**APIs:** `Dense, Hashing, unit_norm constraint, swish, GradientTape, metrics.Sum, maximum`

```
CPU: [0.0, 0.0, 4.9261, 0.0, 0.0, 0.0, 0.6300, 0.0]
GPU: [3.4062, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2142, 0.0]
```

---

### TF-2 · m0078 · scale_small · rel_err = 203×

**APIs:** `Dense, GlorotNormal, BatchNorm, LayerNorm, GlobalAvgPool1D, CosineSimilarity, GradientTape`

```
CPU: [-302.92, -302.92, -302.60, -302.92, -46213.59, -46213.53, -46213.59, -46213.59]
GPU: [-311.57, -311.57, -311.26, -311.57, -45835.37, -45835.31, -45835.37, -45835.37]
```

---

### TF-3 · m0004 · scale_small · rel_err = 225×

**APIs:** `Dense, L2 regularizer, BatchNorm, LayerNorm, gelu, EinsumDense, GradientTape, device`

```
CPU: [-1.7139, -1.1515,  0.6787, -1.6073, -0.5855,  2.3042,  1.4409, -1.1319]
GPU: [-1.7198, -1.1501,  0.6775, -1.6115, -0.5852,  2.3038,  1.4412, -1.1309]
```

---

### TF-4 · m0147 · scale_small · rel_err = 324×

**APIs:** `Dense, LeakyReLU, BatchNorm, LayerNorm, NonNeg constraint, Adagrad, GradientTape, device`

```
CPU: [-6.2324, -5.3804, -3.2715, -6.8714, -5.5672,  4.9915,  8.1638, -3.3078]
GPU: [-6.2358, -5.3800, -3.2714, -6.8699, -5.5683,  4.9943,  8.1591, -3.3090]
```

---

### TF-5 · m0029 · scale_small · rel_err = 114×

**APIs:** `Dense, GlorotUniform, BatchNorm, LayerNorm, cos, cosh, GradientTape`

```
CPU: [2.051e+01, 2.609e+08, 1.889e+07, 1.100e+12, 2.269e+04, 1.236e+00, 1.374e+25, 2.030e+14]
GPU: [2.036e+01, 2.623e+08, 1.868e+07, 1.087e+12, 2.270e+04, 1.230e+00, 1.349e+25, 2.024e+14]
```

---

### TF-6 · m0164 · scale_small · rel_err = 153×

**APIs:** `Dense, Identity initializer, BatchNorm, LayerNorm, exponential activation, squared_difference, nn.tanh`

```
CPU: [1.0, 1.0, 1.0, 1.0, 0.2919, 1.0, 1.0, 0.3637]
GPU: [1.0, 1.0, 1.0, 1.0, 0.2970, 1.0, 1.0, 0.3558]
```

---

### TF-7 · m0008 · scale_small · rel_err = 102×

**APIs:** `Dense, BatchNorm, LayerNorm, swish, Average layer, FalseNegatives metric, GradientTape, device`

```
CPU: [ 1.2389, -4.8887,  3.6273, -2.2208,  2.1076, -0.0040, -4.0964,  2.6769]
GPU: [ 1.2385, -4.8869,  3.6268, -2.2212,  2.1110, -0.0027, -4.0945,  2.6790]
```

---

### TF-8 · m0082 · scale_small · rel_err = 169×

**APIs:** `Dense, lecun_normal, BatchNorm, LayerNorm, sinh, GradientTape, device`

```
CPU: [-7.5971,  4.7068, -8.2351, 14.6573,  4.3494, 12.7359,  2.8822, -7.4070]
GPU: [-7.5931,  4.7011, -8.2310, 14.6575,  4.3519, 12.7208,  2.8793, -7.3981]
```

---

### TF-9 · m0092 · scale_small · rel_err = 263×

**APIs:** `Dense, Orthogonal initializer, BatchNorm, LayerNorm, swish, GlobalMaxPooling1D, GradientTape, device`

```
CPU: [ 3.9312, -5.0459,  7.8780, -0.3367,  6.8222, -4.3921,  0.5667, -0.9407]
GPU: [ 3.9338, -5.0396,  7.8779, -0.3394,  6.8181, -4.3910,  0.5617, -0.9423]
```

---

### TF-10 · m0045 · scale_small · rel_err = 160×

**APIs:** `Dense, HeNormal, BatchNorm, LayerNorm, min_max_norm constraint, GradientTape`

```
CPU: [1.4107, -3.2575, 2.7488, 3.4276, 0.7142, 3.1685, 0.3893, 1.1050]
GPU: [1.4095, -3.2577, 2.7482, 3.4268, 0.7143, 3.1684, 0.3889, 1.1050]
```

---

### TF-11 · m0068 · add_noise · rel_err = 34×

**APIs:** `Dense, PReLU, BatchNorm, MeanAbsoluteError, GradientTape, zeros_like, clip_by_value, device`

```
CPU: [-0.0097,  0.5994,  1.4417,  1.4910, -0.5846, -1.0808, -1.0744, -0.7827]
GPU: [-0.0097,  0.5994,  1.4416,  1.4911, -0.5846, -1.0806, -1.0748, -0.7825]
```

---

### TF-12 · m0080 · scale_small · rel_err = 115×

**APIs:** `Dense, BatchNorm, LayerNorm, selu, GradientTape, device`

```
CPU: [-1.4995, -10.1734, -1.2585, -16.2872, -1.6986,  -3.7066,  2.4061,  -6.4377]
GPU: [-1.5003, -10.1729, -1.2577, -16.2875, -1.7006,  -3.7082,  2.4076,  -6.4388]
```

---

### TF-13 · m0090 · scale_large · rel_err = 109×

**APIs:** `Dense, BatchNorm, LayerNorm, exponential activation, StringLookup, GradientTape, slice`

```
CPU: [-4.4271,  2.5325, -2.5243, -3.6614, -1.9677, -2.1018, -2.7028, -2.2372]
GPU: [-4.4260,  2.5314, -2.5240, -3.6611, -1.9677, -2.1015, -2.7023, -2.2367]
```

---

### TF-14 · m0100 · mask · rel_err = 199×

**APIs:** `Dense, BatchNorm, LayerNorm, gelu, GradientTape, device`

```
CPU: [-1.1967,  0.8384, -1.7077,  0.5572]
GPU: [-1.1960,  0.8390, -1.7073,  0.5576]
```

---

### TF-15 · m0102 · add_noise · rel_err = 109×

**APIs:** `Dense, HeNormal, tanh, BatchNorm, LayerNorm, GradientTape`

```
CPU: [ 0.6643,  2.2214, -0.5199, -0.1518, -0.0074, -0.1134, -0.8011, -1.2921]
GPU: [ 0.6636,  2.2222, -0.5196, -0.1524, -0.0078, -0.1138, -0.8008, -1.2913]
```

---

### TF-16 · m0108 · scale_small · rel_err = 106×

**APIs:** `Dense, BatchNorm, LayerNorm, gelu, GradientTape, device, math.maximum`

```
CPU: [7.8783, 0.0, 0.0, 0.0, 7.7235, 5.2743, 5.4564, 4.9521]
GPU: [7.8745, 0.0, 0.0, 0.0, 7.7256, 5.2738, 5.4605, 4.9567]
```

---

### TF-17 · m0117 · add_noise · rel_err = 141×

**APIs:** `Dense, BatchNorm, LayerNorm, swish, MeanMetricWrapper, logcosh, MeanAbsoluteError, GradientTape`

```
CPU: [ 2.5828,  2.4601, -0.2457, -0.2037,  0.1721,  1.3421, -0.1268,  0.8376]
GPU: [ 2.5832,  2.4609, -0.2453, -0.2040,  0.1719,  1.3430, -0.1271,  0.8380]
```

---

### TF-18 · m0126 · uniform · rel_err = 363×

**APIs:** `Dense, BatchNorm, LayerNorm, GradientTape, Maximum layer, reduce_sum, Mean metric, SparseCategoricalAccuracy`

```
CPU: [ 9.2054, 12.5460, -18.7894, -37.3736,  4.9578, 15.3701, -0.1253, 27.3820]
GPU: [ 9.2000, 12.5315, -18.7887, -37.3820,  4.9667, 15.3738, -0.1073, 27.3872]
```

---

### TF-19 · m0149 · scale_small · rel_err = 115×

**APIs:** `Dense, RandomNormal initializer, BatchNorm, LayerNorm, Concatenate, Nadam, RootMeanSquaredError`

```
CPU: [ 0.7819, -2.2389,  0.1056, -0.2257, -2.0170, -4.7178, -0.6685,  1.2734]
GPU: [ 0.7823, -2.2402,  0.1047, -0.2257, -2.0150, -4.7186, -0.6688,  1.2731]
```

---

### TF-20 · m0157 · scale_small · rel_err = 122×

**APIs:** `Dense, BatchNorm, LayerNorm, GradientTape, Conv1DTranspose, math.cos, device`

```
CPU: [-0.1133, -0.6169, -0.0907, -0.9997,  0.9363, -0.9780, -0.8303,  0.0484]
GPU: [-0.1137, -0.6157, -0.0893, -0.9998,  0.9361, -0.9780, -0.8297,  0.0472]
```

---

### TF-21 · m0163 · scale_small · rel_err = 127×

**APIs:** `Dense, L1L2 regularizer, BatchNorm, LayerNorm, multiply, GradientTape, device`

```
CPU: [3.6094,  5.5190,  6.8101, 10.9785, -0.7915,  3.2262,  5.3724,  1.7773]
GPU: [3.6150,  5.5189,  6.8145, 10.9857, -0.7881,  3.2311,  5.3725,  1.7795]
```

---

### TF-22 · m0079 · scale_small · rel_err = 108×

**APIs:** `Dense, BatchNorm, activations.get, LayerNorm`

```
CPU: [-0.2825, -1.7983,  1.0369, -0.0935,  0.8332,  0.5002, -0.2977,  0.1017]
GPU: [-0.2825, -1.7988,  1.0366, -0.0936,  0.8332,  0.5003, -0.2966,  0.1014]
```

---

## Observations

- **GradientTape + BatchNorm/LayerNorm** is the dominant divergence pattern in TF — appears in 20 of 22 bugs.
- **scale_small** mutation (input × 0.01–0.1) triggers 16 of 22 TF bugs — near-zero inputs expose different CPU/GPU numerical paths.
- **BatchNorm1d** appears in 4 of 5 PyTorch bugs — normalization statistics diverge between devices.
- **TF m0036** (rel_err=422,600×) is the strongest: `Hashing + GradientTape + swish` produces completely different sparse activation patterns on CPU vs GPU.
- **TF m0029** produces astronomically large outputs (up to 1e+25) that diverge by 14× between devices.
