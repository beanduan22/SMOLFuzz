# SMOLFuzz — Confirmed CPU/GPU Divergence Bugs (April 2026)

27 real bugs found and independently re-verified on 2026-04-17.  
All bugs are `INCONSISTENT` type: the model produces numerically different outputs on CPU vs GPU with identical weights and inputs.

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

| Metric | PyTorch | TensorFlow |
|--------|---------|------------|
| Models tested | 102 | 200 |
| Bugs reported | 5 | 23 |
| Verified real | **5** | **22** |
| False positives | 0 | 1 (m0154 — forced CPU device in call) |
| Clean (no bug) | 32 | 105 |
| Synthesis failures | ~66 | 65 |
| Fuzzing budget | 60s/model | 60s/model |
| LLM | qwen2.5-coder:32b | qwen2.5-coder:32b |

## Verification Method

Each bug was re-run independently on 2026-04-17 with GPU memory free.  
PyTorch bugs: executed via `*.repro.py` (self-contained scripts with saved inputs).  
TensorFlow bugs: re-executed using the same CPU/GPU wrapper as the fuzzer, with saved `.inputs.npy`.  
Criterion: `rel_err = max|cpu - gpu| / (atol + rtol * |cpu|) > 10×` with `rtol=1e-4, atol=1e-5`, nondet floor = 0.

## PyTorch Bugs (5/5 confirmed)

| Bug ID | Mutation | APIs | Verified rel\_err |
|--------|----------|------|------------------|
| m0028 | scale\_large | Linear, ELU, BatchNorm1d, sin, clip, cummax, row\_stack | **838×** |
| m0035 | add\_noise | kaiming\_normal\_, glu, unique\_consecutive, tensor\_split, float\_power\_, erf, sort | **3566×** |
| m0046 | scale\_large | Linear, BatchNorm1d, Mish, hardswish, threshold\_, fmod, exp, sin | **163×** |
| m0050 | scale\_large | Linear, silu, BatchNorm1d, sin, arctanh, sigmoid, autograd.grad, concat | **259×** |
| m0088 | mask | Linear, BatchNorm1d, ELU, addcdiv, linalg.matrix\_norm, full\_like, unsqueeze\_, fmod | **2483×** |

### PyTorch Repro Output

```
# m0028
output[0] dtype=torch.float32 rel_err=8.380e+02 shape=[4, 4]

# m0035
output[0] dtype=torch.float32 rel_err=3.566e+03 shape=[4, 1]

# m0046
output[0] dtype=torch.float32 rel_err=1.631e+02 shape=[4, 4]

# m0050
output[0] dtype=torch.float32 rel_err=2.587e+02 shape=[64]

# m0088
output[0] dtype=torch.float32 rel_err=2.483e+03 shape=[4, 8]
```

Run any repro with:
```bash
cd pytorch/
python3 bug_inconsistent_m0028_mut5_20260416_183107.repro.py
```

## TensorFlow Bugs (22/23 confirmed)

| Bug ID | Mutation | Key APIs | Verified rel\_err |
|--------|----------|----------|------------------|
| m0036 | scale\_large | Dense, Hashing, unit\_norm, swish, GradientTape | **422,600×** |
| m0147 | scale\_small | Dense, LeakyReLU, BatchNorm, LayerNorm, NonNeg, Adagrad, GradientTape | **324×** |
| m0164 | scale\_small | Dense, Identity init, BatchNorm, LayerNorm, exponential, squared\_difference | **153×** |
| m0029 | scale\_small | Dense, GlorotUniform, BatchNorm, LayerNorm, cos, cosh, GradientTape | **114×** |
| m0078 | scale\_small | Dense, GlorotNormal, BatchNorm, LayerNorm, GlobalAvgPool1D, CosineSimilarity | **203×** |
| m0004 | scale\_small | Dense, L2 reg, BatchNorm, LayerNorm, gelu, EinsumDense, GradientTape | **225×** |
| m0157 | scale\_small | Dense, BatchNorm, LayerNorm, GradientTape, Conv1DTranspose, cos | **97×** |
| m0082 | scale\_small | Dense, lecun\_normal, BatchNorm, LayerNorm, sinh, GradientTape | **54×** |
| m0092 | scale\_small | Dense, Orthogonal, BatchNorm, LayerNorm, swish, GlobalMaxPool1D | **43×** |
| m0126 | uniform | Dense, BatchNorm, LayerNorm, GradientTape, Maximum, reduce\_sum | **13×** |
| m0163 | scale\_small | Dense, L1L2 reg, BatchNorm, LayerNorm, multiply, GradientTape | **44×** |
| m0080 | scale\_small | Dense, BatchNorm, LayerNorm, selu, GradientTape | **44×** |
| m0045 | scale\_small | Dense, HeNormal, BatchNorm, LayerNorm, min\_max\_norm, GradientTape | **45×** |
| m0008 | scale\_small | Dense, BatchNorm, LayerNorm, swish, Average, FalseNegatives | **41×** |
| m0102 | add\_noise | Dense, HeNormal, tanh, BatchNorm, LayerNorm, GradientTape | **39×** |
| m0117 | add\_noise | Dense, BatchNorm, LayerNorm, swish, logcosh, MeanAbsoluteError | **48×** |
| m0068 | add\_noise | Dense, PReLU, BatchNorm, MeanAbsoluteError, GradientTape, clip\_by\_value | **34×** |
| m0079 | scale\_small | Dense, BatchNorm, activations.get, LayerNorm | **21×** |
| m0090 | scale\_large | Dense, BatchNorm, LayerNorm, exponential, StringLookup, slice | **57×** |
| m0100 | mask | Dense, BatchNorm, LayerNorm, gelu, GradientTape | **24×** |
| m0108 | scale\_small | Dense, BatchNorm, LayerNorm, gelu, GradientTape, math.maximum | **18×** |
| m0149 | scale\_small | Dense, RandomNormal, BatchNorm, LayerNorm, Concatenate, Nadam | **22×** |

### TensorFlow Verification Output

```
[tf_m0004 scale_small] rel_err=2.253e+02  nondet=0.000e+00  → CONFIRMED
[tf_m0008 scale_small] rel_err=4.116e+01  nondet=0.000e+00  → CONFIRMED
[tf_m0029 scale_small] rel_err=1.136e+02  nondet=0.000e+00  → CONFIRMED
[tf_m0036 scale_large] rel_err=4.226e+05  nondet=0.000e+00  → CONFIRMED
[tf_m0045 scale_small] rel_err=4.528e+01  nondet=0.000e+00  → CONFIRMED
[tf_m0068 add_noise]   rel_err=3.385e+01  nondet=0.000e+00  → CONFIRMED
[tf_m0078 scale_small] rel_err=2.029e+02  nondet=0.000e+00  → CONFIRMED
[tf_m0079 scale_small] rel_err=2.118e+01  nondet=0.000e+00  → CONFIRMED
[tf_m0080 scale_small] rel_err=4.441e+01  nondet=0.000e+00  → CONFIRMED
[tf_m0082 scale_small] rel_err=5.401e+01  nondet=0.000e+00  → CONFIRMED
[tf_m0090 scale_large] rel_err=5.699e+01  nondet=0.000e+00  → CONFIRMED
[tf_m0092 scale_small] rel_err=4.307e+01  nondet=0.000e+00  → CONFIRMED
[tf_m0100 mask]        rel_err=2.390e+01  nondet=0.000e+00  → CONFIRMED
[tf_m0102 add_noise]   rel_err=3.931e+01  nondet=0.000e+00  → CONFIRMED
[tf_m0108 scale_small] rel_err=1.751e+01  nondet=0.000e+00  → CONFIRMED
[tf_m0117 add_noise]   rel_err=4.835e+01  nondet=0.000e+00  → CONFIRMED
[tf_m0126 uniform]     rel_err=1.348e+01  nondet=0.000e+00  → CONFIRMED
[tf_m0147 scale_small] rel_err=3.235e+02  nondet=0.000e+00  → CONFIRMED
[tf_m0149 scale_small] rel_err=2.192e+01  nondet=0.000e+00  → CONFIRMED
[tf_m0157 scale_small] rel_err=9.670e+01  nondet=0.000e+00  → CONFIRMED
[tf_m0163 scale_small] rel_err=4.372e+01  nondet=0.000e+00  → CONFIRMED
[tf_m0164 scale_small] rel_err=1.529e+02  nondet=0.000e+00  → CONFIRMED
```

## False Positive Removed

**tf_m0154** (scale\_small): The model used `tf.device('/CPU:0')` inside `call()` for its entire output path, meaning both the CPU model and GPU model executed the critical computation on CPU. No true CPU/GPU divergence was possible. Removed from the bug set.

## Observations

- **GradientTape + BatchNorm/LayerNorm** appears in the majority of TF bugs — gradient computation through normalization layers is a consistent divergence trigger.
- **scale\_small** mutation (shrinking input by 0.01–0.1×) triggers 16 of the 22 TF bugs, suggesting near-zero input regimes expose numerical path differences.
- **BatchNorm1d** appears in 4 of 5 PyTorch bugs — normalization statistics computed differently on CPU vs GPU.
- TF m0036 (rel\_err=422,600×) is the strongest divergence, involving `tf.keras.layers.Hashing` combined with GradientTape.
