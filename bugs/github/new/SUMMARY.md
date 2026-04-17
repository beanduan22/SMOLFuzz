# smolfuzz run — real-bug triage

- Finalized at: 2026-04-16 22:31:09
- PT models started: 102
- PT clean (60s fuzzed, no bug): 32
- TF models started: 200
- TF clean (60s fuzzed, no bug): 105
- Real bugs kept: **27**
- Flagged but rejected as FP on re-triage: 0

## FP filters applied

- `BUG_MARGIN`: rel_err must exceed `100.0×` dtype tolerance
- Rejected if `nondet_floor >= 100.0` (inherently non-deterministic model)
- Rejected if `rel_err < 10.0 × nondet_floor`
- Infrastructure errors (CUBLAS/CUDA OOM/driver) filtered
- GPU-side `Model().to(device)` construction crashes filtered (contention, not library)
- Permutation-invariant API outputs (eig/svd/qr/sort/…) require sorted-magnitude divergence

## Real bugs

### pytorch bug_inconsistent_m0028_mut5_20260416_183107
- Type: `inconsistent`
- Mutation: `scale_large`
- Reason: rel_err=838.0 > 100.0× (nondet=0.0, mut=scale_large)
- Detail: `output[0]: rel_err=8.380e+02 > 1.0 dtype=torch.float32 shape=[4, 4] rtol=1e-04 atol=1e-05 nondet_floor=0.000e+00`
- APIs used: torch.nn.Linear, torch.nn.functional.elu, torch.nn.BatchNorm1d, torch.sin, torch.clip, torch.cummax, torch.Tensor.max, torch.row_stack

### pytorch bug_inconsistent_m0035_mut1_20260416_183902
- Type: `inconsistent`
- Mutation: `add_noise`
- Reason: rel_err=3710.0 > 100.0× (nondet=0.0, mut=add_noise)
- Detail: `output[0]: rel_err=3.710e+03 > 1.0 dtype=torch.float32 shape=[4, 1] rtol=1e-04 atol=1e-05 nondet_floor=0.000e+00`
- APIs used: torch.nn.init.kaiming_normal_, torch.nn.functional.glu, torch.unique_consecutive, torch.Tensor.tensor_split, float_power_, torch.erf, torch.Tensor.sort, torch.abs

### pytorch bug_inconsistent_m0046_mut5_20260416_185555
- Type: `inconsistent`
- Mutation: `scale_large`
- Reason: rel_err=163.1 > 100.0× (nondet=0.0, mut=scale_large)
- Detail: `output[0]: rel_err=1.631e+02 > 1.0 dtype=torch.float32 shape=[4, 4] rtol=1e-04 atol=1e-05 nondet_floor=0.000e+00`
- APIs used: torch.nn.Linear, torch.nn.BatchNorm1d, torch.nn.Mish, torch.nn.functional.hardswish, torch.nn.functional.threshold_, torch.fmod, torch.exp, torch.sin

### pytorch bug_inconsistent_m0050_mut5_20260416_190108
- Type: `inconsistent`
- Mutation: `scale_large`
- Reason: rel_err=258.7 > 100.0× (nondet=0.0, mut=scale_large)
- Detail: `output[0]: rel_err=2.587e+02 > 1.0 dtype=torch.float32 shape=[64] rtol=1e-04 atol=1e-05 nondet_floor=0.000e+00`
- APIs used: torch.nn.Linear, torch.nn.functional.silu, torch.nn.BatchNorm1d, torch.sin, torch.arctanh, torch.sigmoid, torch.autograd.grad, torch.concat

### pytorch bug_inconsistent_m0088_mut3_20260416_200741
- Type: `inconsistent`
- Mutation: `mask`
- Reason: rel_err=4459.0 > 100.0× (nondet=0.0, mut=mask)
- Detail: `output[0]: rel_err=4.459e+03 > 1.0 dtype=torch.float32 shape=[4, 8] rtol=1e-04 atol=1e-05 nondet_floor=0.000e+00`
- APIs used: torch.nn.Linear, torch.nn.BatchNorm1d, torch.nn.ELU, torch.addcdiv, torch.linalg.matrix_norm, torch.full_like, torch.Tensor.unsqueeze_, torch.fmod

### tensorflow bug_inconsistent_m0004_scale_small_20260416_174618
- Type: `inconsistent`
- Mutation: `scale_small`
- Reason: rel_err=181.1 > 100.0× (nondet=0.0, mut=scale_small)
- Detail: `rel_err=1.811e+02 > 100.0×tol  rtol=1e-04 atol=1e-05 nondet_floor=0.000e+00 dtype=float32 n=16`
- APIs used: tf.keras.layers.Dense, tf.keras.regularizers.L2, tf.keras.layers.BatchNormalization, tf.keras.layers.LayerNormalization, tf.nn.gelu, tf.keras.layers.EinsumDense, tf.GradientTape, tf.device

### tensorflow bug_inconsistent_m0008_scale_small_20260416_175223
- Type: `inconsistent`
- Mutation: `scale_small`
- Reason: rel_err=102.8 > 100.0× (nondet=0.0, mut=scale_small)
- Detail: `rel_err=1.028e+02 > 100.0×tol  rtol=1e-04 atol=1e-05 nondet_floor=0.000e+00 dtype=float32 n=32`
- APIs used: tf.keras.layers.Dense, tf.keras.layers.BatchNormalization, tf.keras.layers.LayerNormalization, tf.keras.activations.swish, tf.keras.layers.Average, tf.GradientTape, tf.device, tf.keras.metrics.FalseNegatives

### tensorflow bug_inconsistent_m0029_scale_small_20260416_183049
- Type: `inconsistent`
- Mutation: `scale_small`
- Reason: rel_err=114.9 > 100.0× (nondet=0.0, mut=scale_small)
- Detail: `rel_err=1.149e+02 > 100.0×tol  rtol=1e-04 atol=1e-05 nondet_floor=0.000e+00 dtype=float32 n=32`
- APIs used: tf.keras.layers.Dense, tf.keras.initializers.GlorotUniform, tf.keras.layers.BatchNormalization, tf.keras.layers.LayerNormalization, tf.keras.layers.Activation, tf.cos, tf.math.cosh, tf.GradientTape

### tensorflow bug_inconsistent_m0036_scale_large_20260416_184504
- Type: `inconsistent`
- Mutation: `scale_large`
- Reason: rel_err=236000.0 > 100.0× (nondet=0.0, mut=scale_large)
- Detail: `rel_err=2.360e+05 > 100.0×tol  rtol=1e-04 atol=1e-05 nondet_floor=0.000e+00 dtype=float32 n=32`
- APIs used: tf.keras.layers.Dense, tf.keras.initializers.RandomUniform, tf.keras.layers.Hashing, tf.keras.constraints.unit_norm, tf.nn.swish, tf.GradientTape, tf.metrics.Sum, tf.maximum

### tensorflow bug_inconsistent_m0045_scale_small_20260416_190139
- Type: `inconsistent`
- Mutation: `scale_small`
- Reason: rel_err=160.3 > 100.0× (nondet=0.0, mut=scale_small)
- Detail: `rel_err=1.603e+02 > 100.0×tol  rtol=1e-04 atol=1e-05 nondet_floor=0.000e+00 dtype=float32 n=32`
- APIs used: tf.keras.layers.Dense, tf.keras.initializers.HeNormal, tf.keras.layers.BatchNormalization, tf.keras.layers.LayerNormalization, tf.keras.layers.Activation, tf.keras.constraints.min_max_norm, tf.GradientTape

### tensorflow bug_inconsistent_m0068_add_noise_20260416_194417
- Type: `inconsistent`
- Mutation: `add_noise`
- Reason: rel_err=210.6 > 100.0× (nondet=0.0, mut=add_noise)
- Detail: `rel_err=2.106e+02 > 100.0×tol  rtol=1e-04 atol=1e-05 nondet_floor=0.000e+00 dtype=float32 n=32`
- APIs used: tf.keras.layers.Dense, tf.keras.layers.PReLU, tf.keras.layers.BatchNormalization, tf.keras.losses.MeanAbsoluteError, tf.GradientTape, tf.zeros_like, tf.clip_by_value, tf.device

### tensorflow bug_inconsistent_m0078_scale_small_20260416_200343
- Type: `inconsistent`
- Mutation: `scale_small`
- Reason: rel_err=690.3 > 100.0× (nondet=0.0, mut=scale_small)
- Detail: `rel_err=6.903e+02 > 100.0×tol  rtol=1e-04 atol=1e-05 nondet_floor=0.000e+00 dtype=float32 n=16`
- APIs used: tf.keras.layers.Dense, tf.initializers.GlorotNormal, tf.keras.layers.BatchNormalization, tf.keras.layers.LayerNormalization, tf.keras.layers.Activation, tf.keras.layers.GlobalAvgPool1D, tf.metrics.CosineSimilarity, tf.GradientTape

### tensorflow bug_inconsistent_m0079_scale_small_20260416_200526
- Type: `inconsistent`
- Mutation: `scale_small`
- Reason: rel_err=108.7 > 100.0× (nondet=0.0, mut=scale_small)
- Detail: `rel_err=1.087e+02 > 100.0×tol  rtol=1e-04 atol=1e-05 nondet_floor=0.000e+00 dtype=float32 n=32`
- APIs used: tf.keras.layers.Dense, tf.keras.layers.BatchNormalization, tf.keras.activations.get, tf.keras.layers.LayerNormalization

### tensorflow bug_inconsistent_m0080_scale_small_20260416_200659
- Type: `inconsistent`
- Mutation: `scale_small`
- Reason: rel_err=115.3 > 100.0× (nondet=0.0, mut=scale_small)
- Detail: `rel_err=1.153e+02 > 100.0×tol  rtol=1e-04 atol=1e-05 nondet_floor=0.000e+00 dtype=float32 n=16`
- APIs used: tf.keras.layers.Dense, tf.keras.layers.BatchNormalization, tf.keras.layers.LayerNormalization, tf.keras.activations.selu, tf.GradientTape, tf.device

### tensorflow bug_inconsistent_m0082_scale_small_20260416_201012
- Type: `inconsistent`
- Mutation: `scale_small`
- Reason: rel_err=169.8 > 100.0× (nondet=0.0, mut=scale_small)
- Detail: `rel_err=1.698e+02 > 100.0×tol  rtol=1e-04 atol=1e-05 nondet_floor=0.000e+00 dtype=float32 n=32`
- APIs used: tf.keras.layers.Dense, tf.keras.initializers.lecun_normal, tf.keras.layers.BatchNormalization, tf.keras.layers.LayerNormalization, tf.keras.layers.Activation, tf.sinh, tf.GradientTape, tf.device

### tensorflow bug_inconsistent_m0090_scale_large_20260416_202258
- Type: `inconsistent`
- Mutation: `scale_large`
- Reason: rel_err=109.6 > 100.0× (nondet=0.0, mut=scale_large)
- Detail: `rel_err=1.096e+02 > 100.0×tol  rtol=1e-04 atol=1e-05 nondet_floor=0.000e+00 dtype=float32 n=32`
- APIs used: tf.keras.layers.Dense, tf.keras.layers.BatchNormalization, tf.keras.layers.LayerNormalization, tf.keras.activations.exponential, tf.keras.layers.StringLookup, tf.GradientTape, tf.slice

### tensorflow bug_inconsistent_m0092_scale_small_20260416_202454
- Type: `inconsistent`
- Mutation: `scale_small`
- Reason: rel_err=263.1 > 100.0× (nondet=0.0, mut=scale_small)
- Detail: `rel_err=2.631e+02 > 100.0×tol  rtol=1e-04 atol=1e-05 nondet_floor=0.000e+00 dtype=float32 n=16`
- APIs used: tf.keras.layers.Dense, tf.initializers.Orthogonal, tf.keras.layers.BatchNormalization, tf.keras.layers.LayerNormalization, tf.keras.activations.swish, tf.GradientTape, tf.device, tf.keras.layers.GlobalMaxPooling1D

### tensorflow bug_inconsistent_m0100_mask_20260416_203852
- Type: `inconsistent`
- Mutation: `mask`
- Reason: rel_err=198.9 > 100.0× (nondet=0.0, mut=mask)
- Detail: `rel_err=1.989e+02 > 100.0×tol  rtol=1e-04 atol=1e-05 nondet_floor=0.000e+00 dtype=float32 n=4`
- APIs used: tf.keras.layers.Dense, tf.keras.layers.BatchNormalization, tf.keras.layers.LayerNormalization, tf.keras.activations.gelu, tf.GradientTape, tf.device

### tensorflow bug_inconsistent_m0102_add_noise_20260416_204116
- Type: `inconsistent`
- Mutation: `add_noise`
- Reason: rel_err=109.4 > 100.0× (nondet=0.0, mut=add_noise)
- Detail: `rel_err=1.094e+02 > 100.0×tol  rtol=1e-04 atol=1e-05 nondet_floor=0.000e+00 dtype=float32 n=32`
- APIs used: tf.keras.layers.Dense, tf.keras.initializers.HeNormal, tf.tanh, tf.keras.layers.BatchNormalization, tf.keras.layers.LayerNormalization, tf.GradientTape

### tensorflow bug_inconsistent_m0108_scale_small_20260416_204927
- Type: `inconsistent`
- Mutation: `scale_small`
- Reason: rel_err=106.3 > 100.0× (nondet=0.0, mut=scale_small)
- Detail: `rel_err=1.063e+02 > 100.0×tol  rtol=1e-04 atol=1e-05 nondet_floor=0.000e+00 dtype=float32 n=32`
- APIs used: tf.keras.layers.Dense, tf.keras.layers.BatchNormalization, tf.keras.layers.LayerNormalization, tf.nn.gelu, tf.GradientTape, tf.device, tf.math.maximum

### tensorflow bug_inconsistent_m0117_add_noise_20260416_210321
- Type: `inconsistent`
- Mutation: `add_noise`
- Reason: rel_err=141.1 > 100.0× (nondet=0.0, mut=add_noise)
- Detail: `rel_err=1.411e+02 > 100.0×tol  rtol=1e-04 atol=1e-05 nondet_floor=0.000e+00 dtype=float32 n=32`
- APIs used: tf.keras.layers.Dense, tf.keras.layers.BatchNormalization, tf.keras.layers.LayerNormalization, tf.keras.activations.swish, tf.metrics.MeanMetricWrapper, tf.keras.losses.logcosh, tf.metrics.MeanAbsoluteError, tf.GradientTape

### tensorflow bug_inconsistent_m0126_uniform_20260416_211538
- Type: `inconsistent`
- Mutation: `uniform`
- Reason: rel_err=363.6 > 100.0× (nondet=0.0, mut=uniform)
- Detail: `rel_err=3.636e+02 > 100.0×tol  rtol=1e-04 atol=1e-05 nondet_floor=0.000e+00 dtype=float32 n=32`
- APIs used: tf.keras.layers.Dense, tf.keras.layers.BatchNormalization, tf.keras.layers.LayerNormalization, tf.GradientTape, tf.keras.layers.Maximum, tf.math.reduce_sum, tf.keras.metrics.Mean, tf.keras.metrics.SparseCategoricalAccuracy

### tensorflow bug_inconsistent_m0147_scale_small_20260416_214039
- Type: `inconsistent`
- Mutation: `scale_small`
- Reason: rel_err=359.7 > 100.0× (nondet=0.0, mut=scale_small)
- Detail: `rel_err=3.597e+02 > 100.0×tol  rtol=1e-04 atol=1e-05 nondet_floor=0.000e+00 dtype=float32 n=32`
- APIs used: tf.keras.layers.Dense, tf.keras.layers.LeakyReLU, tf.keras.layers.BatchNormalization, tf.keras.layers.LayerNormalization, tf.keras.constraints.NonNeg, tf.keras.optimizers.Adagrad, tf.GradientTape, tf.device

### tensorflow bug_inconsistent_m0149_scale_small_20260416_214243
- Type: `inconsistent`
- Mutation: `scale_small`
- Reason: rel_err=115.6 > 100.0× (nondet=0.0, mut=scale_small)
- Detail: `rel_err=1.156e+02 > 100.0×tol  rtol=1e-04 atol=1e-05 nondet_floor=0.000e+00 dtype=float32 n=32`
- APIs used: tf.keras.layers.Dense, tf.initializers.RandomNormal, tf.keras.layers.BatchNormalization, tf.keras.layers.LayerNormalization, tf.keras.layers.Concatenate, tf.keras.layers.Activation, tf.keras.optimizers.Nadam, tf.metrics.RootMeanSquaredError


### tensorflow bug_inconsistent_m0157_scale_small_20260416_215058
- Type: `inconsistent`
- Mutation: `scale_small`
- Reason: rel_err=122.5 > 100.0× (nondet=0.0, mut=scale_small)
- Detail: `rel_err=1.225e+02 > 100.0×tol  rtol=1e-04 atol=1e-05 nondet_floor=0.000e+00 dtype=float32 n=32`
- APIs used: tf.keras.layers.Dense, tf.keras.layers.BatchNormalization, tf.keras.layers.LayerNormalization, tf.GradientTape, tf.keras.layers.Convolution1DTranspose, tf.math.cos, tf.device

### tensorflow bug_inconsistent_m0163_scale_small_20260416_215721
- Type: `inconsistent`
- Mutation: `scale_small`
- Reason: rel_err=127.6 > 100.0× (nondet=0.0, mut=scale_small)
- Detail: `rel_err=1.276e+02 > 100.0×tol  rtol=1e-04 atol=1e-05 nondet_floor=0.000e+00 dtype=float32 n=32`
- APIs used: tf.keras.layers.Dense, tf.keras.regularizers.L1L2, tf.keras.layers.BatchNormalization, tf.keras.layers.LayerNormalization, tf.keras.layers.Activation, tf.math.multiply, tf.GradientTape, tf.device

### tensorflow bug_inconsistent_m0164_scale_small_20260416_215811
- Type: `inconsistent`
- Mutation: `scale_small`
- Reason: rel_err=638.6 > 100.0× (nondet=0.0, mut=scale_small)
- Detail: `rel_err=6.386e+02 > 100.0×tol  rtol=1e-04 atol=1e-05 nondet_floor=0.000e+00 dtype=float32 n=32`
- APIs used: tf.keras.layers.Dense, tf.initializers.Identity, tf.keras.layers.BatchNormalization, tf.keras.layers.LayerNormalization, tf.keras.activations.exponential, tf.GradientTape, tf.math.squared_difference, tf.nn.tanh
