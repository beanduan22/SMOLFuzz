# Open / Unfixed Bug Reproducers

Minimal repros for **open and unfixed** bugs in PyTorch / TensorFlow.
Each file is a single bare bug — no helpers, no harness — and runs in a few lines.

## Layout

```
cpu_gpu/   one file per bug, multiple input variants inlined
crashes/   one file per crash variant (file dies on the crash)
```

## CPU vs GPU

| File | Issue |
|---|---|
| `cpu_gpu/pt_162235_neg_zero.py` | pytorch#162235 — `-0.0` CPU vs CUDA in `maximum` / `relu` / `argsort` / `amin` |
| `cpu_gpu/pt_52241_ctcloss_grad.py` | pytorch#52241 — `nn.CTCLoss` gradcheck fails |
| `cpu_gpu/tf_86378_biasaddgrad.py` | tensorflow#86378 — `tf.raw_ops.BiasAddGrad` |
| `cpu_gpu/tf_86256_adjust_hue.py` | tensorflow#86256 — `tf.image.adjust_hue` |
| `cpu_gpu/tf_96180_reciprocal_complex_inf.py` | tensorflow#96180 — `tf.math.reciprocal` complex inf |
| `cpu_gpu/tf_97204_notequal_nonbroadcast.py` | tensorflow#97204 — `tf.raw_ops.NotEqual` non-broadcastable |
| `cpu_gpu/tf_86350_batchmatmulv2.py` | tensorflow#86350 — `tf.raw_ops.BatchMatMulV2` |

## Crashes

| File | Issue | Expected |
|---|---|---|
| `crashes/pt_177829_lu_unpack_a_empty_pivots.py` | pytorch#177829 | SIGSEGV |
| `crashes/pt_177829_lu_unpack_b_pivots_1x0.py`   | pytorch#177829 | SIGSEGV (variant: shape (1,0)) |
| `crashes/pt_177829_lu_unpack_c_batched.py`      | pytorch#177829 | SIGSEGV (variant: batched) |
| `crashes/pt_173574_arange_a_int64_float_step.py` | pytorch#173574 | SIGFPE |
| `crashes/pt_173574_arange_b_neg_range.py`        | pytorch#173574 | SIGFPE (variant: negative range) |
| `crashes/pt_173574_arange_c_step_0_25.py`        | pytorch#173574 | SIGFPE (variant: step=0.25) |
| `crashes/tf_76726_encode_png_a_tile_zero.py`     | tensorflow#76726 | aborted (core dumped) |
| `crashes/tf_76726_encode_png_b_zero_height.py`   | tensorflow#76726 | aborted (variant: 0×W×C) |
| `crashes/tf_76726_encode_png_c_zero_width.py`    | tensorflow#76726 | aborted (variant: H×0×C) |
