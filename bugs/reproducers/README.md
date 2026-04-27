# Open / Unfixed Bug Reproducers

Minimal reproducers for **open and unfixed** bugs in PyTorch / TensorFlow.
Each file ships the original issue's repro plus 2–4 variations (different shapes / dtypes / formats / inputs) targeting the same root cause, so they double as fuzzer seeds.

## Layout

```
cpu_gpu/   — CPU vs GPU silent divergence
crashes/   — segfaults / SIGABRT / SIGFPE / hangs (run isolated in subprocesses)
```

## CPU vs GPU divergence

| File | Issue | Title |
|---|---|---|
| `cpu_gpu/pt_162235_neg_zero.py` | pytorch#162235 | Inconsistent `-0.0` between CPU and CUDA (`maximum`, `relu`, `argsort`, `sort`, `amin`, `amax`, ...) |
| `cpu_gpu/pt_52241_ctcloss_grad.py` | pytorch#52241 | `nn.CTCLoss` gradient incorrect (gradcheck fails) |
| `cpu_gpu/tf_86378_biasaddgrad.py` | tensorflow#86378 | `tf.raw_ops.BiasAddGrad` CPU vs GPU |
| `cpu_gpu/tf_86256_adjust_hue.py` | tensorflow#86256 | `tf.image.adjust_hue` CPU vs GPU |
| `cpu_gpu/tf_96180_reciprocal_complex_inf.py` | tensorflow#96180 | `tf.math.reciprocal` complex128 inf CPU vs GPU |
| `cpu_gpu/tf_97204_notequal_nonbroadcast.py` | tensorflow#97204 | `tf.raw_ops.NotEqual` non-broadcastable shapes — silent on CPU, raises on GPU |
| `cpu_gpu/tf_86350_batchmatmulv2.py` | tensorflow#86350 | `tf.raw_ops.BatchMatMulV2` CPU vs GPU |

## Crash / Internal Error

| File | Issue | Title |
|---|---|---|
| `crashes/pt_177829_lu_unpack_empty.py` | pytorch#177829 | `torch.lu_unpack` segfault on empty `LU_pivots` |
| `crashes/pt_173574_arange_int64_out.py` | pytorch#173574 | `torch.arange` SIGFPE writing float into int64 `out` |
| `crashes/tf_76726_encode_png.py` | tensorflow#76726 | `tf.io.encode_png` core dump on zero-sized image |

## Running

```bash
# CPU/GPU files require CUDA (PyTorch) or a visible GPU device (TensorFlow).
python3 bugs/reproducers/cpu_gpu/pt_162235_neg_zero.py

# Crash reproducers work without GPU. Each variant runs in its own subprocess
# so the harness survives SIGSEGV / SIGABRT / SIGFPE.
python3 bugs/reproducers/crashes/pt_177829_lu_unpack_empty.py
```

## Output convention

CPU/GPU files print one line per variant: `[name] shape=… dtype=… max|diff|=…` and any signbit / NaN-count anomaly.

Crash files print one line per variant: `[name] {ok | SIGSEGV | SIGFPE | SIGABRT | exit=N}` with the last stderr line.
