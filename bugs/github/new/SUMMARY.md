# smolfuzz run — real-bug triage

- Finalized at: 2026-04-16 22:31:09
- PT models started: 102
- PT clean (60s fuzzed, no bug): 32
- TF models started: 200
- TF clean (60s fuzzed, no bug): 105
- Initially flagged: 27
- **Real bugs kept: 4**
- Rejected as FP on re-triage: 23

## FP filters applied (original run)

- `BUG_MARGIN`: rel_err must exceed `100.0×` dtype tolerance
- Rejected if `nondet_floor >= 100.0` (inherently non-deterministic model)
- Rejected if `rel_err < 10.0 × nondet_floor`
- Infrastructure errors (CUBLAS/CUDA OOM/driver) filtered
- GPU-side `Model().to(device)` construction crashes filtered (contention, not library)
- Permutation-invariant API outputs (eig/svd/qr/sort/…) require sorted-magnitude divergence

## Additional FP filters applied (manual re-triage 2026-04-18)

- **All 22 TF bugs rejected**: Root cause was different random weight initialization between CPU
  and GPU models in the original fuzzer run. `tf.random.set_seed(42)` before GPU model init does
  not fully isolate TF's global op counter after the CPU warm-up pass, so weights diverged.
  With identical weights, every TF bug gives rel_err=0.0 exactly — no real CPU/GPU difference.
- **PT m0035 rejected**: `torch.nn.init.kaiming_normal_(x)` called on intermediate activation
  inside `forward()`, overwriting it with device-specific random numbers (CPU RNG ≠ GPU RNG).
  The divergence was RNG noise, not a library numerical bug.

## Real bugs (4 PyTorch)

### pytorch bug_inconsistent_m0028_mut5_20260416_183107
- Type: `inconsistent`
- Mutation: `scale_large`
- Reason: rel_err=838.0 > 100.0× (nondet=0.0, mut=scale_large)
- Detail: `output[0]: rel_err=8.380e+02 > 1.0 dtype=torch.float32 shape=[4, 4] rtol=1e-04 atol=1e-05 nondet_floor=0.000e+00`
- APIs used: torch.nn.Linear, torch.nn.functional.elu, torch.nn.BatchNorm1d, torch.sin, torch.clip, torch.cummax, torch.Tensor.max, torch.row_stack

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
