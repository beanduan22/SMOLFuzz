"""
Bug: Casting float32 NaN or +Inf to integer types gives completely different results
on CPU vs GPU in PyTorch.

Origin: Found in PaddlePaddle (issue #72779) where float32(NaN)->int32 gives
CPU=INT32_MIN, GPU=0. The same asymmetry exists in PyTorch.

Root cause: The C and CUDA standards both leave float-to-int conversion for
out-of-range values (NaN, Inf) as undefined behavior. CPU uses x86 CVTTSS2SI
which saturates to INT_MIN for any out-of-range value. GPU uses CUDA's
float-to-int conversion which has different saturation semantics:
  - NaN -> 0 (GPU) vs INT_MIN (CPU)
  - +Inf -> INT_MAX (GPU) vs INT_MIN (CPU)
This silent asymmetry corrupts data whenever NaN/Inf floats are cast to int
in mixed CPU/GPU pipelines.
"""
import torch

cases = [
    (float("nan"),  torch.int32,  "float32(nan)  -> int32"),
    (float("inf"),  torch.int32,  "float32(inf)  -> int32"),
    (float("inf"),  torch.int64,  "float32(inf)  -> int64"),
]

for val, dtype, name in cases:
    x = torch.tensor(val, dtype=torch.float32)
    cpu_v = x.to(dtype).item()
    gpu_v = x.cuda().to(dtype).cpu().item()
    print(f"{name}:  CPU={cpu_v}  GPU={gpu_v}  -> {'BUG' if cpu_v != gpu_v else 'ok'}")
