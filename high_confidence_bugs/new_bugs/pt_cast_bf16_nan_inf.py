"""
Bug: Casting bfloat16 NaN or +Inf to integer types gives different results
on CPU vs GPU in PyTorch.

Root cause: Same undefined behavior as the float32 and float16 cast bugs
(from PaddlePaddle #72779). The C and CUDA standards both leave float-to-int
conversion for out-of-range values (NaN, Inf) as undefined behavior for ALL
floating-point types including bfloat16. CPU x86 saturates to INT_MIN for any
out-of-range value. GPU CUDA bfloat16-to-int returns 0 for NaN and INT_MAX
for +Inf — opposite saturation semantics from CPU.
"""
import torch

cases = [
    (float("nan"), torch.int32, "bfloat16(nan) -> int32"),
    (float("inf"), torch.int32, "bfloat16(inf) -> int32"),
    (float("inf"), torch.int64, "bfloat16(inf) -> int64"),
]

for val, dtype, name in cases:
    x = torch.tensor(val, dtype=torch.bfloat16)
    cpu_v = x.to(dtype).item()
    gpu_v = x.cuda().to(dtype).cpu().item()
    print(f"{name}:  CPU={cpu_v}  GPU={gpu_v}  -> {'BUG' if cpu_v != gpu_v else 'ok'}")
