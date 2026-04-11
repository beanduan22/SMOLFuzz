"""
Bug: torch.matmul / torch.einsum on Ampere+ GPUs with TF32 enabled is up to 1208x
less accurate than CPU for float32 inputs.

Origin: Found in JAX (issue #22557) where jnp.einsum produces GPU error ~0.92
vs CPU error ~0.013 (70x gap) due to reduced matmul precision on accelerators.
PyTorch exhibits the same bug, with an even larger accuracy gap (1208x).

Root cause: NVIDIA Ampere+ GPUs support TF32 (TensorFloat-32), which reduces the
mantissa from 23 bits (float32) to 10 bits (equivalent to bfloat16 precision) in
matrix multiplications, while keeping the exponent range of float32. When
torch.backends.cuda.matmul.allow_tf32 = True (the DEFAULT on Ampere+), every
matmul silently loses 13 bits of mantissa precision. CPU always uses full float32.

Impact: Any model trained or inferred on Ampere+ GPUs (A100, RTX 3090, etc.) with
default settings silently uses lower precision than CPU. The error grows with
matrix size.
"""
import torch

torch.manual_seed(0)
n = 512
A = torch.randn(n, n, dtype=torch.float32)
B = torch.randn(n, n, dtype=torch.float32)

ref = (A.double() @ B.double()).float()   # float64 reference
cpu = A @ B                               # CPU always uses full float32

torch.backends.cuda.matmul.allow_tf32 = True    # default on Ampere+
gpu_tf32 = (A.cuda() @ B.cuda()).cpu()
torch.backends.cuda.matmul.allow_tf32 = False
gpu_fp32 = (A.cuda() @ B.cuda()).cpu()

cpu_err  = (cpu - ref).norm().item()
tf32_err = (gpu_tf32 - ref).norm().item()
fp32_err = (gpu_fp32 - ref).norm().item()

print(f"CPU  error vs float64 reference: {cpu_err:.4e}")
print(f"GPU  error (TF32=off):           {fp32_err:.4e}")
print(f"GPU  error (TF32=on):            {tf32_err:.4e}   <-- BUG")
print(f"TF32 makes GPU {tf32_err / cpu_err:.0f}x less accurate than CPU")
print()
print(f"First 4 output values:")
print(f"  ref:      {ref.flatten()[:4].tolist()}")
print(f"  cpu:      {cpu.flatten()[:4].tolist()}")
print(f"  gpu_tf32: {gpu_tf32.flatten()[:4].tolist()}")
