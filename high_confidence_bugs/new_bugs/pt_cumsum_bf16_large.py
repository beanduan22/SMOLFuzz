"""
Bug: torch.cumsum with bfloat16 on large inputs — CPU uses sequential
accumulation while GPU uses a parallel prefix scan (Blelloch algorithm).
For all-positive inputs at large N, the parallel scan introduces more
rounding error because partial sums are computed at lower precision,
causing GPU to be 2-3x less accurate than CPU.
"""
import torch
import numpy as np

print("=== PyTorch cumsum bfloat16 large: CPU vs GPU accuracy ===")
print(f"PyTorch version: {torch.__version__}")

if not torch.cuda.is_available():
    print("No CUDA"); exit(0)

rng = np.random.default_rng(42)

print("\n--- bfloat16 cumsum accuracy (all-positive, monotone accumulation) ---")
for N in [100000, 500000, 1000000, 5000000, 10000000]:
    x_np = np.abs(rng.standard_normal(N)).astype(np.float32)
    ref = np.cumsum(x_np.astype(np.float64))

    x_cpu = torch.tensor(x_np, dtype=torch.bfloat16)
    x_gpu = x_cpu.cuda()

    out_cpu = torch.cumsum(x_cpu, 0).float().numpy().astype(np.float64)
    out_gpu = torch.cumsum(x_gpu, 0).float().cpu().numpy().astype(np.float64)

    cpu_err = float(np.max(np.abs(out_cpu - ref)))
    gpu_err = float(np.max(np.abs(out_gpu - ref)))
    diff = float(np.max(np.abs(out_cpu - out_gpu)))
    ratio = gpu_err / (cpu_err + 1e-10)

    print(f"N={N:>10}: CPU_err={cpu_err:.3e}, GPU_err={gpu_err:.3e}, diff={diff:.3e}, ratio={ratio:.1f}x", end="")
    if ratio > 1.5 and diff > 1.0:
        print(f"  *** GPU {ratio:.0f}x less accurate ***", end="")
    print()

# Mixed-sign inputs (smaller but still visible)
print("\n--- bfloat16 cumsum: mixed-sign large inputs ---")
for N in [1000000, 5000000, 10000000]:
    x_np = rng.standard_normal(N).astype(np.float32)
    ref = np.cumsum(x_np.astype(np.float64))

    x_cpu = torch.tensor(x_np, dtype=torch.bfloat16)
    x_gpu = x_cpu.cuda()

    out_cpu = torch.cumsum(x_cpu, 0).float().numpy().astype(np.float64)
    out_gpu = torch.cumsum(x_gpu, 0).float().cpu().numpy().astype(np.float64)

    cpu_err = float(np.max(np.abs(out_cpu - ref)))
    gpu_err = float(np.max(np.abs(out_gpu - ref)))
    diff = float(np.max(np.abs(out_cpu - out_gpu)))
    ratio = gpu_err / (cpu_err + 1e-10)

    print(f"mixed N={N:>10}: cpu_err={cpu_err:.3e}, gpu_err={gpu_err:.3e}, diff={diff:.3e}, ratio={ratio:.1f}x")

# Comparison: float32 behaves the same (shows the bf16 issue is dtype-specific precision loss)
print("\n--- float32 cumsum at same sizes (reference) ---")
for N in [1000000, 10000000]:
    x_np = np.abs(rng.standard_normal(N)).astype(np.float32)
    ref = np.cumsum(x_np.astype(np.float64))
    x_cpu = torch.tensor(x_np)
    x_gpu = x_cpu.cuda()
    out_cpu = torch.cumsum(x_cpu, 0).numpy().astype(np.float64)
    out_gpu = torch.cumsum(x_gpu, 0).cpu().numpy().astype(np.float64)
    cpu_err = float(np.max(np.abs(out_cpu - ref)))
    gpu_err = float(np.max(np.abs(out_gpu - ref)))
    ratio = gpu_err / (cpu_err + 1e-10)
    print(f"f32 N={N:>10}: cpu_err={cpu_err:.3e}, gpu_err={gpu_err:.3e}, ratio={ratio:.1f}x")

print(f"\n=== BUG SUMMARY ===")
N = 10000000
x_np = np.abs(np.random.default_rng(99).standard_normal(N)).astype(np.float32)
ref = np.cumsum(x_np.astype(np.float64))

x_cpu = torch.tensor(x_np, dtype=torch.bfloat16)
x_gpu = x_cpu.cuda()
out_cpu = torch.cumsum(x_cpu, 0).float().numpy().astype(np.float64)
out_gpu = torch.cumsum(x_gpu, 0).float().cpu().numpy().astype(np.float64)

cpu_err = float(np.max(np.abs(out_cpu - ref)))
gpu_err = float(np.max(np.abs(out_gpu - ref)))
diff = float(np.max(np.abs(out_cpu - out_gpu)))
ratio = gpu_err / (cpu_err + 1e-10)
print(f"cumsum(bfloat16, N={N}, all-positive):")
print(f"CPU_err={cpu_err:.3e}, GPU_err={gpu_err:.3e}")
print(f"CPU vs GPU diff={diff:.3e}, GPU/CPU accuracy ratio={ratio:.1f}x")
if ratio > 1.5 and diff > 1.0:
    print(f"*** BUG: GPU is {ratio:.0f}x less accurate than CPU for bfloat16 cumsum ***")
else:
    print(f"No significant divergence (ratio={ratio:.2f}x)")
