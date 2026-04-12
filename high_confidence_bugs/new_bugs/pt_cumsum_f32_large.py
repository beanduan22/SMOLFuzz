"""
Bug: torch.cumsum with float32 and very large arrays (N=10M+) —
CPU uses sequential single-pass accumulation, GPU uses parallel prefix sum
(Hillis-Steele or Blelloch scan). For large N, the parallel scan is more
accurate due to pairwise summation, but the algorithms can diverge for
specific input distributions.
"""
import torch
import numpy as np

print("=== PyTorch cumsum float32 large array: CPU vs GPU ===")
print(f"PyTorch version: {torch.__version__}")

if not torch.cuda.is_available():
    print("No CUDA"); exit(0)

rng = np.random.default_rng(42)

print("\n--- float32 cumsum accuracy at large N ---")
for N in [1_000_000, 5_000_000, 10_000_000, 50_000_000]:
    x_np = rng.standard_normal(N).astype(np.float32)
    ref = np.cumsum(x_np.astype(np.float64))
    x = torch.tensor(x_np)

    try:
        cpu_cumsum = torch.cumsum(x, dim=0)
        gpu_cumsum = torch.cumsum(x.cuda(), dim=0).cpu()

        cpu_err = float((cpu_cumsum.double() - torch.tensor(ref)).abs().max().item())
        gpu_err = float((gpu_cumsum.double() - torch.tensor(ref)).abs().max().item())
        diff = float((cpu_cumsum - gpu_cumsum).abs().max().item())
        ratio = gpu_err / (cpu_err + 1e-30)

        print(f"N={N:>11}: CPU_err={cpu_err:.4e}, GPU_err={gpu_err:.4e}, diff={diff:.4e}, ratio={ratio:.1f}x", end="")
        if ratio > 10 and diff > 1.0:
            print(f"  *** GPU {ratio:.0f}x less accurate ***", end="")
        elif 1/ratio > 10 and diff > 1.0:
            print(f"  *** CPU {1/ratio:.0f}x less accurate ***", end="")
        print()
    except Exception as e:
        print(f"N={N}: ERROR: {e}")

# All-positive (monotone accumulation - worst case for cancellation)
print("\n--- float32 cumsum with all-positive values ---")
for N in [1_000_000, 10_000_000]:
    x_np = np.abs(rng.standard_normal(N)).astype(np.float32)
    ref = np.cumsum(x_np.astype(np.float64))
    x = torch.tensor(x_np)

    try:
        cpu_cumsum = torch.cumsum(x, dim=0)
        gpu_cumsum = torch.cumsum(x.cuda(), dim=0).cpu()

        cpu_err = float((cpu_cumsum.double() - torch.tensor(ref)).abs().max().item())
        gpu_err = float((gpu_cumsum.double() - torch.tensor(ref)).abs().max().item())
        diff = float((cpu_cumsum - gpu_cumsum).abs().max().item())
        ratio = gpu_err / (cpu_err + 1e-30)

        print(f"positive N={N}: CPU_err={cpu_err:.4e}, GPU_err={gpu_err:.4e}, diff={diff:.4e}, ratio={ratio:.1f}x", end="")
        if ratio > 10 and diff > 1.0:
            print(f"  *** GPU {ratio:.0f}x less accurate ***", end="")
        elif 1/ratio > 10 and diff > 1.0:
            print(f"  *** CPU {1/ratio:.0f}x less accurate ***", end="")
        print()
    except Exception as e:
        print(f"positive N={N}: ERROR: {e}")

print(f"\n=== BUG SUMMARY ===")
# Use all-positive values to maximize absolute error visibility
N = 10_000_000
x_np = np.abs(rng.standard_normal(N)).astype(np.float32)
ref = np.cumsum(x_np.astype(np.float64))
x = torch.tensor(x_np)
try:
    cpu_cumsum = torch.cumsum(x, dim=0)
    gpu_cumsum = torch.cumsum(x.cuda(), dim=0).cpu()

    cpu_err = float((cpu_cumsum.double() - torch.tensor(ref)).abs().max().item())
    gpu_err = float((gpu_cumsum.double() - torch.tensor(ref)).abs().max().item())
    diff = float((cpu_cumsum - gpu_cumsum).abs().max().item())
    ratio = gpu_err / (cpu_err + 1e-30)

    print(f"cumsum(float32, N=10M, all-positive): CPU_err={cpu_err:.4e}, GPU_err={gpu_err:.4e}, diff={diff:.4e}")
    if ratio > 5 and diff > 0.1:
        print(f"*** BUG: GPU is {ratio:.0f}x less accurate than CPU ***")
    elif 1/ratio > 5 and diff > 0.1:
        print(f"*** BUG: CPU is {1/ratio:.0f}x less accurate than GPU ***")
    else:
        print(f"No significant divergence (ratio={ratio:.1f}x)")
except Exception as e:
    print(f"ERROR: {e}")
