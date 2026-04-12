"""
Bug: torch.scatter_reduce with float16 — 'sum' mode uses non-deterministic atomicAdd,
'mean' mode divides by count afterward, 'prod' mode uses non-deterministic atomicMul.
Different reduce modes have different GPU implementations.

This tests the newer scatter_reduce API (PyTorch >= 1.12).
"""
import torch
import numpy as np

print("=== PyTorch scatter_reduce float16: CPU vs GPU ===")
print(f"PyTorch version: {torch.__version__}")

if not torch.cuda.is_available():
    print("No CUDA"); exit(0)

rng = np.random.default_rng(42)

print("\n--- float16 scatter_reduce 'sum' mode ---")
for M, N in [(10000, 100), (100000, 500), (500000, 1000)]:
    src_np = rng.standard_normal(M).astype(np.float16)
    idx_np = rng.integers(0, N, size=M).astype(np.int64)

    # Reference float64
    ref = np.zeros(N, dtype=np.float64)
    for i in range(M):
        ref[idx_np[i]] += src_np[i]

    src = torch.tensor(src_np, dtype=torch.float16)
    idx = torch.tensor(idx_np, dtype=torch.long)

    try:
        cpu_out = torch.zeros(N, dtype=torch.float16)
        cpu_out.scatter_reduce_(0, idx, src, reduce='sum')
        cpu_np = cpu_out.float().numpy().astype(np.float64)

        gpu_out = torch.zeros(N, dtype=torch.float16, device='cuda')
        gpu_out.scatter_reduce_(0, idx.cuda(), src.cuda(), reduce='sum')
        gpu_np = gpu_out.float().cpu().numpy().astype(np.float64)

        cpu_err = float(np.max(np.abs(cpu_np - ref)))
        gpu_err = float(np.max(np.abs(gpu_np - ref)))
        diff = float(np.max(np.abs(cpu_np - gpu_np)))
        ratio = gpu_err / (cpu_err + 1e-30)

        print(f"M={M}, N={N}: CPU_err={cpu_err:.3e}, GPU_err={gpu_err:.3e}, diff={diff:.3e}", end="")
        if ratio > 5 and diff > 0.5:
            print(f"  *** GPU {ratio:.1f}x worse ***", end="")
        elif diff > 1.0:
            print(f"  *** SIGNIFICANT DIVERGENCE ***", end="")
        print()
    except Exception as e:
        print(f"M={M}, N={N}: ERROR: {e}")

# 'mean' mode
print("\n--- float16 scatter_reduce 'mean' mode ---")
for M, N in [(10000, 100), (100000, 500)]:
    src_np = rng.standard_normal(M).astype(np.float16)
    idx_np = rng.integers(0, N, size=M).astype(np.int64)

    # Reference
    ref_sum = np.zeros(N, dtype=np.float64)
    ref_count = np.zeros(N, dtype=np.int64)
    for i in range(M):
        ref_sum[idx_np[i]] += src_np[i]
        ref_count[idx_np[i]] += 1
    ref = ref_sum / np.maximum(ref_count, 1)

    src = torch.tensor(src_np, dtype=torch.float16)
    idx = torch.tensor(idx_np, dtype=torch.long)

    try:
        cpu_out = torch.zeros(N, dtype=torch.float16)
        cpu_out.scatter_reduce_(0, idx, src, reduce='mean')
        cpu_np = cpu_out.float().numpy().astype(np.float64)

        gpu_out = torch.zeros(N, dtype=torch.float16, device='cuda')
        gpu_out.scatter_reduce_(0, idx.cuda(), src.cuda(), reduce='mean')
        gpu_np = gpu_out.float().cpu().numpy().astype(np.float64)

        cpu_err = float(np.max(np.abs(cpu_np - ref)))
        gpu_err = float(np.max(np.abs(gpu_np - ref)))
        diff = float(np.max(np.abs(cpu_np - gpu_np)))
        ratio = gpu_err / (cpu_err + 1e-30)
        print(f"mean M={M}, N={N}: CPU_err={cpu_err:.3e}, GPU_err={gpu_err:.3e}, diff={diff:.3e}", end="")
        if ratio > 5 and diff > 0.1:
            print(f"  *** GPU {ratio:.1f}x worse ***", end="")
        elif diff > 0.5:
            print(f"  *** DIVERGENCE ***", end="")
        print()
    except Exception as e:
        print(f"mean M={M}, N={N}: ERROR: {e}")

# GPU non-determinism check
print("\n--- GPU non-determinism: scatter_reduce sum ---")
M, N = 500000, 1000
src_np = rng.standard_normal(M).astype(np.float16)
idx_np = rng.integers(0, N, size=M).astype(np.int64)
src = torch.tensor(src_np, dtype=torch.float16, device='cuda')
idx = torch.tensor(idx_np, dtype=torch.long, device='cuda')

runs = []
for _ in range(3):
    out = torch.zeros(N, dtype=torch.float16, device='cuda')
    out.scatter_reduce_(0, idx, src, reduce='sum')
    runs.append(out.float().cpu().numpy())

diff01 = float(np.max(np.abs(runs[0] - runs[1])))
diff02 = float(np.max(np.abs(runs[0] - runs[2])))
print(f"M={M}, N={N}: run0_vs_run1={diff01:.3e}, run0_vs_run2={diff02:.3e}", end="")
if diff01 > 0 or diff02 > 0:
    print("  *** GPU NON-DETERMINISTIC ***", end="")
print()

print(f"\n=== BUG SUMMARY ===")
M, N = 100000, 500
src_np = rng.standard_normal(M).astype(np.float16)
idx_np = rng.integers(0, N, size=M).astype(np.int64)
ref = np.zeros(N, dtype=np.float64)
for i in range(M):
    ref[idx_np[i]] += src_np[i]

src = torch.tensor(src_np, dtype=torch.float16)
idx = torch.tensor(idx_np, dtype=torch.long)
cpu_out = torch.zeros(N, dtype=torch.float16)
cpu_out.scatter_reduce_(0, idx, src, reduce='sum')
cpu_np = cpu_out.float().numpy().astype(np.float64)

gpu_out = torch.zeros(N, dtype=torch.float16, device='cuda')
gpu_out.scatter_reduce_(0, idx.cuda(), src.cuda(), reduce='sum')
gpu_np = gpu_out.float().cpu().numpy().astype(np.float64)

cpu_err = float(np.max(np.abs(cpu_np - ref)))
gpu_err = float(np.max(np.abs(gpu_np - ref)))
diff = float(np.max(np.abs(cpu_np - gpu_np)))
ratio = gpu_err / (cpu_err + 1e-30)
print(f"float16 scatter_reduce 'sum' M={M}, N={N}:")
print(f"CPU_err={cpu_err:.3e}, GPU_err={gpu_err:.3e}, diff={diff:.3e}")
if ratio > 5 and diff > 0.1:
    print(f"*** GPU {ratio:.1f}x less accurate ***")
elif diff > 0.1:
    print(f"*** SIGNIFICANT CPU/GPU DIVERGENCE ***")
else:
    print("No significant divergence")
