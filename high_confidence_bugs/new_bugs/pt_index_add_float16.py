"""
Bug: torch.index_add_ / torch.index_add with float16 and duplicate indices.
GPU uses non-deterministic atomicAdd (like scatter_add), CPU uses sequential accumulation.
Duplicate indices cause non-associative floating-point additions on GPU.
"""
import torch
import numpy as np

print("=== PyTorch index_add float16: CPU vs GPU accuracy ===")
print(f"PyTorch version: {torch.__version__}")

if not torch.cuda.is_available():
    print("No CUDA"); exit(0)

rng = np.random.default_rng(42)
torch.manual_seed(42)

print("\n--- float16 index_add with high collision rate ---")
for M, N, repeats in [(100, 20, 5), (1000, 50, 20), (10000, 100, 100), (100000, 500, 200)]:
    # N output slots, M updates, many collisions
    src_np = rng.standard_normal(M).astype(np.float16)
    idx_np = rng.integers(0, N, size=M).astype(np.int64)
    ref_np = np.zeros(N, dtype=np.float64)
    for i in range(M):
        ref_np[idx_np[i]] += src_np[i]

    src = torch.tensor(src_np, dtype=torch.float16)
    idx = torch.tensor(idx_np, dtype=torch.long)

    cpu_out = torch.zeros(N, dtype=torch.float16)
    cpu_out.index_add_(0, idx, src)
    cpu_np = cpu_out.float().numpy().astype(np.float64)

    gpu_out = torch.zeros(N, dtype=torch.float16, device='cuda')
    gpu_out.index_add_(0, idx.cuda(), src.cuda())
    gpu_np = gpu_out.float().cpu().numpy().astype(np.float64)

    cpu_err = float(np.max(np.abs(cpu_np - ref_np)))
    gpu_err = float(np.max(np.abs(gpu_np - ref_np)))
    diff = float(np.max(np.abs(cpu_np - gpu_np)))
    ratio = gpu_err / (cpu_err + 1e-30)

    print(f"M={M}, N={N}: CPU_err={cpu_err:.3e}, GPU_err={gpu_err:.3e}, CPU_vs_GPU={diff:.3e}", end="")
    if ratio > 5 and diff > 0.1:
        print(f"  *** GPU {ratio:.1f}x worse ***", end="")
    elif diff > 1.0:
        print(f"  *** SIGNIFICANT DIVERGENCE ***", end="")
    print()

# GPU non-determinism check
print("\n--- GPU non-determinism for float16 index_add ---")
for M, N in [(50000, 200), (100000, 500)]:
    src_np = rng.standard_normal(M).astype(np.float16)
    idx_np = rng.integers(0, N, size=M).astype(np.int64)
    src = torch.tensor(src_np, dtype=torch.float16, device='cuda')
    idx = torch.tensor(idx_np, dtype=torch.long, device='cuda')

    runs = []
    for _ in range(3):
        out = torch.zeros(N, dtype=torch.float16, device='cuda')
        out.index_add_(0, idx, src)
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
ref_np = np.zeros(N, dtype=np.float64)
for i in range(M):
    ref_np[idx_np[i]] += src_np[i]

src = torch.tensor(src_np, dtype=torch.float16)
idx = torch.tensor(idx_np, dtype=torch.long)
cpu_out = torch.zeros(N, dtype=torch.float16)
cpu_out.index_add_(0, idx, src)
cpu_np = cpu_out.float().numpy().astype(np.float64)

gpu_out = torch.zeros(N, dtype=torch.float16, device='cuda')
gpu_out.index_add_(0, idx.cuda(), src.cuda())
gpu_np = gpu_out.float().cpu().numpy().astype(np.float64)

cpu_err = float(np.max(np.abs(cpu_np - ref_np)))
gpu_err = float(np.max(np.abs(gpu_np - ref_np)))
diff = float(np.max(np.abs(cpu_np - gpu_np)))
ratio = gpu_err / (cpu_err + 1e-30)
print(f"float16 index_add M={M}, N={N}:")
print(f"CPU_err={cpu_err:.3e}, GPU_err={gpu_err:.3e}, CPU_vs_GPU={diff:.3e}")
if ratio > 5 and diff > 1.0:
    print(f"*** GPU {ratio:.1f}x less accurate ***")
elif diff > 1.0:
    print(f"*** SIGNIFICANT CPU/GPU DIVERGENCE ***")
