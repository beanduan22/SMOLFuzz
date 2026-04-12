"""
Bug: torch.Tensor.index_reduce_ with float16 — GPU uses non-deterministic
atomicAdd/atomicMul depending on reduce mode. The 'mean' mode computes
an online mean (potentially atomicAdd + count tracking), while 'sum' uses
atomicAdd directly. For float16, this is highly non-deterministic.
"""
import torch
import numpy as np

print("=== PyTorch index_reduce_ float16: CPU vs GPU ===")
print(f"PyTorch version: {torch.__version__}")

if not torch.cuda.is_available():
    print("No CUDA"); exit(0)

rng = np.random.default_rng(42)

print("\n--- float16 index_reduce_ 'mean' mode ---")
for M, N in [(10000, 100), (100000, 500), (500000, 1000)]:
    src_np = rng.standard_normal(M).astype(np.float16)
    idx_np = rng.integers(0, N, size=M).astype(np.int64)

    # Reference float64 mean
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
        cpu_out.index_reduce_(0, idx, src, 'mean')
        cpu_np = cpu_out.float().numpy().astype(np.float64)

        gpu_out = torch.zeros(N, dtype=torch.float16, device='cuda')
        gpu_out.index_reduce_(0, idx.cuda(), src.cuda(), 'mean')
        gpu_np = gpu_out.float().cpu().numpy().astype(np.float64)

        cpu_err = float(np.max(np.abs(cpu_np - ref)))
        gpu_err = float(np.max(np.abs(gpu_np - ref)))
        diff = float(np.max(np.abs(cpu_np - gpu_np)))

        print(f"M={M}, N={N}: CPU_err={cpu_err:.3e}, GPU_err={gpu_err:.3e}, diff={diff:.3e}", end="")
        if diff > 0.01:
            print(f"  *** DIVERGENCE ***", end="")
        print()
    except Exception as e:
        print(f"M={M}, N={N}: ERROR: {e}")

# GPU non-determinism check
print("\n--- GPU non-determinism: index_reduce_ float16 ---")
M, N = 500000, 1000
src_np = rng.standard_normal(M).astype(np.float16)
idx_np = rng.integers(0, N, size=M).astype(np.int64)
src = torch.tensor(src_np, dtype=torch.float16, device='cuda')
idx = torch.tensor(idx_np, dtype=torch.long, device='cuda')

runs = []
for _ in range(3):
    out = torch.zeros(N, dtype=torch.float16, device='cuda')
    try:
        out.index_reduce_(0, idx, src, 'mean')
        runs.append(out.float().cpu().numpy())
    except Exception as e:
        print(f"ERROR: {e}")
        break

if len(runs) == 3:
    diff01 = float(np.max(np.abs(runs[0] - runs[1])))
    diff02 = float(np.max(np.abs(runs[0] - runs[2])))
    print(f"M={M}, N={N}: run0_vs_run1={diff01:.3e}, run0_vs_run2={diff02:.3e}", end="")
    if diff01 > 0 or diff02 > 0:
        print("  *** GPU NON-DETERMINISTIC ***", end="")
    print()

print(f"\n=== BUG SUMMARY ===")
M, N = 500000, 1000
src_np2 = np.random.default_rng(99).standard_normal(M).astype(np.float16)
idx_np2 = np.random.default_rng(99).integers(0, N, size=M).astype(np.int64)
src2 = torch.tensor(src_np2, dtype=torch.float16, device='cuda')
idx2 = torch.tensor(idx_np2, dtype=torch.long, device='cuda')

try:
    runs2 = []
    for _ in range(3):
        out2 = torch.zeros(N, dtype=torch.float16, device='cuda')
        out2.index_reduce_(0, idx2, src2, 'mean')
        runs2.append(out2.float().cpu().numpy())

    nd01 = float(np.max(np.abs(runs2[0] - runs2[1])))
    nd02 = float(np.max(np.abs(runs2[0] - runs2[2])))
    print(f"index_reduce_(float16, M={M}, N={N}, 'mean'):")
    print(f"GPU run0_vs_run1={nd01:.3e}, run0_vs_run2={nd02:.3e}")
    if nd01 > 0 or nd02 > 0:
        print(f"*** BUG: GPU non-deterministic for index_reduce_ float16 'mean' (diff={max(nd01,nd02):.3e}) ***")
    else:
        print("No significant non-determinism")
except Exception as e:
    print(f"ERROR: {e}")
