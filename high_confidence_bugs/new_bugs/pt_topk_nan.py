"""
Bug: torch.topk with NaN inputs — CPU and GPU may return NaN in different positions.
IEEE 754 does not specify NaN handling for comparisons, so implementations differ.

Related to pt_sort_nan_position (which tests sort).
topk is used in attention mechanisms and beam search — NaN handling matters.
"""
import torch
import numpy as np

print("=== PyTorch torch.topk NaN handling CPU vs GPU ===")
print(f"PyTorch version: {torch.__version__}")

if not torch.cuda.is_available():
    print("No CUDA"); exit(0)

print("\n--- topk with NaN at different positions ---")
test_cases = [
    ("NaN at start", [float('nan'), 5.0, 3.0, 1.0, 2.0]),
    ("NaN in middle", [5.0, 3.0, float('nan'), 1.0, 2.0]),
    ("NaN at end", [5.0, 3.0, 1.0, 2.0, float('nan')]),
    ("NaN is largest", [float('nan'), 1.0, 2.0, 3.0, 4.0]),
    ("Multiple NaN", [float('nan'), 5.0, float('nan'), 1.0, 2.0]),
    ("All equal with NaN", [1.0, float('nan'), 1.0, 1.0, 1.0]),
    ("NaN and inf", [float('nan'), float('inf'), 5.0, 3.0, 1.0]),
    ("NaN and -inf", [float('nan'), float('-inf'), 5.0, 3.0, 1.0]),
]

for name, vals in test_cases:
    x = torch.tensor(vals, dtype=torch.float32)
    for k in [1, 2, 3]:
        if k > len(vals):
            continue
        try:
            cpu_vals, cpu_idx = torch.topk(x, k)
            gpu_vals, gpu_idx = torch.topk(x.cuda(), k)
            gpu_vals, gpu_idx = gpu_vals.cpu(), gpu_idx.cpu()

            vals_match = torch.allclose(cpu_vals, gpu_vals, equal_nan=True)
            idx_match = torch.all(cpu_idx == gpu_idx).item()

            # Check NaN patterns
            cpu_nan = torch.isnan(cpu_vals)
            gpu_nan = torch.isnan(gpu_vals)
            nan_diff = (cpu_nan != gpu_nan).any().item()

            if nan_diff or not idx_match:
                print(f"{name} k={k}: CPU_vals={cpu_vals.tolist()} CPU_idx={cpu_idx.tolist()}")
                print(f"  GPU_vals={gpu_vals.tolist()} GPU_idx={gpu_idx.tolist()}", end="")
                if nan_diff:
                    print("  *** NaN PATTERN DIFFERS ***", end="")
                if not idx_match:
                    print("  *** INDICES DIFFER ***", end="")
                print()
        except Exception as e:
            print(f"{name} k={k}: ERROR: {e}")

# Large arrays with scattered NaN
print("\n--- Large arrays: topk with NaN ---")
rng = np.random.default_rng(42)
for N, k, nan_frac in [(1000, 10, 0.01), (10000, 100, 0.001), (100000, 1000, 0.0001)]:
    x_np = rng.standard_normal(N).astype(np.float32)
    nan_positions = rng.integers(0, N, size=int(N * nan_frac))
    x_np[nan_positions] = float('nan')
    x = torch.tensor(x_np)

    cpu_vals, cpu_idx = torch.topk(x, k)
    gpu_vals, gpu_idx = torch.topk(x.cuda(), k)
    gpu_vals, gpu_idx = gpu_vals.cpu(), gpu_idx.cpu()

    cpu_nan_count = torch.isnan(cpu_vals).sum().item()
    gpu_nan_count = torch.isnan(gpu_vals).sum().item()
    idx_diff = (cpu_idx != gpu_idx).sum().item()

    print(f"N={N}, k={k}, NaN_frac={nan_frac}: CPU_nan_in_topk={cpu_nan_count}, GPU_nan_in_topk={gpu_nan_count}, idx_diff={idx_diff}", end="")
    if cpu_nan_count != gpu_nan_count:
        print(f"  *** NaN COUNT DIFFERS ***", end="")
    elif idx_diff > 0:
        print(f"  *** INDICES DIFFER ***", end="")
    print()

# float16 topk
print("\n--- float16 topk NaN ---")
for scale in [1.0, 100.0, 1000.0]:
    x_np = np.array([float('nan'), scale, scale*0.5, scale*0.1, scale*0.01], dtype=np.float16)
    x = torch.tensor(x_np, dtype=torch.float16)
    cpu_vals, cpu_idx = torch.topk(x, 3)
    gpu_vals, gpu_idx = torch.topk(x.cuda(), 3)
    gpu_vals, gpu_idx = gpu_vals.cpu(), gpu_idx.cpu()
    cpu_nan = torch.isnan(cpu_vals).any().item()
    gpu_nan = torch.isnan(gpu_vals).any().item()
    idx_match = torch.all(cpu_idx == gpu_idx).item()
    print(f"f16 scale={scale}: CPU_vals={cpu_vals.tolist()} GPU_vals={gpu_vals.tolist()}", end="")
    if cpu_nan != gpu_nan or not idx_match:
        print("  *** DIVERGENCE ***", end="")
    print()

print(f"\n=== BUG SUMMARY ===")
# Use multiple NaN case to trigger the bug
vals = [float('nan'), 5.0, float('nan'), 1.0, 2.0]
x = torch.tensor(vals, dtype=torch.float32)
cpu_vals, cpu_idx = torch.topk(x, 3)
gpu_vals, gpu_idx = torch.topk(x.cuda(), 3)
gpu_vals, gpu_idx = gpu_vals.cpu(), gpu_idx.cpu()
print(f"topk([NaN, 5, NaN, 1, 2], k=3):")
print(f"CPU: vals={cpu_vals.tolist()}, idx={cpu_idx.tolist()}")
print(f"GPU: vals={gpu_vals.tolist()}, idx={gpu_idx.tolist()}")
cpu_nan = torch.isnan(cpu_vals)
gpu_nan = torch.isnan(gpu_vals)
if (cpu_nan != gpu_nan).any():
    print("*** BUG: CPU and GPU return NaN in different positions ***")
elif (cpu_idx != gpu_idx).any():
    print("*** BUG: CPU and GPU return different NaN indices ***")
    print(f"CPU idx={cpu_idx.tolist()} vs GPU idx={gpu_idx.tolist()}")
else:
    print("CPU and GPU agree")
