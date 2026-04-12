"""
Bug: torch.sort with NaN inputs — CPU and GPU handle NaN position differently.
IEEE 754: NaN comparison is always False.
CPU (sequential): NaN stays at its original position (treat NaN as less-than).
GPU (parallel bitonic sort): NaN may be moved to end or beginning.

When multiple NaN values appear, CPU and GPU may return different sorted orders.
"""
import torch
import numpy as np

print("=== PyTorch torch.sort NaN position: CPU vs GPU ===")
print(f"PyTorch version: {torch.__version__}")

if not torch.cuda.is_available():
    print("No CUDA"); exit(0)

print("\n--- sort with NaN at different positions ---")
test_cases = [
    ("NaN at start", [float('nan'), 3.0, 1.0, 2.0, 4.0]),
    ("NaN in middle", [1.0, 3.0, float('nan'), 2.0, 4.0]),
    ("NaN at end", [3.0, 1.0, 2.0, 4.0, float('nan')]),
    ("Multiple NaN", [float('nan'), 3.0, float('nan'), 1.0, float('nan')]),
    ("All NaN", [float('nan'), float('nan'), float('nan')]),
    ("NaN and inf", [float('nan'), float('inf'), 5.0, float('-inf'), 0.0]),
    ("Large array: NaN scattered", None),  # handled separately
]

for name, vals in test_cases[:6]:
    x = torch.tensor(vals, dtype=torch.float32)
    cpu_sorted, cpu_idx = torch.sort(x)
    gpu_sorted, gpu_idx = torch.sort(x.cuda())
    gpu_sorted, gpu_idx = gpu_sorted.cpu(), gpu_idx.cpu()

    # Check NaN positions
    cpu_nan_pos = [i for i, v in enumerate(cpu_sorted.tolist()) if v != v]  # NaN != NaN
    gpu_nan_pos = [i for i, v in enumerate(gpu_sorted.tolist()) if v != v]

    print(f"{name}:")
    print(f"  CPU sorted: {cpu_sorted.tolist()}")
    print(f"  GPU sorted: {gpu_sorted.tolist()}", end="")
    if cpu_nan_pos != gpu_nan_pos:
        print(f"  *** NaN POSITION DIFFERS: CPU={cpu_nan_pos}, GPU={gpu_nan_pos} ***", end="")
    elif cpu_idx.tolist() != gpu_idx.tolist():
        print(f"  *** INDICES DIFFER ***", end="")
    print()

# Sort descending with NaN
print("\n--- sort descending with NaN ---")
for name, vals in [
    ("NaN at start", [float('nan'), 3.0, 1.0, 2.0, 4.0]),
    ("NaN in middle", [1.0, 3.0, float('nan'), 2.0, 4.0]),
    ("Multiple NaN", [float('nan'), 3.0, float('nan'), 1.0, 2.0]),
]:
    x = torch.tensor(vals, dtype=torch.float32)
    cpu_sorted, _ = torch.sort(x, descending=True)
    gpu_sorted, _ = torch.sort(x.cuda(), descending=True)
    gpu_sorted = gpu_sorted.cpu()

    cpu_nan_pos = [i for i, v in enumerate(cpu_sorted.tolist()) if v != v]
    gpu_nan_pos = [i for i, v in enumerate(gpu_sorted.tolist()) if v != v]

    print(f"desc {name}: CPU={cpu_sorted.tolist()}, GPU={gpu_sorted.tolist()}", end="")
    if cpu_nan_pos != gpu_nan_pos:
        print(f"  *** NaN POSITION DIFFERS ***", end="")
    print()

# Large array sort with NaN
print("\n--- Large array sort with NaN ---")
rng = np.random.default_rng(42)
for N, nan_frac in [(10000, 0.01), (100000, 0.001)]:
    x_np = rng.standard_normal(N).astype(np.float32)
    nan_idx = rng.integers(0, N, size=int(N * nan_frac))
    x_np[nan_idx] = float('nan')
    x = torch.tensor(x_np)

    cpu_sorted, _ = torch.sort(x)
    gpu_sorted, _ = torch.sort(x.cuda())
    gpu_sorted = gpu_sorted.cpu()

    cpu_nan_positions = (cpu_sorted != cpu_sorted).nonzero(as_tuple=True)[0].tolist()
    gpu_nan_positions = (gpu_sorted != gpu_sorted).nonzero(as_tuple=True)[0].tolist()

    if len(cpu_nan_positions) > 0 and len(gpu_nan_positions) > 0:
        cpu_first_nan = cpu_nan_positions[0]
        gpu_first_nan = gpu_nan_positions[0]
        print(f"N={N}: CPU first NaN at pos {cpu_first_nan}, GPU first NaN at pos {gpu_first_nan}", end="")
        if cpu_first_nan != gpu_first_nan:
            print(f"  *** NaN POSITION DIFFERS ***", end="")
        print()

# float16 sort with NaN
print("\n--- float16 sort with NaN ---")
x = torch.tensor([float('nan'), 100.0, float('nan'), 50.0, float('nan')], dtype=torch.float16)
cpu_sorted, _ = torch.sort(x)
gpu_sorted, _ = torch.sort(x.cuda())
gpu_sorted = gpu_sorted.cpu()
cpu_nan_pos = [i for i, v in enumerate(cpu_sorted.tolist()) if v != v]
gpu_nan_pos = [i for i, v in enumerate(gpu_sorted.tolist()) if v != v]
print(f"f16 sort([NaN,100,NaN,50,NaN]):")
print(f"  CPU: {cpu_sorted.tolist()} (NaN at {cpu_nan_pos})")
print(f"  GPU: {gpu_sorted.tolist()} (NaN at {gpu_nan_pos})", end="")
if cpu_nan_pos != gpu_nan_pos:
    print("  *** NaN POSITION DIFFERS ***", end="")
print()

print(f"\n=== BUG SUMMARY ===")
vals = [float('nan'), 3.0, float('nan'), 1.0, float('nan')]
x = torch.tensor(vals, dtype=torch.float32)
cpu_sorted, cpu_idx = torch.sort(x)
gpu_sorted, gpu_idx = torch.sort(x.cuda())
gpu_sorted, gpu_idx = gpu_sorted.cpu(), gpu_idx.cpu()
print(f"sort([NaN, 3, NaN, 1, NaN]):")
print(f"CPU: {cpu_sorted.tolist()}")
print(f"GPU: {gpu_sorted.tolist()}")
cpu_nan_pos = [i for i, v in enumerate(cpu_sorted.tolist()) if v != v]
gpu_nan_pos = [i for i, v in enumerate(gpu_sorted.tolist()) if v != v]
if cpu_nan_pos != gpu_nan_pos:
    print(f"*** BUG: NaN positions differ — CPU: {cpu_nan_pos}, GPU: {gpu_nan_pos} ***")
elif cpu_idx.tolist() != gpu_idx.tolist():
    print(f"*** BUG: Different indices returned ***")
else:
    print("CPU and GPU agree")
