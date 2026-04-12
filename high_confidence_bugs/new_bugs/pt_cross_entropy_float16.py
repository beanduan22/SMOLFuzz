"""
Bug: torch.nn.functional.cross_entropy with float16 logits.
Large logit values (>8 in float16) can cause exp() overflow.
CPU may upcast to float32 before the softmax, GPU may not.
Also: label_smoothing behavior with NaN logits.
"""
import torch
import torch.nn.functional as F
import numpy as np

print("=== PyTorch cross_entropy float16 CPU vs GPU ===")
print(f"PyTorch version: {torch.__version__}")

if not torch.cuda.is_available():
    print("No CUDA"); exit(0)

rng = np.random.default_rng(42)
torch.manual_seed(42)

print("\n--- float16 cross_entropy: various logit scales ---")
for scale in [1.0, 4.0, 8.0, 9.0, 10.0, 12.0, 16.0, 32.0]:
    x_np = rng.standard_normal((64, 1000)).astype(np.float16) * scale
    y_np = rng.integers(0, 1000, size=64).astype(np.int64)
    x_f16 = torch.tensor(x_np, dtype=torch.float16)
    y = torch.tensor(y_np)

    try:
        cpu_loss = F.cross_entropy(x_f16, y).float().item()
        gpu_loss = F.cross_entropy(x_f16.cuda(), y.cuda()).float().cpu().item()

        cpu_nan = np.isnan(cpu_loss)
        gpu_nan = np.isnan(gpu_loss)
        diff = abs(cpu_loss - gpu_loss)

        print(f"scale={scale:.1f}: CPU={cpu_loss:.4f}(nan={cpu_nan}), GPU={gpu_loss:.4f}(nan={gpu_nan}), diff={diff:.4f}", end="")
        if cpu_nan != gpu_nan:
            print(f"  *** BUG: CPU={'NaN' if cpu_nan else 'finite'}, GPU={'NaN' if gpu_nan else 'finite'} ***", end="")
        elif diff > 0.5:
            print(f"  *** SIGNIFICANT DIVERGENCE ***", end="")
        print()
    except Exception as e:
        print(f"scale={scale:.1f}: ERROR: {e}")

# With NaN logits
print("\n--- float16 cross_entropy with NaN logits ---")
for nan_frac in [0.001, 0.01, 0.1]:
    x_np = rng.standard_normal((64, 1000)).astype(np.float16)
    # Insert NaN in some logits
    nan_positions = rng.integers(0, 64*1000, size=int(64*1000*nan_frac))
    x_np.flat[nan_positions] = float('nan')
    y_np = rng.integers(0, 1000, size=64).astype(np.int64)
    x_f16 = torch.tensor(x_np, dtype=torch.float16)
    y = torch.tensor(y_np)

    try:
        cpu_loss = F.cross_entropy(x_f16, y).float().item()
        gpu_loss = F.cross_entropy(x_f16.cuda(), y.cuda()).float().cpu().item()
        cpu_nan = np.isnan(cpu_loss)
        gpu_nan = np.isnan(gpu_loss)
        print(f"nan_frac={nan_frac}: CPU={cpu_loss:.4f}(nan={cpu_nan}), GPU={gpu_loss:.4f}(nan={gpu_nan})", end="")
        if cpu_nan != gpu_nan:
            print(f"  *** BUG ***", end="")
        print()
    except Exception as e:
        print(f"nan_frac={nan_frac}: ERROR: {e}")

# Very large number of classes (overflow likely)
print("\n--- float16 cross_entropy large num_classes ---")
for num_classes in [10000, 50000, 100000]:
    x_np = rng.standard_normal((16, num_classes)).astype(np.float16)
    y_np = rng.integers(0, num_classes, size=16).astype(np.int64)
    x_f16 = torch.tensor(x_np, dtype=torch.float16)
    y = torch.tensor(y_np)

    try:
        cpu_loss = F.cross_entropy(x_f16, y).float().item()
        gpu_loss = F.cross_entropy(x_f16.cuda(), y.cuda()).float().cpu().item()
        cpu_nan = np.isnan(cpu_loss)
        gpu_nan = np.isnan(gpu_loss)
        diff = abs(cpu_loss - gpu_loss)
        print(f"num_classes={num_classes}: CPU={cpu_loss:.4f}(nan={cpu_nan}), GPU={gpu_loss:.4f}(nan={gpu_nan}), diff={diff:.4f}", end="")
        if cpu_nan != gpu_nan:
            print(f"  *** BUG ***", end="")
        elif diff > 0.5:
            print(f"  *** DIVERGENCE ***", end="")
        print()
    except Exception as e:
        print(f"num_classes={num_classes}: ERROR: {e}")

# bfloat16 cross_entropy
print("\n--- bfloat16 cross_entropy ---")
for scale in [1.0, 8.0, 32.0, 100.0]:
    x_np = rng.standard_normal((64, 1000)).astype(np.float32) * scale
    y_np = rng.integers(0, 1000, size=64).astype(np.int64)
    x_bf16 = torch.tensor(x_np, dtype=torch.bfloat16)
    y = torch.tensor(y_np)
    ref_loss = F.cross_entropy(torch.tensor(x_np, dtype=torch.float32), y).item()

    try:
        cpu_loss = F.cross_entropy(x_bf16, y).float().item()
        gpu_loss = F.cross_entropy(x_bf16.cuda(), y.cuda()).float().cpu().item()
        diff = abs(cpu_loss - gpu_loss)
        cpu_err = abs(cpu_loss - ref_loss)
        gpu_err = abs(gpu_loss - ref_loss)
        print(f"bf16 scale={scale}: CPU_err={cpu_err:.4f}, GPU_err={gpu_err:.4f}, diff={diff:.4f}", end="")
        if diff > 0.5:
            print(f"  *** DIVERGENCE ***", end="")
        print()
    except Exception as e:
        print(f"bf16 scale={scale}: ERROR: {e}")

print(f"\n=== BUG SUMMARY ===")
# bfloat16 at large scale shows GPU significantly more accurate (CPU stays in bf16 for softmax)
scale = 100.0
x_np = rng.standard_normal((64, 1000)).astype(np.float32) * scale
y_np = rng.integers(0, 1000, size=64).astype(np.int64)
x_bf16 = torch.tensor(x_np, dtype=torch.bfloat16)
y = torch.tensor(y_np)
ref_loss = F.cross_entropy(torch.tensor(x_np, dtype=torch.float32), y).item()
cpu_loss = F.cross_entropy(x_bf16, y).float().item()
gpu_loss = F.cross_entropy(x_bf16.cuda(), y.cuda()).float().cpu().item()
cpu_err = abs(cpu_loss - ref_loss)
gpu_err = abs(gpu_loss - ref_loss)
diff = abs(cpu_loss - gpu_loss)
print(f"bfloat16 cross_entropy scale={scale}:")
print(f"ref={ref_loss:.4f}, CPU={cpu_loss:.4f}, GPU={gpu_loss:.4f}")
print(f"CPU_err={cpu_err:.4f}, GPU_err={gpu_err:.4f}, diff={diff:.4f}")
ratio = cpu_err / (gpu_err + 1e-10)
if ratio > 3 and diff > 0.5:
    print(f"*** BUG: CPU {ratio:.1f}x less accurate than GPU (CPU stays in bfloat16 for softmax) ***")
elif diff > 0.5:
    print(f"*** SIGNIFICANT CPU/GPU DIVERGENCE: {diff:.4f} ***")
else:
    print("No significant divergence")
