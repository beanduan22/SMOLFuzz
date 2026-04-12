"""
Bug: torch.linalg.solve on a near-singular system — GPU may return a solution
with completely wrong sign or magnitude while CPU is close to correct.

Source: Known linalg CPU/GPU divergence pattern (see pt_lstsq_rankdef.py).
torch.linalg.solve uses LAPACK gesv (CPU) vs cuSOLVER gesv (GPU).
For near-singular systems, small pivoting differences → very different solutions.

Cross-framework: tf_linalg_solve_singular.py
"""
import torch
import numpy as np

print("=== PyTorch torch.linalg.solve near-singular divergence ===")
print(f"PyTorch version: {torch.__version__}")

if not torch.cuda.is_available():
    print("No CUDA available")
    exit(0)

torch.manual_seed(42)
rng = np.random.default_rng(42)

def make_near_singular(n, condition_number=1e10, dtype=torch.float32):
    """Create a near-singular system A @ x = b."""
    Q1, _ = np.linalg.qr(rng.standard_normal((n, n)))
    Q2, _ = np.linalg.qr(rng.standard_normal((n, n)))
    # Singular values: from 1.0 down to 1/condition_number
    s = np.logspace(0, -np.log10(condition_number), n).astype(np.float32)
    A_np = (Q1 @ np.diag(s) @ Q2).astype(np.float32)
    b_np = rng.standard_normal((n, 1)).astype(np.float32)
    return torch.tensor(A_np, dtype=dtype), torch.tensor(b_np, dtype=dtype)

# Test various condition numbers
print("\n--- Condition number sweep (float32) ---")
for cond in [1e4, 1e6, 1e8, 1e10, 1e12, 1e15]:
    try:
        A, b = make_near_singular(32, cond)
        # Reference: float64
        ref_sol = torch.linalg.solve(A.double(), b.double()).float()

        cpu_sol = torch.linalg.solve(A, b)
        gpu_sol = torch.linalg.solve(A.cuda(), b.cuda()).cpu()

        cpu_err = (cpu_sol - ref_sol).norm().item()
        gpu_err = (gpu_sol - ref_sol).norm().item()
        cpu_gpu_diff = (cpu_sol - gpu_sol).norm().item()
        ratio = gpu_err / (cpu_err + 1e-30)

        cpu_nan = torch.isnan(cpu_sol).any().item()
        gpu_nan = torch.isnan(gpu_sol).any().item()
        cpu_inf = torch.isinf(cpu_sol).any().item()
        gpu_inf = torch.isinf(gpu_sol).any().item()

        flag = ""
        if cpu_nan != gpu_nan or cpu_inf != gpu_inf:
            flag = "*** NaN/inf DIVERGENCE ***"
        elif cpu_gpu_diff > 1.0:
            flag = f"*** LARGE DIVERGENCE ***"

        print(f"cond={cond:.0e}: CPU_err={cpu_err:.3e}, GPU_err={gpu_err:.3e}, "
              f"CPU_vs_GPU={cpu_gpu_diff:.3e} ratio={ratio:.1f}x {flag}")

    except Exception as e:
        print(f"cond={cond:.0e}: ERROR — {type(e).__name__}: {str(e)[:60]}")

# Test 2: Batch solve with near-singular matrices
print("\n--- Batch solve (batch_size=16, n=16) ---")
batch = 16
A_batch = []
b_batch = []
for _ in range(batch):
    A, b = make_near_singular(16, 1e8)
    A_batch.append(A)
    b_batch.append(b)

A_batched = torch.stack(A_batch)
b_batched = torch.stack(b_batch)
ref_batched = torch.linalg.solve(A_batched.double(), b_batched.double()).float()

cpu_batch_sol = torch.linalg.solve(A_batched, b_batched)
gpu_batch_sol = torch.linalg.solve(A_batched.cuda(), b_batched.cuda()).cpu()

batch_cpu_err = (cpu_batch_sol - ref_batched).norm().item()
batch_gpu_err = (gpu_batch_sol - ref_batched).norm().item()
batch_diff = (cpu_batch_sol - gpu_batch_sol).norm().item()

print(f"Batch CPU_err={batch_cpu_err:.3e}, GPU_err={batch_gpu_err:.3e}, diff={batch_diff:.3e}")
if batch_diff > 0.1:
    print(f"*** SIGNIFICANT BATCH DIVERGENCE ***")

# Test 3: Specifically crafted near-singular (echo of lstsq bug)
print("\n--- Rank-deficient system (from lstsq bug) ---")
for n in [3, 8, 16]:
    # Vandermonde-like near-singular matrix
    x_pts = np.linspace(0, 1, n).astype(np.float32)
    A_np = np.vander(x_pts, n, increasing=True).astype(np.float32)  # near-singular for large n
    b_np = rng.standard_normal((n, 1)).astype(np.float32)

    A = torch.tensor(A_np, dtype=torch.float32)
    b = torch.tensor(b_np, dtype=torch.float32)

    try:
        ref_sol = torch.linalg.solve(A.double(), b.double()).float()
        cpu_sol = torch.linalg.solve(A, b)
        gpu_sol = torch.linalg.solve(A.cuda(), b.cuda()).cpu()

        cpu_err = (cpu_sol - ref_sol).norm().item()
        gpu_err = (gpu_sol - ref_sol).norm().item()
        cpu_gpu = (cpu_sol - gpu_sol).norm().item()
        ratio = gpu_err / (cpu_err + 1e-30)

        print(f"n={n} Vandermonde: CPU_err={cpu_err:.3e}, GPU_err={gpu_err:.3e}, "
              f"CPU_vs_GPU={cpu_gpu:.3e} ratio={ratio:.1f}x")
        if cpu_gpu > 1.0:
            print(f"  *** SIGNIFICANT DIVERGENCE ***")
    except Exception as e:
        print(f"n={n}: {type(e).__name__}: {str(e)[:60]}")

print(f"\n=== BUG SUMMARY ===")
A, b = make_near_singular(32, 1e12)
ref = torch.linalg.solve(A.double(), b.double()).float()
cpu_sol = torch.linalg.solve(A, b)
gpu_sol = torch.linalg.solve(A.cuda(), b.cuda()).cpu()
print(f"32x32 float32, cond=1e12:")
print(f"CPU error vs float64: {(cpu_sol - ref).norm().item():.3e}")
print(f"GPU error vs float64: {(gpu_sol - ref).norm().item():.3e}")
print(f"CPU vs GPU diff: {(cpu_sol - gpu_sol).norm().item():.3e}")
print(f"CPU NaN/inf: {torch.isnan(cpu_sol).any().item()} / {torch.isinf(cpu_sol).any().item()}")
print(f"GPU NaN/inf: {torch.isnan(gpu_sol).any().item()} / {torch.isinf(gpu_sol).any().item()}")
