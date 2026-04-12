"""
Bug: tf.linalg.lstsq with complex64 rank-deficient matrix — GPU error catastrophically
larger than CPU.
Source: Cross-framework port of PyTorch bug (pt_lstsq_complex64.py, pt_lstsq_rankdef.py)
        PyTorch: GPU L2 error = 1.9e6 vs CPU = 9.85e-7 for complex64 lstsq.

TensorFlow's tf.linalg.lstsq wraps cuSOLVER's complex least-squares path on GPU.
Same cuSOLVER driver selection issue as PyTorch.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

print("=== TensorFlow tf.linalg.lstsq complex64 rank-deficient bug ===")
print(f"TF version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs available: {len(gpus)}")

rng = np.random.default_rng(42)

def lstsq_err(A, b, solution):
    """Compute ||A @ x - b||_F relative to ||b||_F"""
    residual = np.linalg.norm(A @ solution - b)
    return residual

# Test 1: Rank-deficient complex64 (exact rank deficiency)
print("\n--- Test 1: Exact rank-deficient complex64 ---")
for n in [3, 8, 16]:
    # Rank-1 complex matrix
    v = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
    A_np = np.outer(v, v.conj()) / np.linalg.norm(v)**2
    b_np = v.reshape(-1, 1).astype(np.complex64)

    A_tf = tf.constant(A_np, dtype=tf.complex64)
    b_tf = tf.constant(b_np, dtype=tf.complex64)

    # Reference: NumPy lstsq
    ref_sol, _, _, _ = np.linalg.lstsq(A_np, b_np, rcond=None)
    ref_err = lstsq_err(A_np, b_np, ref_sol)

    # TF CPU
    try:
        cpu_sol = tf.linalg.lstsq(A_tf, b_tf, fast=False).numpy()
        cpu_err = lstsq_err(A_np, b_np, cpu_sol)
        print(f"\nn={n}: NumPy err={ref_err:.3e}, TF CPU err={cpu_err:.3e}")
    except Exception as e:
        print(f"\nn={n}: TF CPU error: {e}")
        continue

    if gpus:
        try:
            with tf.device('/GPU:0'):
                gpu_sol = tf.linalg.lstsq(A_tf, b_tf, fast=False).numpy()
            gpu_err = lstsq_err(A_np, b_np, gpu_sol)
            print(f"  TF GPU err={gpu_err:.3e}")
            ratio = gpu_err / (cpu_err + 1e-30)
            if ratio > 100:
                print(f"  *** BUG: GPU {ratio:.0f}x worse than CPU ***")
        except Exception as e:
            print(f"  GPU error: {type(e).__name__}: {e}")

# Test 2: Near-singular complex64 (mirrors PyTorch pt_lstsq_complex64.py)
print("\n--- Test 2: Near-singular complex64 (3x3 rank-deficient) ---")
A_np = np.array([[1+0j, 2+0j, 3+0j],
                  [4+0j, 5+0j, 6+0j],
                  [7+0j, 8+0j, 9+0j]], dtype=np.complex64)
b_np = (rng.standard_normal(3) + 1j * rng.standard_normal(3)).astype(np.complex64).reshape(-1, 1)

A_tf = tf.constant(A_np, dtype=tf.complex64)
b_tf = tf.constant(b_np, dtype=tf.complex64)

ref_sol, _, _, _ = np.linalg.lstsq(A_np.astype(np.complex128), b_np.astype(np.complex128), rcond=None)
ref_err = lstsq_err(A_np, b_np, ref_sol.astype(np.complex64))

try:
    cpu_sol = tf.linalg.lstsq(A_tf, b_tf, fast=False).numpy()
    cpu_err = lstsq_err(A_np, b_np, cpu_sol)
    print(f"NumPy err={ref_err:.3e}")
    print(f"TF CPU err={cpu_err:.3e}")

    if gpus:
        with tf.device('/GPU:0'):
            gpu_sol = tf.linalg.lstsq(A_tf, b_tf, fast=False).numpy()
        gpu_err = lstsq_err(A_np, b_np, gpu_sol)
        print(f"TF GPU err={gpu_err:.3e}")
        ratio = gpu_err / (cpu_err + 1e-30)
        print(f"GPU/CPU ratio: {ratio:.1f}x")
        if ratio > 100:
            print(f"*** BUG: GPU {ratio:.0f}x worse ***")

except Exception as e:
    print(f"Error: {e}")

# Test 3: fast=True vs fast=False (Cholesky vs QR path)
print("\n--- Test 3: fast=True vs fast=False on rank-deficient complex64 ---")
A_tf = tf.constant(A_np, dtype=tf.complex64)
b_tf = tf.constant(b_np, dtype=tf.complex64)

for fast in [True, False]:
    try:
        cpu_sol = tf.linalg.lstsq(A_tf, b_tf, fast=fast).numpy()
        cpu_err = lstsq_err(A_np, b_np, cpu_sol)
        print(f"CPU fast={fast}: err={cpu_err:.3e}")
        if gpus:
            with tf.device('/GPU:0'):
                gpu_sol = tf.linalg.lstsq(A_tf, b_tf, fast=fast).numpy()
            gpu_err = lstsq_err(A_np, b_np, gpu_sol)
            print(f"GPU fast={fast}: err={gpu_err:.3e}", end="")
            ratio = gpu_err / (cpu_err + 1e-30)
            if ratio > 100:
                print(f"  *** BUG: GPU {ratio:.0f}x worse ***", end="")
            print()
    except Exception as e:
        print(f"fast={fast}: {type(e).__name__}: {e}")

print(f"\n=== BUG SUMMARY ===")
A_np = np.array([[1+0j,2+0j,3+0j],[4+0j,5+0j,6+0j],[7+0j,8+0j,9+0j]], dtype=np.complex64)
b_np = np.array([[1+0j],[0+0j],[0+0j]], dtype=np.complex64)
A_tf = tf.constant(A_np); b_tf = tf.constant(b_np)
ref, _, _, _ = np.linalg.lstsq(A_np, b_np, rcond=None)
cpu_sol = tf.linalg.lstsq(A_tf, b_tf, fast=False).numpy()
print(f"CPU L2 error vs NumPy: {np.linalg.norm(cpu_sol - ref):.3e}")
if gpus:
    with tf.device('/GPU:0'):
        gpu_sol = tf.linalg.lstsq(A_tf, b_tf, fast=False).numpy()
    print(f"GPU L2 error vs NumPy: {np.linalg.norm(gpu_sol - ref):.3e}")
