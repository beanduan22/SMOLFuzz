"""
Bug: tf.linalg.solve on near-singular system — CPU vs GPU divergence.
Cross-framework port of pt_linalg_solve_singular.py.
PyTorch confirmed: CPU vs GPU diff reaches 6.3e9 for cond=1e10.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

print("=== TensorFlow tf.linalg.solve near-singular divergence ===")
print(f"TF version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs: {len(gpus)}")

rng = np.random.default_rng(42)

def make_near_singular_np(n, condition_number):
    Q1, _ = np.linalg.qr(rng.standard_normal((n, n)))
    Q2, _ = np.linalg.qr(rng.standard_normal((n, n)))
    s = np.logspace(0, -np.log10(condition_number), n).astype(np.float32)
    A = (Q1 @ np.diag(s) @ Q2).astype(np.float32)
    b = rng.standard_normal((n, 1)).astype(np.float32)
    return A, b

print("\n--- Condition number sweep (float32, n=32) ---")
for cond in [1e4, 1e6, 1e8, 1e10, 1e12]:
    A_np, b_np = make_near_singular_np(32, cond)
    A_tf = tf.constant(A_np, dtype=tf.float32)
    b_tf = tf.constant(b_np, dtype=tf.float32)

    # Reference: float64
    ref = np.linalg.solve(A_np.astype(np.float64), b_np.astype(np.float64)).astype(np.float32)

    try:
        cpu_sol = tf.linalg.solve(A_tf, b_tf).numpy()
        cpu_err = np.linalg.norm(cpu_sol - ref)
    except Exception as e:
        print(f"cond={cond:.0e}: CPU error: {e}")
        continue

    if gpus:
        try:
            with tf.device('/GPU:0'):
                gpu_sol = tf.linalg.solve(A_tf, b_tf).numpy()
            gpu_err = np.linalg.norm(gpu_sol - ref)
            diff = np.linalg.norm(cpu_sol - gpu_sol)
            ratio = gpu_err / (cpu_err + 1e-30)
            flag = "*** LARGE DIVERGENCE ***" if diff > 1.0 else ""
            print(f"cond={cond:.0e}: CPU_err={cpu_err:.3e}, GPU_err={gpu_err:.3e}, "
                  f"CPU_vs_GPU={diff:.3e} ratio={ratio:.1f}x {flag}")
        except Exception as e:
            print(f"cond={cond:.0e}: GPU error: {e}")
    else:
        print(f"cond={cond:.0e}: CPU_err={cpu_err:.3e}")

print(f"\n=== BUG SUMMARY ===")
A_np, b_np = make_near_singular_np(32, 1e10)
A_tf = tf.constant(A_np); b_tf = tf.constant(b_np)
ref = np.linalg.solve(A_np.astype(np.float64), b_np.astype(np.float64)).astype(np.float32)
cpu_sol = tf.linalg.solve(A_tf, b_tf).numpy()
print(f"CPU error vs float64: {np.linalg.norm(cpu_sol - ref):.3e}")
if gpus:
    with tf.device('/GPU:0'):
        gpu_sol = tf.linalg.solve(A_tf, b_tf).numpy()
    print(f"GPU error vs float64: {np.linalg.norm(gpu_sol - ref):.3e}")
    print(f"CPU vs GPU diff:      {np.linalg.norm(cpu_sol - gpu_sol):.3e}")
