"""
Bug: tf.linalg.logdet on a singular/near-singular matrix — CPU returns -inf,
GPU returns a finite (wrong) value.
Source: Cross-framework port of PyTorch bug (pt396.py in high_confidence_bugs/minimal/)
        PyTorch: CPU=-inf, GPU=-108.34 for logdet of rank-1 matrix.

Root cause: LAPACK (CPU) correctly identifies near-zero pivots → -inf.
cuSOLVER (GPU) uses different LU pivoting strategy → reports finite log|det|.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

print("=== TensorFlow tf.linalg.logdet singular matrix bug ===")
print(f"TF version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs available: {len(gpus)}")

# Test 1: Rank-1 matrix (determinant = 0, logdet = -inf)
print("\n--- Test 1: Rank-1 matrices ---")
for n in [4, 8, 16, 32]:
    # Uniform matrix → rank 1 → det = 0 → logdet = -inf
    A_np = np.ones((n, n), dtype=np.float32)
    A_tf = tf.constant(A_np, dtype=tf.float32)

    ref = np.log(np.abs(np.linalg.det(A_np)))  # should be -inf
    cpu_out = tf.linalg.logdet(A_tf).numpy()

    print(f"n={n}: NumPy logdet={ref:.4f}, TF CPU={cpu_out:.4f}")

    if gpus:
        with tf.device('/GPU:0'):
            gpu_out = tf.linalg.logdet(A_tf).numpy()
        cpu_inf = np.isinf(cpu_out)
        gpu_inf = np.isinf(gpu_out)
        print(f"  TF GPU={gpu_out:.4f}, CPU_inf={cpu_inf}, GPU_inf={gpu_inf}")
        if cpu_inf and not gpu_inf:
            print(f"  *** BUG: CPU=-inf, GPU={gpu_out:.4f} (finite, wrong) ***")
        if not cpu_inf and not gpu_inf:
            diff = abs(cpu_out - gpu_out)
            if diff > 1.0:
                print(f"  *** DIVERGENCE: CPU vs GPU diff={diff:.3e} ***")

# Test 2: Near-singular matrix via arccosh activations (mirrors PyTorch pt396)
print("\n--- Test 2: Near-singular via neural-like ops ---")

def make_near_singular_tf(n, device='CPU'):
    """Replicate the PyTorch pt396 pattern in TensorFlow."""
    tf.random.set_seed(42)
    with tf.device(f'/{device}:0'):
        x = tf.fill([n, n], 0.4033372700214386)
        # arccosh(x + 2): requires x > -1, arccosh defined for input >= 1
        y = tf.math.acosh(x + 2.0)
        # Linear transform to create near-singular matrix
        W = tf.constant(np.ones((n, n), dtype=np.float32) * 0.1)
        z = tf.linalg.matmul(y, W)
        return tf.linalg.logdet(z)

for n in [4, 8]:
    cpu_ld = make_near_singular_tf(n, 'CPU').numpy()
    print(f"\nn={n}: TF CPU logdet = {cpu_ld:.6f}")
    if gpus:
        gpu_ld = make_near_singular_tf(n, 'GPU').numpy()
        print(f"n={n}: TF GPU logdet = {gpu_ld:.6f}")
        cpu_inf = np.isinf(cpu_ld) or np.isnan(cpu_ld)
        gpu_inf = np.isinf(gpu_ld) or np.isnan(gpu_ld)
        print(f"  CPU isinf/isnan: {cpu_inf}, GPU isinf/isnan: {gpu_inf}")
        if cpu_inf and not gpu_inf:
            print(f"  *** BUG: CPU=-inf/NaN, GPU={gpu_ld:.4f} (finite, wrong) ***")
        diff = abs(cpu_ld - gpu_ld)
        if diff > 1.0:
            print(f"  *** DIVERGENCE: diff={diff:.3e} ***")

# Test 3: Matrix with exactly zero eigenvalue
print("\n--- Test 3: Matrix with zero eigenvalue ---")
for n in [8, 16]:
    np.random.seed(42)
    # Create matrix with one zero eigenvalue
    vecs = np.random.randn(n, n).astype(np.float32)
    eigs = np.abs(np.random.randn(n).astype(np.float32)) + 0.1
    eigs[0] = 0.0  # force one zero eigenvalue → det=0 → logdet=-inf
    A_np = vecs @ np.diag(eigs) @ np.linalg.inv(vecs)
    A_tf = tf.constant(A_np, dtype=tf.float32)

    ref = np.log(np.abs(np.linalg.det(A_np)))
    cpu_ld = tf.linalg.logdet(A_tf).numpy()
    print(f"\nn={n}: NumPy={ref:.4f}, TF CPU={cpu_ld:.4f}")

    if gpus:
        with tf.device('/GPU:0'):
            gpu_ld = tf.linalg.logdet(A_tf).numpy()
        print(f"TF GPU={gpu_ld:.4f}")
        if (np.isinf(cpu_ld) or np.isnan(cpu_ld)) and not (np.isinf(gpu_ld) or np.isnan(gpu_ld)):
            print(f"*** BUG: CPU=-inf/NaN but GPU={gpu_ld:.4f} ***")
        diff = abs(float(cpu_ld) - float(gpu_ld)) if not (np.isinf(cpu_ld) or np.isinf(gpu_ld)) else float('inf')
        if not np.isinf(diff) and diff > 1.0:
            print(f"*** DIVERGENCE: CPU={cpu_ld:.4f} GPU={gpu_ld:.4f} diff={diff:.3e} ***")

print(f"\n=== BUG SUMMARY ===")
A = tf.constant(np.ones((8, 8), dtype=np.float32))
cpu_ld = tf.linalg.logdet(A).numpy()
print(f"Rank-1 matrix (8x8 ones): TF CPU logdet = {cpu_ld}")
if gpus:
    with tf.device('/GPU:0'):
        gpu_ld = tf.linalg.logdet(A).numpy()
    print(f"Rank-1 matrix (8x8 ones): TF GPU logdet = {gpu_ld}")
    print(f"CPU is -inf: {np.isinf(cpu_ld) and cpu_ld < 0}")
    print(f"GPU is -inf: {np.isinf(gpu_ld) and gpu_ld < 0}")
    if (np.isinf(cpu_ld) and cpu_ld < 0) and not (np.isinf(gpu_ld) and gpu_ld < 0):
        print("*** BUG CONFIRMED: CPU=-inf, GPU=finite ***")
