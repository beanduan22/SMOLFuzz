"""
Bug: tf.linalg.eigh CPU is 23x less accurate than GPU for float32 symmetric matrices.

Root cause: TF's CPU eigensolver uses LAPACK ssyevd (single-precision Divide & Conquer)
which suffers from float32 rounding. TF's GPU path calls cuSOLVER which internally
promotes to float64 during the computation, yielding eigenvalues much closer to the
float64 reference. This is the opposite of PyTorch's eigh behavior, where GPU is less
accurate.

Note: This is a bug in the CPU path — it should promote to float64 internally like
cuSOLVER does.
"""
import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf

np.random.seed(0)
n = 50
A = np.random.randn(n, n)
A_sym = A @ A.T                    # float64 symmetric
ref = np.linalg.eigh(A_sym)[0].astype(np.float32)   # float64 reference

A32 = tf.constant(A_sym.astype(np.float32))

with tf.device("/CPU:0"):
    cpu = tf.linalg.eigh(A32)[0].numpy()
with tf.device("/GPU:0"):
    gpu = tf.linalg.eigh(A32)[0].numpy()

cpu_err = np.linalg.norm(cpu - ref)
gpu_err = np.linalg.norm(gpu - ref)

print(f"CPU eigenvalue error vs float64 reference: {cpu_err:.4e}")
print(f"GPU eigenvalue error vs float64 reference: {gpu_err:.4e}")
print(f"CPU is {cpu_err / gpu_err:.0f}x less accurate than GPU")
print(f"CPU vs GPU L2: {np.linalg.norm(cpu - gpu):.4e}")
print()
print("First 5 eigenvalues:")
print(f"  CPU: {cpu[:5].tolist()}")
print(f"  GPU: {gpu[:5].tolist()}")
print(f"  Ref: {ref[:5].tolist()}")
