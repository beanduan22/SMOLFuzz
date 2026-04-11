"""
Bug: tf.linalg.eigh CPU is 49x less accurate than GPU for complex64 (Hermitian) matrices.

Root cause: TF's CPU eigensolver for complex64 calls LAPACK cheevd in single
precision. TF's GPU path uses cuSOLVER which internally promotes complex64 to
complex128 for Hermitian eigendecomposition, producing eigenvalues much closer
to the complex128 reference.

This is the complex64 version of tf_eigh_float32 (real float32, CPU 23x less accurate).
Both bugs show TF GPU being more accurate than CPU due to cuSOLVER's internal
precision promotion — the opposite direction from PyTorch's eigh bug.
"""
import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf

np.random.seed(0)
n = 200
A_np = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(np.complex64)
A_np = A_np + A_np.conj().T  # Hermitian

ref = np.linalg.eigvalsh(A_np.astype(np.complex128)).astype(np.float32)
A_tf = tf.constant(A_np, dtype=tf.complex64)

with tf.device("/CPU:0"):
    cpu = tf.linalg.eigvalsh(A_tf).numpy()
with tf.device("/GPU:0"):
    gpu = tf.linalg.eigvalsh(A_tf).numpy()

cpu_err = np.linalg.norm(cpu - ref)
gpu_err = np.linalg.norm(gpu - ref)

print(f"CPU eigenvalue error vs complex128 reference: {cpu_err:.4e}   <-- BUG")
print(f"GPU eigenvalue error vs complex128 reference: {gpu_err:.4e}")
print(f"CPU is {cpu_err / gpu_err:.0f}x less accurate than GPU")
print()
print(f"First 5 eigenvalues:")
print(f"  CPU: {cpu[:5].tolist()}")
print(f"  GPU: {gpu[:5].tolist()}")
print(f"  Ref: {ref[:5].tolist()}")
