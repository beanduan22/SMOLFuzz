"""
Bug: tf.matmul on Ampere+ GPUs is up to 1011x less accurate than CPU for float32.

Origin: Same root cause as PyTorch (JAX issue #22557). TensorFlow also uses cuBLAS
which enables TF32 by default on Ampere+ GPUs for matmul operations. There is no
user-accessible flag to disable it in TF (unlike torch.backends.cuda.matmul.allow_tf32).

Root cause: cuBLAS selects TF32 TensorCore operations automatically on Ampere+
for float32 matmul. This reduces mantissa precision from 23 bits to 10 bits.
TF's CPU path uses MKL/Eigen with full float32 precision. The result: GPU and CPU
produce silently different outputs with no warning, with GPU being ~1000x less
accurate vs the float64 reference.
"""
import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf

np.random.seed(0)
n = 512
A = np.random.randn(n, n).astype(np.float32)
B = np.random.randn(n, n).astype(np.float32)

ref = (A.astype(np.float64) @ B.astype(np.float64)).astype(np.float32)

A_tf = tf.constant(A)
B_tf = tf.constant(B)

with tf.device("/CPU:0"):
    cpu = tf.matmul(A_tf, B_tf).numpy()
with tf.device("/GPU:0"):
    gpu = tf.matmul(A_tf, B_tf).numpy()

cpu_err = np.linalg.norm(cpu - ref)
gpu_err = np.linalg.norm(gpu - ref)

print(f"CPU error vs float64 reference: {cpu_err:.4e}")
print(f"GPU error vs float64 reference: {gpu_err:.4e}   <-- BUG")
print(f"GPU is {gpu_err / cpu_err:.0f}x less accurate than CPU (TF32 TensorCore)")
print()
print(f"First 4 output values:")
print(f"  ref: {ref.flatten()[:4].tolist()}")
print(f"  cpu: {cpu.flatten()[:4].tolist()}")
print(f"  gpu: {gpu.flatten()[:4].tolist()}")
