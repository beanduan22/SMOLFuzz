"""
Bug: tf.math.cumsum CPU is 12x less accurate than GPU for float16 inputs.

Root cause: TF's CPU cumsum kernel operates natively in float16 without internal
precision promotion. TF's GPU CUDA kernel promotes to float32 for accumulation,
greatly reducing rounding error. The result: CPU produces 12x more numerical
error than GPU vs the float64 reference.

Note: This is the opposite of PyTorch's cumsum float16 bug, where GPU is less
accurate than CPU.
"""
import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf

np.random.seed(0)
n = 10_000
x16_np = np.random.randn(n).astype(np.float16)
x16 = tf.constant(x16_np)

ref = np.cumsum(x16_np.astype(np.float64)).astype(np.float32)

with tf.device("/CPU:0"):
    cpu = tf.math.cumsum(x16).numpy().astype(np.float32)
with tf.device("/GPU:0"):
    gpu = tf.math.cumsum(x16).numpy().astype(np.float32)

cpu_err = np.linalg.norm(cpu - ref)
gpu_err = np.linalg.norm(gpu - ref)

print(f"CPU error vs float64 reference: {cpu_err:.4e}")
print(f"GPU error vs float64 reference: {gpu_err:.4e}")
print(f"CPU is {cpu_err / gpu_err:.1f}x less accurate than GPU")
print(f"CPU vs GPU L2: {np.linalg.norm(cpu - gpu):.4e}")
