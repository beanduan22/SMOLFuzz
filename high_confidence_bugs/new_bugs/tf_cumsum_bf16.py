"""
Bug: tf.math.cumsum bfloat16 CPU is 23x less accurate than GPU.

Root cause: TF's CPU cumsum kernel accumulates natively in bfloat16
(7-bit mantissa, same exponent range as float32). TF's GPU CUDA kernel
internally promotes accumulation to float32, then rounds back to bfloat16,
greatly reducing rounding error over 10,000 additions.
This is consistent with tf_cumsum_f16 (CPU less accurate) but opposite to
PyTorch's bfloat16 cumsum (GPU less accurate).
"""
import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf

np.random.seed(0)
x_np = np.random.randn(10_000).astype(np.float32)
ref = np.cumsum(x_np.astype(np.float64)).astype(np.float32)

x_bf16 = tf.constant(x_np, dtype=tf.bfloat16)

with tf.device("/CPU:0"):
    cpu = tf.cast(tf.math.cumsum(x_bf16), tf.float32).numpy()
with tf.device("/GPU:0"):
    gpu = tf.cast(tf.math.cumsum(x_bf16), tf.float32).numpy()

cpu_err = np.linalg.norm(cpu - ref)
gpu_err = np.linalg.norm(gpu - ref)

print(f"CPU error vs float64 reference: {cpu_err:.4e}   <-- BUG")
print(f"GPU error vs float64 reference: {gpu_err:.4e}")
print(f"CPU is {cpu_err / gpu_err:.1f}x less accurate than GPU")
print()
print(f"Last 5 values:")
print(f"  ref: {ref[-5:].tolist()}")
print(f"  cpu: {cpu[-5:].tolist()}")
print(f"  gpu: {gpu[-5:].tolist()}")
