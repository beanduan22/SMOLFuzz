"""
Bug: tf.math.cumprod float16 GPU is 7x less accurate than CPU.

Root cause: TF's CPU cumprod kernel promotes float16 to float32 for
sequential multiplication, then rounds back to float16. TF's GPU kernel
performs cumprod natively in float16, accumulating 1000 multiplications
without precision promotion. The float16 mantissa (10 bits) cannot
represent the accumulated product accurately after many steps.
This is the same root cause as pt_cumprod_f16, confirming the bug
across frameworks.
"""
import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf

np.random.seed(0)
x_np = np.abs(np.random.randn(1000).astype(np.float16)) * 0.01 + 0.99
ref = np.cumprod(x_np.astype(np.float64)).astype(np.float32)

x_f16 = tf.constant(x_np, dtype=tf.float16)

with tf.device("/CPU:0"):
    cpu = tf.cast(tf.math.cumprod(x_f16), tf.float32).numpy()
with tf.device("/GPU:0"):
    gpu = tf.cast(tf.math.cumprod(x_f16), tf.float32).numpy()

cpu_err = np.linalg.norm(cpu - ref)
gpu_err = np.linalg.norm(gpu - ref)

print(f"CPU error vs float64 reference: {cpu_err:.4e}")
print(f"GPU error vs float64 reference: {gpu_err:.4e}   <-- BUG")
print(f"GPU is {gpu_err / cpu_err:.0f}x less accurate than CPU")
print()
print(f"Last 5 values:")
print(f"  ref: {ref[-5:].tolist()}")
print(f"  cpu: {cpu[-5:].tolist()}")
print(f"  gpu: {gpu[-5:].tolist()}")
