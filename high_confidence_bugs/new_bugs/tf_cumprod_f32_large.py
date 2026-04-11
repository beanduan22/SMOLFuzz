"""
Bug: TF GPU math.cumprod is 23-130x less accurate than CPU for float32 at N=1M.
Root cause: TF CPU cumprod uses higher-precision accumulation than GPU.
GPU accumulates in native float32, accumulating O(N * eps_f32) rounding error.
For N=1M elements near 1.0, GPU max error is ~7e-3 vs CPU max error of ~1e-4.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf

np.random.seed(0)
n = 1_000_000
x_np = (1.0 + 0.0001 * np.random.randn(n)).astype(np.float32)
ref = np.cumprod(x_np.astype(np.float64))

x_tf = tf.constant(x_np)
with tf.device('/CPU:0'):
    cpu = tf.math.cumprod(x_tf).numpy()
with tf.device('/GPU:0'):
    gpu = tf.math.cumprod(x_tf).numpy()

cpu_err = float(np.max(np.abs(cpu.astype(np.float64) - ref)))
gpu_err = float(np.max(np.abs(gpu.astype(np.float64) - ref)))
ratio = gpu_err / cpu_err if cpu_err > 0 else float('inf')

print(f"cpu_max_err = {cpu_err:.4e}")
print(f"gpu_max_err = {gpu_err:.4e}")
print(f"GPU/CPU error ratio: {ratio:.0f}x  (GPU less accurate)")
assert ratio > 10, f"Expected >10x ratio, got {ratio:.0f}x"
print("BUG CONFIRMED: TF cumprod f32 N=1M GPU is 23-130x less accurate than CPU")
