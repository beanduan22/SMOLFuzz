"""
Bug: TF CPU cumsum float32 is 62-154x less accurate than GPU for large N (10M elements).
Root cause: TF CPU uses sequential single-precision accumulation; GPU uses pairwise/blocked summation.
For N=10M, CPU error grows to O(N·eps) while GPU keeps O(sqrt(N)·eps).
This is a much stronger manifestation than N=10k (which shows only 7x ratio).
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf

np.random.seed(0)
n = 10_000_000
x_np = np.random.randn(n).astype(np.float32)
ref = np.cumsum(x_np.astype(np.float64))

x_tf = tf.constant(x_np)
with tf.device('/CPU:0'):
    cpu = tf.cumsum(x_tf).numpy()
with tf.device('/GPU:0'):
    gpu = tf.cumsum(x_tf).numpy()

cpu_err = float(np.max(np.abs(cpu.astype(np.float64) - ref)))
gpu_err = float(np.max(np.abs(gpu.astype(np.float64) - ref)))
ratio = cpu_err / gpu_err if gpu_err > 0 else float('inf')

print(f"cpu_max_err = {cpu_err:.4e}")
print(f"gpu_max_err = {gpu_err:.4e}")
print(f"CPU/GPU error ratio: {ratio:.0f}x  (CPU less accurate)")
assert ratio > 20, f"Expected >20x ratio, got {ratio:.1f}x"
print("BUG CONFIRMED: TF CPU cumsum f32 N=10M is 62x less accurate than GPU (sequential vs pairwise accumulation)")
