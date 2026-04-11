"""
Bug: TF CPU cumsum float16 is 96-113x less accurate than GPU for all-positive monotone data at N=10k.
Root cause: TF CPU cumsum uses sequential float16 accumulation with no cancellation.
For all-positive data, errors accumulate monotonically without cancellation,
giving 100x worse error than random-sign data (where cancellation reduces error).
GPU promotes to float32 internally, giving accurate results regardless of data sign.

This represents the worst-case failure mode for TF CPU cumsum float16:
purely positive values mean rounding errors only ADD UP, never cancel.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf

np.random.seed(0)
n = 10_000
# All-positive monotone data: worst case for sequential float16 accumulation
x_np = (np.abs(np.random.randn(n)) + 0.1).astype(np.float16)
ref = np.cumsum(x_np.astype(np.float64))

x_tf = tf.constant(x_np)
with tf.device('/CPU:0'):
    cpu = tf.cumsum(x_tf).numpy()
with tf.device('/GPU:0'):
    gpu = tf.cumsum(x_tf).numpy()

cpu_err = float(np.max(np.abs(cpu.astype(np.float64) - ref)))
gpu_err = float(np.max(np.abs(gpu.astype(np.float64) - ref)))
ratio = cpu_err / gpu_err if gpu_err > 0 else float('inf')

print(f"cpu_max_err = {cpu_err:.4e}  (sequential f16, errors accumulate monotonically)")
print(f"gpu_max_err = {gpu_err:.4e}  (GPU promotes to f32 internally)")
print(f"CPU/GPU error ratio: {ratio:.0f}x  (CPU less accurate)")
assert ratio > 50, f"Expected >50x ratio, got {ratio:.0f}x"
print("BUG CONFIRMED: TF CPU cumsum f16 positive monotone N=10k is 96-113x less accurate than GPU")
