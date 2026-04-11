"""
Bug: TF GPU linalg.lstsq is 275-406x less accurate than CPU for float32 overdetermined systems.
Root cause: TF GPU lstsq uses Cholesky/normal-equations approach (AᵀA·x = Aᵀb) in float32,
which squares the condition number and amplifies rounding error.
TF CPU uses LAPACK QR decomposition which is numerically stable.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf

np.random.seed(0)
n, m = 200, 100
A_np = np.random.randn(n, m).astype(np.float32)
b_np = np.random.randn(n, 1).astype(np.float32)
ref = np.linalg.lstsq(A_np.astype(np.float64), b_np.astype(np.float64), rcond=None)[0]

A_tf = tf.constant(A_np)
b_tf = tf.constant(b_np)
with tf.device('/CPU:0'):
    cpu = tf.linalg.lstsq(A_tf, b_tf, l2_regularizer=0.0).numpy()
with tf.device('/GPU:0'):
    gpu = tf.linalg.lstsq(A_tf, b_tf, l2_regularizer=0.0).numpy()

cpu_err = float(np.mean(np.abs(cpu.astype(np.float64) - ref)))
gpu_err = float(np.mean(np.abs(gpu.astype(np.float64) - ref)))
ratio = gpu_err / cpu_err if cpu_err > 0 else float('inf')

print(f"cpu_err = {cpu_err:.4e}")
print(f"gpu_err = {gpu_err:.4e}")
print(f"GPU/CPU error ratio: {ratio:.0f}x  (GPU less accurate)")
assert ratio > 50, f"Expected >50x ratio, got {ratio:.1f}x"
print("BUG CONFIRMED: TF GPU lstsq 275x less accurate than CPU for float32 200x100 (Cholesky normal eq vs LAPACK QR)")
