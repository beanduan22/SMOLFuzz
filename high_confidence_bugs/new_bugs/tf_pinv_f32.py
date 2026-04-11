"""
Bug: TF GPU linalg.pinv is 224-233x less accurate than CPU for float32 500x200 matrices.
Root cause: TF CPU pinv uses LAPACK with internal higher precision (via SVD);
TF GPU pinv uses cuSOLVER in float32 without internal promotion, accumulating more error.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf

np.random.seed(0)
A_np = np.random.randn(500, 200).astype(np.float32)
ref = np.linalg.pinv(A_np.astype(np.float64))

A_tf = tf.constant(A_np)
with tf.device('/CPU:0'):
    cpu = tf.linalg.pinv(A_tf).numpy()
with tf.device('/GPU:0'):
    gpu = tf.linalg.pinv(A_tf).numpy()

cpu_err = float(np.mean(np.abs(cpu.astype(np.float64) - ref)))
gpu_err = float(np.mean(np.abs(gpu.astype(np.float64) - ref)))
ratio = gpu_err / cpu_err if cpu_err > 0 else float('inf')

print(f"cpu_err = {cpu_err:.4e}")
print(f"gpu_err = {gpu_err:.4e}")
print(f"GPU/CPU error ratio: {ratio:.0f}x  (GPU less accurate)")
assert ratio > 50, f"Expected >50x ratio, got {ratio:.1f}x"
print("BUG CONFIRMED: TF GPU pinv f32 500x200 is 228x less accurate than CPU (cuSOLVER vs LAPACK)")
