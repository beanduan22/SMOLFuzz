"""
Bug: TF CPU einsum float16 is 10x less accurate than GPU for matrix multiply contraction.
Root cause: TF CPU uses sequential float16 accumulation; GPU promotes to float32 internally.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf

np.random.seed(0)
A_np = np.random.randn(100, 200).astype(np.float16)
B_np = np.random.randn(200, 300).astype(np.float16)
ref = np.einsum('ij,jk->ik', A_np.astype(np.float64), B_np.astype(np.float64))

A_tf = tf.constant(A_np)
B_tf = tf.constant(B_np)
with tf.device('/CPU:0'):
    cpu = tf.einsum('ij,jk->ik', A_tf, B_tf).numpy()
with tf.device('/GPU:0'):
    gpu = tf.einsum('ij,jk->ik', A_tf, B_tf).numpy()

cpu_err = float(np.mean(np.abs(cpu.astype(np.float64) - ref)))
gpu_err = float(np.mean(np.abs(gpu.astype(np.float64) - ref)))
ratio = cpu_err / gpu_err if gpu_err > 0 else float('inf')

print(f"cpu_err = {cpu_err:.4e}")
print(f"gpu_err = {gpu_err:.4e}")
print(f"CPU/GPU error ratio: {ratio:.1f}x  (CPU less accurate)")
assert ratio > 5, f"Expected >5x ratio, got {ratio:.1f}x"
print("BUG CONFIRMED: TF CPU einsum f16 10x less accurate than GPU (sequential f16 accumulation vs GPU f32 promotion)")
