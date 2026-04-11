"""
Bug: TF CPU nuclear norm (sum of singular values) is 83x less accurate than GPU for float32 500x500.
Root cause: TF CPU SVD uses LAPACK sgesdd in single precision; GPU cuSOLVER promotes internally to float64.
CPU is 83x less accurate than GPU.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf

np.random.seed(0)
M_np = np.random.randn(500, 500).astype(np.float32)
ref = float(np.sum(np.linalg.svd(M_np.astype(np.float64), compute_uv=False)))

M_tf = tf.constant(M_np)
with tf.device('/CPU:0'):
    cpu_svd = tf.linalg.svd(M_tf, compute_uv=False).numpy()
    cpu_nuc = float(np.sum(cpu_svd))
with tf.device('/GPU:0'):
    gpu_svd = tf.linalg.svd(M_tf, compute_uv=False).numpy()
    gpu_nuc = float(np.sum(gpu_svd))

cpu_err = abs(cpu_nuc - ref)
gpu_err = abs(gpu_nuc - ref)
ratio = cpu_err / gpu_err if gpu_err > 0 else float('inf')

print(f"ref      = {ref:.6f}")
print(f"cpu_nuc  = {cpu_nuc:.6f}  err={cpu_err:.4e}")
print(f"gpu_nuc  = {gpu_nuc:.6f}  err={gpu_err:.4e}")
print(f"CPU/GPU error ratio: {ratio:.0f}x  (CPU less accurate)")
assert ratio > 10, f"Expected >10x ratio, got {ratio:.1f}x"
print("BUG CONFIRMED: TF CPU nuclear norm 83x less accurate than GPU (float32 500x500)")
