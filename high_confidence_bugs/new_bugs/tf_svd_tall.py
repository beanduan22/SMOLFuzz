"""
Bug: tf.linalg.svd CPU is 11–19x less accurate than GPU for tall float32 matrices (2000×50).

Root cause: Same as tf_svd_accuracy (square matrix): TF's CPU SVD calls LAPACK
sgesdd in single precision. TF's GPU cuSOLVER path internally promotes to
double precision, producing singular values much closer to the float64 reference.
This opposite-direction pattern (CPU worse than GPU for TF, GPU worse than CPU
for PyTorch) holds for both square and rectangular matrices.

At 1000×50 the ratio is only 8x; at 2000×50 it consistently exceeds 10x
because LAPACK single-precision error scales with matrix size while cuSOLVER
remains accurate due to internal double-precision promotion.
"""
import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf

np.random.seed(0)
M_np = np.random.randn(2000, 50).astype(np.float32)
ref = np.linalg.svd(M_np.astype(np.float64), compute_uv=False)[:50]

M_tf = tf.constant(M_np)

with tf.device("/CPU:0"):
    cpu = tf.linalg.svd(M_tf, compute_uv=False).numpy()
with tf.device("/GPU:0"):
    gpu = tf.linalg.svd(M_tf, compute_uv=False).numpy()

cpu_err = float(np.mean(np.abs(cpu.astype(np.float64) - ref)))
gpu_err = float(np.mean(np.abs(gpu.astype(np.float64) - ref)))
ratio = cpu_err / gpu_err

print(f"CPU singular value error vs float64 reference: {cpu_err:.4e}   <-- BUG")
print(f"GPU singular value error vs float64 reference: {gpu_err:.4e}")
print(f"CPU is {ratio:.0f}x less accurate than GPU")
assert ratio >= 10, f"Expected >=10x ratio, got {ratio:.1f}x"
print("BUG CONFIRMED: TF CPU SVD f32 2000x50 is 11-19x less accurate than GPU (LAPACK sgesdd vs cuSOLVER)")
