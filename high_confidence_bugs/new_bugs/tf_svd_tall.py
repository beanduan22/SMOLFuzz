"""
Bug: tf.linalg.svd CPU is 8x less accurate than GPU for tall float32 matrices (1000×50).

Root cause: Same as tf_svd_accuracy (square matrix): TF's CPU SVD calls LAPACK
sgesdd in single precision. TF's GPU cuSOLVER path internally upconverts to
double precision, producing singular values much closer to the float64 reference.
This opposite-direction pattern (CPU worse than GPU for TF, GPU worse than CPU
for PyTorch) holds for both square and rectangular matrices.

Note: The corresponding PyTorch bug (pt_svdvals_tall) goes in the opposite
direction — GPU is 25x less accurate for the same 1000×50 matrix.
"""
import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf

np.random.seed(0)
M_np = np.random.randn(1000, 50).astype(np.float32)
ref = np.linalg.svd(M_np.astype(np.float64), compute_uv=False).astype(np.float32)

M_tf = tf.constant(M_np)

with tf.device("/CPU:0"):
    cpu = tf.linalg.svd(M_tf, compute_uv=False).numpy()
with tf.device("/GPU:0"):
    gpu = tf.linalg.svd(M_tf, compute_uv=False).numpy()

cpu_err = np.linalg.norm(cpu - ref)
gpu_err = np.linalg.norm(gpu - ref)

print(f"CPU singular value error vs float64 reference: {cpu_err:.4e}   <-- BUG")
print(f"GPU singular value error vs float64 reference: {gpu_err:.4e}")
print(f"CPU is {cpu_err / gpu_err:.0f}x less accurate than GPU")
print()
print(f"First 5 singular values:")
print(f"  CPU: {cpu[:5].tolist()}")
print(f"  GPU: {gpu[:5].tolist()}")
print(f"  Ref: {ref[:5].tolist()}")
