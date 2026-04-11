"""
Bug: tf.linalg.svd CPU is 17x less accurate than GPU for float32 matrices.

Root cause: TF's CPU path calls LAPACK sgesdd in single precision. TF's GPU path
uses cuSOLVER which internally upconverts to double precision, producing singular
values much closer to the float64 reference. This is a missing precision-promotion
in TF's CPU SVD kernel, symmetric with the eigh CPU accuracy bug.
"""
import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf

np.random.seed(0)
n = 200
M_np = np.random.randn(n, n).astype(np.float32)
M = tf.constant(M_np)

ref = np.linalg.svd(M_np.astype(np.float64), compute_uv=False).astype(np.float32)

with tf.device("/CPU:0"):
    cpu = tf.linalg.svd(M, compute_uv=False).numpy()
with tf.device("/GPU:0"):
    gpu = tf.linalg.svd(M, compute_uv=False).numpy()

cpu_err = np.linalg.norm(cpu - ref)
gpu_err = np.linalg.norm(gpu - ref)

print(f"CPU singular value error vs float64 reference: {cpu_err:.4e}")
print(f"GPU singular value error vs float64 reference: {gpu_err:.4e}")
print(f"CPU is {cpu_err / gpu_err:.0f}x less accurate than GPU")
print(f"CPU vs GPU L2: {np.linalg.norm(cpu - gpu):.4e}")
print()
print("First 5 singular values:")
print(f"  CPU: {cpu[:5].tolist()}")
print(f"  GPU: {gpu[:5].tolist()}")
print(f"  Ref: {ref[:5].tolist()}")
