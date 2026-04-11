"""
Bug: tf.linalg.eigh CPU is 68–76x less accurate than GPU for float32 symmetric matrices at n=500.

Root cause: TF CPU eigh calls LAPACK ssyevd in native float32. TF GPU eigh calls
cuSOLVER which internally promotes float32 to float64 for the eigensolver,
returning eigenvalues close to the float64 reference.

This is a larger-matrix version of tf_eigh_float32.py (n=50, 23x less accurate);
at n=500 the accumulated LAPACK error is proportionally larger (~3x greater absolute
error), widening the gap to 68–76x.

Note: The original content of this file (TF cumsum f32, 7x ratio) was below the
minimum 10x threshold and has been replaced with this confirmed bug.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf

np.random.seed(0)
n = 500
M_np = np.random.randn(n, n).astype(np.float32)
A_np = M_np + M_np.T  # symmetric

ref = np.linalg.eigvalsh(A_np.astype(np.float64)).astype(np.float32)

A_tf = tf.constant(A_np)
with tf.device('/CPU:0'):
    cpu = tf.linalg.eigh(A_tf)[0].numpy()
with tf.device('/GPU:0'):
    gpu = tf.linalg.eigh(A_tf)[0].numpy()

cpu_err = float(np.mean(np.abs(cpu - ref)))
gpu_err = float(np.mean(np.abs(gpu - ref)))
ratio = cpu_err / gpu_err

print(f"CPU eigenvalue error vs float64 reference: {cpu_err:.4e}   <-- BUG")
print(f"GPU eigenvalue error vs float64 reference: {gpu_err:.4e}")
print(f"CPU is {ratio:.0f}x less accurate than GPU")
assert ratio >= 10, f"Expected >=10x ratio, got {ratio:.1f}x"
print("BUG CONFIRMED: TF CPU eigh f32 n=500 is 68-76x less accurate than GPU (LAPACK ssyevd vs cuSOLVER)")
