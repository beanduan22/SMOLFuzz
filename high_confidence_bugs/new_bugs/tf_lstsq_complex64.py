"""Bug 59: tf.linalg.lstsq with fast=True on rank-deficient complex matrix returns NaN.
Cholesky path (fast=True) fails when A^H A is singular. QR path (fast=False) gives finite result.
"""
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

A = tf.constant([[1, 2, 3], [2, 4, 6], [0, 0, 0]], dtype=tf.complex64)  # rank-1
b = tf.constant([[1], [1], [1]], dtype=tf.complex64)

cpu_fast = tf.linalg.lstsq(A, b, fast=True)
with tf.device('/GPU:0'):
    gpu_fast = tf.linalg.lstsq(A, b, fast=True)
cpu_qr = tf.linalg.lstsq(A, b, fast=False)

print("fast=True  CPU:", cpu_fast.numpy().ravel())   # BUG: NaN
print("fast=True  GPU:", gpu_fast.numpy().ravel())   # BUG: NaN
print("fast=False CPU:", cpu_qr.numpy().ravel())     # finite — QR path works
assert np.any(np.isnan(cpu_fast.numpy())) and np.any(np.isnan(gpu_fast.numpy())), "bug not reproduced"
print("BUG CONFIRMED: lstsq fast=True returns NaN for rank-deficient complex matrix")
