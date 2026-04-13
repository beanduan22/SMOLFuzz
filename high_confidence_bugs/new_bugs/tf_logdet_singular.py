"""Bug 58: tf.linalg.logdet on singular matrix returns NaN instead of -inf.
NumPy and PyTorch correctly return -inf. TF CPU and GPU both return NaN.
"""
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

A = tf.ones((8, 8))  # rank-1, det=0, logdet should be -inf

cpu = tf.linalg.logdet(A)
with tf.device('/GPU:0'):
    gpu = tf.linalg.logdet(A)

print("CPU:      ", cpu.numpy())   # BUG: nan
print("GPU:      ", gpu.numpy())   # BUG: nan
print("Expected: -inf  (NumPy:", np.linalg.slogdet(np.ones((8, 8)))[1], ")")
assert np.isnan(cpu.numpy()) and np.isnan(gpu.numpy()), "bug not reproduced"
print("BUG CONFIRMED: logdet(singular) returns NaN instead of -inf")
