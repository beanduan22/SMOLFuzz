"""
Bug: TF CPU reduce_mean returns 0.0 for float16 when N >= 65536.
Root cause: TF CPU implementation stores count N as float16; float16(65536) = inf → sum/inf = 0.
GPU computes correctly, returning the proper mean value.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf

np.random.seed(0)
x_np = np.random.randn(65536).astype(np.float16)
ref = float(np.mean(x_np.astype(np.float64)))

x_tf = tf.constant(x_np)
with tf.device('/CPU:0'):
    cpu = float(tf.reduce_mean(x_tf).numpy())
with tf.device('/GPU:0'):
    gpu = float(tf.reduce_mean(x_tf).numpy())

print(f"ref = {ref:.6f}")
print(f"cpu = {cpu:.6f}  (WRONG: should not be 0.0)")
print(f"gpu = {gpu:.6f}")
assert cpu == 0.0, f"Expected CPU to return 0.0, got {cpu}"
assert abs(gpu - ref) < 0.01, f"GPU result {gpu} far from ref {ref}"
print("BUG CONFIRMED: TF CPU reduce_mean returns 0.0 for float16 N=65536 (float16(N)=inf)")
