"""
Bug: TF CPU reduce_std returns 0.0 for float16 when N >= 65536.
Root cause: Same as reduce_mean: TF CPU stores count N as float16; float16(65536) = inf.
sqrt(variance) where variance=0 returns 0.0 instead of the correct ~1.0.
GPU computes correctly.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf

np.random.seed(0)
x_np = np.random.randn(65536).astype(np.float16)
ref = float(np.std(x_np.astype(np.float64)))

x_tf = tf.constant(x_np)
with tf.device('/CPU:0'):
    cpu = float(tf.math.reduce_std(x_tf).numpy())
with tf.device('/GPU:0'):
    gpu = float(tf.math.reduce_std(x_tf).numpy())

print(f"ref = {ref:.6f}")
print(f"cpu = {cpu:.6f}  (WRONG: should be ~1.0)")
print(f"gpu = {gpu:.6f}")
assert cpu == 0.0, f"Expected CPU to return 0.0, got {cpu}"
assert gpu > 0.9, f"GPU result {gpu} too far from ref {ref}"
print("BUG CONFIRMED: TF CPU reduce_std returns 0.0 for float16 N=65536 (float16(N)=inf)")
