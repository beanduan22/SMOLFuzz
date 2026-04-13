"""Bug 55: tf.math.unsorted_segment_sum float16 — GPU diverges from CPU.
GPU uses non-deterministic atomicAdd; float16 non-associativity causes divergence.
"""
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

N, M = 100000, 50
rng = np.random.default_rng(42)
src_np = (rng.standard_normal(N) * 100).astype(np.float16)
idx_np = rng.integers(0, M, N, dtype=np.int32)

src = tf.constant(src_np, dtype=tf.float16)
idx = tf.constant(idx_np, dtype=tf.int32)
ref = tf.math.unsorted_segment_sum(tf.constant(src_np.astype(np.float64), tf.float64), idx, M)

cpu = tf.math.unsorted_segment_sum(src, idx, M)
with tf.device('/GPU:0'):
    gpu = tf.math.unsorted_segment_sum(src, idx, M)

cpu_err = tf.reduce_max(tf.abs(tf.cast(cpu, tf.float64) - ref)).numpy()
gpu_err = tf.reduce_max(tf.abs(tf.cast(gpu, tf.float64) - ref)).numpy()
diff    = tf.reduce_max(tf.abs(tf.cast(cpu - gpu, tf.float32))).numpy()

print(f"CPU error vs float64: {cpu_err:.3e}")
print(f"GPU error vs float64: {gpu_err:.3e}")
print(f"CPU vs GPU max diff:  {diff:.3e}")    # BUG: significant
assert diff > 1.0, "bug not reproduced"
print("BUG CONFIRMED: unsorted_segment_sum float16 CPU vs GPU divergence")
