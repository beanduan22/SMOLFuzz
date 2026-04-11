"""
Bug: tf.math.reduce_std for float16 returns nan on CPU but inf on GPU.

Root cause: For float16 input values ~1e4:
- x^2 overflows float16 to inf (float16 max ≈ 65504; 1e4^2 = 1e8 >> max)
- The partial sum of x values themselves also overflows float16 to inf
- CPU variance kernel computes E[X²] - E[X]², giving inf - inf = nan, then sqrt(nan) = nan
- GPU variance kernel uses a different algorithm (possibly Welford's online
  formula or a different summation order), computing variance as inf, then sqrt(inf) = inf

Result: CPU and GPU give opposite IEEE 754 special values (nan vs inf) for the
same input, with no warning. This violates the expectation that the same operation
on the same data gives the same result regardless of device.
"""
import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf

np.random.seed(0)
x_np = np.random.randn(1000).astype(np.float32) * 1e4
x_f16 = tf.constant(x_np.astype(np.float16))

with tf.device("/CPU:0"):
    cpu = float(tf.cast(tf.math.reduce_std(x_f16), tf.float32).numpy())
with tf.device("/GPU:0"):
    gpu = float(tf.cast(tf.math.reduce_std(x_f16), tf.float32).numpy())

ref = float(np.std(x_np.astype(np.float64)))

print(f"Reference (float64): {ref:.4e}")
print(f"CPU (float16): {cpu}   <-- BUG (nan: inf - inf = nan in E[X²] - E[X]²)")
print(f"GPU (float16): {gpu}   <-- BUG (inf: different algorithm, avoids inf-inf)")
print()
print(f"CPU returns nan: {np.isnan(cpu)}")
print(f"GPU returns inf: {np.isinf(gpu)}")
print(f"CPU != GPU: {cpu != gpu and not (np.isnan(cpu) and np.isnan(gpu))}")
print()
# Variance shows the same divergence
with tf.device("/CPU:0"):
    cpu_var = float(tf.cast(tf.math.reduce_variance(x_f16), tf.float32).numpy())
with tf.device("/GPU:0"):
    gpu_var = float(tf.cast(tf.math.reduce_variance(x_f16), tf.float32).numpy())
print(f"reduce_variance CPU: {cpu_var}  GPU: {gpu_var}")
