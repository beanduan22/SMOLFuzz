"""
Bug: tf.math.cumsum float32 CPU is 7x less accurate than GPU for large values.

Root cause: TF's CPU cumsum kernel accumulates in native float32 sequentially,
leading to rounding error proportional to the input magnitude. TF's GPU kernel
uses parallel tree reduction (or a blocked algorithm) which distributes rounding
error more evenly, resulting in a 7x more accurate result for N=10,000 values
scaled to ~1e10.

Note: This is the OPPOSITE of PyTorch's cumsum float32 behavior (pt_cumsum_f32),
where CPU exactly matches float64 (by using float64 accumulation internally)
while GPU has error. TF CPU uses float32 throughout; TF GPU uses parallel reduction.
"""
import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf

np.random.seed(0)
x_np = np.random.randn(10_000).astype(np.float32) * 1e10
ref = np.cumsum(x_np.astype(np.float64)).astype(np.float32)

x_f32 = tf.constant(x_np)

with tf.device("/CPU:0"):
    cpu = tf.math.cumsum(x_f32).numpy()
with tf.device("/GPU:0"):
    gpu = tf.math.cumsum(x_f32).numpy()

cpu_err = np.linalg.norm(cpu - ref)
gpu_err = np.linalg.norm(gpu - ref)

print(f"CPU error vs float64 reference: {cpu_err:.4e}   <-- BUG")
print(f"GPU error vs float64 reference: {gpu_err:.4e}")
print(f"CPU is {cpu_err / gpu_err:.0f}x less accurate than GPU")
print()
print(f"Last 3 cumulative sums:")
for i in range(-3, 0):
    print(f"  [{i}] ref={ref[i]:.2f}  cpu={cpu[i]:.2f}  gpu={gpu[i]:.2f}")
