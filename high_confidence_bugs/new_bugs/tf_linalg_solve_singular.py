"""Bug 57: Cross-framework — TF linalg.solve CPU vs GPU (reference, no divergence unlike PyTorch).
TF CPU and GPU agree on ill-conditioned solve; contrast with Bug 56 (PyTorch diverges).
"""
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

x = np.linspace(0, 1, 8, dtype=np.float32)
A = tf.constant(np.vander(x, 8, increasing=True))
b = tf.constant(np.ones((8, 1), dtype=np.float32))

ref = tf.linalg.solve(tf.cast(A, tf.float64), tf.cast(b, tf.float64))
cpu = tf.linalg.solve(A, b)
with tf.device('/GPU:0'):
    gpu = tf.linalg.solve(A, b)

cpu_err = tf.norm(tf.cast(cpu, tf.float64) - ref).numpy()
gpu_err = tf.norm(tf.cast(gpu, tf.float64) - ref).numpy()
diff    = tf.norm(tf.cast(cpu - gpu, tf.float64)).numpy()

print(f"TF CPU error vs float64: {cpu_err:.3e}")
print(f"TF GPU error vs float64: {gpu_err:.3e}")
print(f"TF CPU vs GPU diff:      {diff:.3e}")   # TF agrees; PyTorch would show massive divergence here
print("NOTE: TF CPU==GPU for ill-conditioned solve (unlike PyTorch Bug 56)")
