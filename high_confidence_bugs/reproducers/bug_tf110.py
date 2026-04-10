#!/usr/bin/env python3
"""
SMOLFuzz Bug Reproducer — TensorFlow
Model   : 110
Bug Type: baseline inconsistency (no mutation needed)
L2      : l2=1.0559e-02 > threshold=1e-03 finite_elements=1
APIs    : tf.experimental.numpy.select, tf.experimental.numpy.flipud, tf.experimental.numpy.outer, tf.math.scalar_mul

Run: python3 bug_tf110.py
Requires: TensorFlow with GPU
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf

# ── Model ──────────────────────────────────────────────────────
class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.dense1 = tf.keras.layers.Dense(16, input_shape=(8,))
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.add = tf.keras.layers.Add()
        self.reshape = tf.keras.layers.Reshape((4, 4))
        self.dense2 = tf.keras.layers.Dense(1)

    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.batch_norm(x)
        x1 = tf.experimental.numpy.select([x > 0], [tf.math.scalar_mul(2.0, x)])
        x2 = tf.math.scalar_mul(-1.0, x)
        x = self.add([x1, x2])
        x = self.dropout(x, training=training)
        x = self.reshape(x)
        x = tf.experimental.numpy.outer(tf.reduce_mean(x, axis=1), tf.reduce_mean(x, axis=1))
        x = tf.reduce_sum(x)
        return x

def make_inputs():
    rng = np.random.RandomState(42)
    return [rng.randn(4, 8).astype(np.float32)]

# ── Reproducer ─────────────────────────────────────────────────
def run():
    tf.random.set_seed(42)
    inputs = make_inputs()
    x_np = inputs[0]

    with tf.device("/CPU:0"):
        model_cpu = Model()
        # Build model by running once
        _ = model_cpu(tf.constant(x_np), training=False)

    with tf.device("/GPU:0"):
        model_gpu = Model()
        _ = model_gpu(tf.constant(x_np), training=False)
        # Copy weights from CPU to GPU model
        for vc, vg in zip(model_cpu.variables, model_gpu.variables):
            vg.assign(tf.cast(vc, vg.dtype))

    with tf.device("/CPU:0"):
        cpu_out = model_cpu(tf.constant(x_np), training=False)
    with tf.device("/GPU:0"):
        gpu_out = model_gpu(tf.constant(x_np), training=False)

    cpu_np = np.array(cpu_out).flatten().astype(np.float32)
    gpu_np = np.array(gpu_out).flatten().astype(np.float32)
    
    print(f"CPU output[:4]: {cpu_np[:4]}")
    print(f"GPU output[:4]: {gpu_np[:4]}")

    fin = np.isfinite(cpu_np) & np.isfinite(gpu_np)
    asym_nan = int(np.sum(np.isnan(cpu_np) != np.isnan(gpu_np)))
    
    if not fin.any():
        print("All outputs non-finite (both devices)")
        return
    
    l2 = float(np.sqrt(np.sum((cpu_np[fin] - gpu_np[fin]) ** 2)))
    
    if asym_nan > 0:
        print(f"BUG: Asymmetric NaN — {asym_nan} positions differ")
    elif l2 > 1e-3:
        print(f"BUG CONFIRMED: L2={l2:.4e} between CPU and GPU (same model, same input, no mutation)")
    else:
        print(f"L2={l2:.4e} — inconsistency not reproduced (possibly hardware-dependent)")

if __name__ == "__main__":
    run()
