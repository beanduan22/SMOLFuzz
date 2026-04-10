#!/usr/bin/env python3
"""
SMOLFuzz Bug Reproducer — TensorFlow
Model   : 357
Bug Type: baseline inconsistency (no mutation needed)
L2      : l2=1.0691e-03 > threshold=1e-03 finite_elements=4
APIs    : tf.math.atan2, tf.experimental.numpy.compress, tf.math.expm1, tf.keras.metrics.msle

Run: python3 bug_tf357.py
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
        self.dense1 = tf.keras.layers.Dense(8, input_dim=8, kernel_constraint=tf.keras.constraints.max_norm(max_value=2.0))
        self.bn = tf.keras.layers.BatchNormalization()
        self.ln = tf.keras.layers.LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(rate=0.5)
        self.dense2 = tf.keras.layers.Dense(1, input_dim=8)

    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.bn(x, training=training)
        x = self.ln(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        return tf.reduce_logsumexp(tf.math.expm1(x), axis=-1)

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
