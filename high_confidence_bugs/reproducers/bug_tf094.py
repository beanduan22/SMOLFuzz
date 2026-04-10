#!/usr/bin/env python3
"""
SMOLFuzz Bug Reproducer — TensorFlow
Model   : 94
Bug Type: baseline inconsistency (no mutation needed)
L2      : l2=8.7652e-03 > threshold=1e-03 finite_elements=1
APIs    : tf.keras.metrics.squared_hinge, tf.nn.max_pool2d, tf.square, tf.keras.layers.experimental.preprocessing.Discretization

Run: python3 bug_tf094.py
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
        self.dense1 = tf.keras.layers.Dense(16, input_dim=8)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.dense2 = tf.keras.layers.Dense(32)

    def call(self, x, training=False):
        x = self.dense1(x)
        x = tf.sin(tf.square(x))
        x = self.batch_norm(x)
        x = self.dropout(x, training=training)
        x = self.layer_norm(x)
        return tf.reduce_sum(self.dense2(x))

def make_inputs():
    return [np.array([[0.49671414494514465, -0.13826429843902588, 0.6476885676383972, 1.5230298042297363, -0.2341533750295639, -0.23413695394992828, 1.5792127847671509, 0.7674347162246704], [-0.4694743752479553, 0.5425600409507751, -0.4634176790714264, -0.4657297432422638, 0.241962268948555, -1.9132802486419678, -1.7249178886413574, -0.5622875094413757], [-1.0128310918807983, 0.31424733996391296, -0.9080240726470947, -1.4123036861419678, 1.4656487703323364, -0.2257762998342514, 0.06752820312976837, -1.424748182296753], [-0.5443827509880066, 0.11092258989810944, -1.1509935855865479, 0.3756980299949646, -0.6006386876106262, -0.2916937470436096, -0.6017066240310669, 1.852278232574463]], dtype=np.float32)]

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
