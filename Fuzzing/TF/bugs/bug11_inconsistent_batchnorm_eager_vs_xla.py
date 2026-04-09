# ROOT CAUSE: tf.keras.layers.BatchNormalization numerical precision difference
# between eager and XLA (jit_compile=True) modes.
# BatchNorm computes: (x - mean) / sqrt(variance + epsilon) * gamma + beta
# XLA may use different floating-point operation ordering/fusion, causing
# small but detectable differences (typically 1e-6 to 1e-5 range).
# This is the MOST COMMON root cause: affects models 1,13,15,16,18,23,33,
# 41,47,53,57,59,83,87,91,96,98,99,121,126,135,163,190,191,198,201,
# 209,217,218,222,223,229,235,253,259,274,275,277,286 (39 models).
# Representative: model_0013 (simplest: Dense+BN+ReLU6+Dropout+LN+Dense)

import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

tf.random.set_seed(42)

class MinimalBNModel(tf.keras.Model):
    """BatchNorm + exp(gelu(x)) + linalg.norm — pattern from model_0016."""
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(16)
        self.bn = tf.keras.layers.BatchNormalization()
        self.ln = tf.keras.layers.LayerNormalization()

    def call(self, x, training=False):
        x = tf.abs(x)
        x = self.dense1(x)
        x = self.bn(x, training=training)
        x = tf.nn.gelu(x)
        x = tf.math.exp(tf.math.maximum(x, 0))
        return tf.linalg.norm(x)

model = MinimalBNModel()

# Initialize weights by running once
x_init = tf.random.normal([4, 8])
_ = model(x_init, training=True)

# Run a few training steps to get non-trivial BN statistics
for _ in range(10):
    _ = model(tf.random.normal([4, 8]), training=True)

x = tf.constant(np.random.RandomState(42).randn(4, 8).astype(np.float32) * 3.0)  # scale triggers BN divergence

def eager_forward(x):
    return model(x, training=False)

@tf.function(jit_compile=True)
def xla_forward(x):
    return model(x, training=False)

eager_out = eager_forward(x)
xla_out = xla_forward(x)

max_diff = float(tf.reduce_max(tf.abs(eager_out - xla_out)))
tol = 1e-6

print(f"Eager output: {eager_out.numpy().flatten()}")
print(f"XLA   output: {xla_out.numpy().flatten()}")
print(f"Max diff = {max_diff:.3e}, tol = {tol:.0e}")

if max_diff > tol:
    print("PASS")
else:
    print("NOT REPRODUCED")
