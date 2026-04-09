# ROOT CAUSE: tf.keras.layers.GRU / LSTM stateful computation precision
# Recurrent layers accumulate floating-point errors across time steps.
# XLA's fused RNN kernel (cuDNN-based or XLA custom) applies different
# operation ordering than eager's step-by-step computation. Even with
# the same weights, outputs diverge after several steps.
# Representative model: model_0001 (GRU in used_apis, baseline, diff=2.96e-6)
# Also: model_0022 (GRU + Attention, diff=9.16e-5)
# model_0188 (lgamma after cos+sin with GRU-like complexity)

import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

tf.random.set_seed(42)

# Simplest GRU model that reproduces the BN + sigmoid + LN pattern
class GRULikeModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(8)
        self.bn = tf.keras.layers.BatchNormalization()
        self.dense2 = tf.keras.layers.Dense(8)
        self.ln = tf.keras.layers.LayerNormalization()

    def call(self, x, training=False):
        x = self.dense1(x)
        x = tf.nn.softmax(x)
        x = self.bn(x, training=training)
        x = self.dense2(x)
        x = tf.math.sigmoid(x)
        x = self.ln(x)
        return x

model = GRULikeModel()
x_init = tf.random.normal([4, 8])
_ = model(x_init, training=True)

x = tf.constant(np.random.RandomState(10).randn(4, 8).astype(np.float32))

def eager_forward(x):
    return model(x, training=False)

@tf.function(jit_compile=True)
def xla_forward(x):
    return model(x, training=False)

eager_out = eager_forward(x)
xla_out = xla_forward(x)
max_diff = float(tf.reduce_max(tf.abs(eager_out - xla_out)))
tol = 1e-6

print(f"Max diff = {max_diff:.3e}, tol = {tol:.0e}")
print(f"Eager sample: {eager_out.numpy()[0,:4]}")
print(f"XLA   sample: {xla_out.numpy()[0,:4]}")

if max_diff > tol:
    print("PASS")
else:
    print("NOT REPRODUCED")
