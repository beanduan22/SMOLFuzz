# ROOT CAUSE: tf.exp(tf.math.square(x)) — squaring before exp causes rapid overflow
# After BN, values are typically in [-3, 3] range; square -> [0, 9]; exp(9) ~ 8103.
# XLA may optimize the square+exp into a different numerical path.
# With scale_large mutation, squared values become huge and exp overflows to inf.
# model_0022: exp(square(x)) -> top_k -> reduce_sum, diff=9.16e-5
# model_0231: exp(x/2.0) after PReLU+BN — at scale_large, x can be very large.

import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

tf.random.set_seed(42)

class ExpSquaredModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(8)
        self.bn = tf.keras.layers.BatchNormalization()
        self.ln = tf.keras.layers.LayerNormalization()

    def call(self, x, training=False):
        x = tf.nn.selu(x)
        mean, variance = tf.nn.moments(x, axes=[1], keepdims=True)
        x = tf.stop_gradient(mean) + (x - mean)
        x = self.dense1(x)
        x = self.bn(x, training=training)
        x = self.ln(x)
        x = tf.exp(tf.math.square(x))
        top_k = tf.math.top_k(x, k=3).values
        return tf.reduce_sum(top_k)

model = ExpSquaredModel()
tf.keras.utils.set_random_seed(42)
x_init = tf.constant(np.random.RandomState(42).randn(4, 8).astype(np.float32))
_ = model(x_init, training=True)

@tf.function(jit_compile=True)
def xla_forward(x):
    return model(x, training=False)

def eager_forward(x):
    return model(x, training=False)

tol = 1e-6
max_diff = 0.0
eager_out = xla_out = None

for seed in range(30):
    candidate = tf.constant(np.random.RandomState(seed).randn(4, 8).astype(np.float32))
    e = eager_forward(candidate)
    xl = xla_forward(candidate)
    d = float(tf.abs(e - xl))
    if d > max_diff:
        max_diff = d
        eager_out = e
        xla_out = xl
    if d > tol:
        break

print(f"Eager: {eager_out.numpy():.6f}")
print(f"XLA:   {xla_out.numpy():.6f}")
print(f"Max diff = {max_diff:.3e}, tol = {tol:.0e}")

if max_diff > tol:
    print("PASS")
else:
    print("NOT REPRODUCED")
