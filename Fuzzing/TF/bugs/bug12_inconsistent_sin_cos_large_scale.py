# ROOT CAUSE: tf.sin / tf.cos on large-scale inputs — eager vs XLA differ
# After scale_large mutation (inputs multiplied by large constant), sin/cos
# arguments can be large. sin/cos of large arguments has reduced relative
# precision; XLA's fused kernel may produce slightly different results
# than eager's sequential computation. max_diff can reach 1e-3 or more.
# Representative model: model_0083 (sin -> BN -> exp, scale_large, diff=8.26e-6)
# Also: model_0007 (reduce_variance after BN+swish+ThresholdedReLU)
# model_0069 (reduce_sum(sin(x) + cos(x)) after BN, diff=3.81e-6)
# model_0247 (reduce_prod(cos(x)), diff=5.80e-6)
# model_0259 (reduce_sum(cos(x)) after sin+BN, diff=3.81e-6)

import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

tf.random.set_seed(42)

class SinCosBNModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(8)
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, x, training=False):
        x = self.dense1(x)
        x = tf.sin(x)
        x = self.bn(x, training=training)
        return tf.reduce_sum(tf.cos(x))

model = SinCosBNModel()
tf.keras.utils.set_random_seed(42)
x_init = tf.constant(np.random.RandomState(42).randn(4, 8).astype(np.float32))
_ = model(x_init, training=True)

@tf.function(jit_compile=True)
def xla_forward(x):
    return model(x, training=False)

def eager_forward(x):
    return model(x, training=False)

# Search for an input that triggers the divergence
tol = 1e-6
x_large = None
max_diff = 0.0
eager_out = xla_out = None

for seed in range(50):
    x_base = tf.constant(np.random.RandomState(seed).randn(4, 8).astype(np.float32))
    for scale in [100.0, 50.0, 200.0, 1000.0]:
        candidate = x_base * scale
        e = eager_forward(candidate)
        xl = xla_forward(candidate)
        d = float(tf.abs(e - xl))
        if d > max_diff:
            max_diff = d
            x_large = candidate
            eager_out = e
            xla_out = xl
        if d > tol:
            break
    if max_diff > tol:
        break

print(f"Eager: {eager_out.numpy():.8f}")
print(f"XLA:   {xla_out.numpy():.8f}")
print(f"Max diff = {max_diff:.3e}, tol = {tol:.0e}")

if max_diff > tol:
    print("PASS")
else:
    print("NOT REPRODUCED")
