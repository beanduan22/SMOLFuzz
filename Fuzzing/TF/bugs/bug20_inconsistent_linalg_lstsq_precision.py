# ROOT CAUSE: tf.linalg.lstsq numerical precision difference eager vs XLA
# lstsq uses a QR decomposition (eager: LAPACK, XLA: custom implementation).
# With ill-conditioned or near-singular matrices, the two implementations
# produce different solutions due to different pivoting strategies.
# Representative model: model_0109 (lstsq(x@xT, ones), diff=2.13e-4 at add_noise)
# Also: model_0274 (lstsq in used_apis, scale_large, diff=2.44e-4)

import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

tf.random.set_seed(42)

class LstsqModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(8, activation=tf.nn.swish)
        self.bn = tf.keras.layers.BatchNormalization()
        self.dense2 = tf.keras.layers.Dense(8, activation=tf.nn.tanh)
        self.ln = tf.keras.layers.LayerNormalization()

    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.bn(x)
        x = self.dense2(x)
        x = self.ln(x)
        gram = tf.matmul(x, x, transpose_b=True)  # 4x4 Gram matrix
        sol = tf.linalg.lstsq(gram, tf.ones_like(gram))
        return tf.reduce_mean(tf.math.pow(sol, 2.0))

model = LstsqModel()
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

for seed in range(50):
    x_base = tf.constant(np.random.RandomState(seed).randn(4, 8).astype(np.float32))
    # add_noise: small perturbation on top
    x_noisy = x_base + tf.constant(np.random.RandomState(seed + 100).randn(4, 8).astype(np.float32) * 0.01)
    try:
        e = eager_forward(x_noisy)
        xl = xla_forward(x_noisy)
        d = float(tf.abs(e - xl))
        if d > max_diff and not (tf.math.is_nan(e) or tf.math.is_nan(xl)):
            max_diff = d
            eager_out = e
            xla_out = xl
        if d > tol:
            break
    except Exception:
        continue

if eager_out is None:
    eager_out = xla_out = tf.constant(0.0)

print(f"Eager: {eager_out.numpy():.8f}")
print(f"XLA:   {xla_out.numpy():.8f}")
print(f"Max diff = {max_diff:.3e}, tol = {tol:.0e}")

if max_diff > tol:
    print("PASS")
else:
    print("NOT REPRODUCED")
