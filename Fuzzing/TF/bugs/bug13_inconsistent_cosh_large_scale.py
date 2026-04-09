# ROOT CAUSE: tf.math.cosh on large-scale inputs causes large numerical
# discrepancy between eager and XLA. cosh(x) = (exp(x)+exp(-x))/2 grows
# rapidly; at large scale inputs, floating-point ordering in XLA fusion
# produces different intermediate values. max_diff can exceed 1e-3.
# Representative model: model_0035 (reduce_prod(cosh(x)) after BN+tanh, diff=2.81e-1)
# Also: model_0195 (pow(x,2) + einsum, scale_large, diff=1.46e-3)
# model_0255 (cosh+LayerNorm+Conv1D, diff=4.53e-6)
# model_0270 (cosh(x) as first op -> BN -> dense, NaN with scale_large)

import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

tf.random.set_seed(42)

class CoshBNModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(8, activation='relu')
        self.bn = tf.keras.layers.BatchNormalization()
        self.dense2 = tf.keras.layers.Dense(8, activation='tanh')
        self.ln = tf.keras.layers.LayerNormalization()

    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.bn(x, training=training)
        x = self.dense2(x)
        x = self.ln(x)
        return tf.reduce_prod(tf.math.cosh(x))

model = CoshBNModel()
x_init = tf.random.normal([4, 8])
_ = model(x_init, training=True)

x_base = tf.constant(np.random.RandomState(2).randn(4, 8).astype(np.float32))

def eager_forward(x):
    return model(x, training=False)

@tf.function(jit_compile=True)
def xla_forward(x):
    return model(x, training=False)

# Test at baseline scale
eager_base = eager_forward(x_base)
xla_base = xla_forward(x_base)
diff_base = float(tf.abs(eager_base - xla_base))
print(f"Baseline scale: eager={eager_base.numpy():.6f}, xla={xla_base.numpy():.6f}, diff={diff_base:.3e}")

# Test at large scale (scale_large mutation)
x_large = x_base * 10.0
eager_large = eager_forward(x_large)
xla_large = xla_forward(x_large)
diff_large = float(tf.abs(eager_large - xla_large))
print(f"Large scale: eager={eager_large.numpy():.6f}, xla={xla_large.numpy():.6f}, diff={diff_large:.3e}")

tol = 1e-6
if diff_base > tol or diff_large > tol:
    print("PASS")
else:
    print("NOT REPRODUCED")
