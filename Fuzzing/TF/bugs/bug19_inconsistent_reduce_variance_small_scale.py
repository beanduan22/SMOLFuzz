# ROOT CAUSE: tf.math.reduce_variance diverges between eager and XLA (scale_large mutation)
# reduce_variance(x) = E[(x - E[x])^2]. With scale_large (inputs x100), floating-point
# summation order differences between eager and XLA's parallel reduction produce
# slightly different variance estimates beyond 1e-6 tolerance.
# Representative model: model_0007 (Flatten+Dense(swish)+BN+Dropout+ThresholdedReLU+reduce_variance)
# Triggered by scale_large mutation: max_diff=1.10e-03

import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

tf.random.set_seed(42)

class ReduceVarianceModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64, activation=tf.nn.swish)
        self.bn = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(rate=0.2)
        self.thresholded_relu = tf.keras.layers.ThresholdedReLU(theta=0.5)

    def call(self, x, training=False):
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.bn(x, training=training)
        x = self.dropout(x, training=training)
        x = self.thresholded_relu(x)
        return tf.math.reduce_variance(x)

model = ReduceVarianceModel()
x_init = tf.random.normal([4, 8])
_ = model(x_init, training=True)

x_base = tf.constant(np.random.RandomState(8).randn(4, 8).astype(np.float32))

def eager_forward(x):
    return model(x, training=False)

@tf.function(jit_compile=True)
def xla_forward(x):
    return model(x, training=False)

# scale_large: larger inputs trigger floating-point summation order divergence
x_large = x_base * 100.0
eager_large = eager_forward(x_large)
xla_large = xla_forward(x_large)
diff_large = float(tf.abs(eager_large - xla_large))

print(f"Large scale: eager={eager_large.numpy():.6f}, xla={xla_large.numpy():.6f}, diff={diff_large:.3e}")

tol = 1e-6
if diff_large > tol:
    print("PASS")
else:
    print("NOT REPRODUCED")
