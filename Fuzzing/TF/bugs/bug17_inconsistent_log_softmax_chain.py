# ROOT CAUSE: tf.nn.log_softmax followed by aggregation (reduce_min/matmul)
# log_softmax(x) = log(exp(x) / sum(exp(x))) = x - log(sum(exp(x)))
# XLA uses a numerically-stabilized fused kernel that may produce slightly
# different results than eager's sequential subtract-then-log-sum-exp.
# When combined with matmul and reduce_min, the small per-element differences
# accumulate.
# Representative model: model_0136 (log_softmax(matmul(x,xT)), diff=1.34e-5)
# Also: model_0187 (reduce_min(log_softmax(x)), diff=1.53e-5)
# model_0218 (log_softmax with sort+floormod, diff=3.81e-6)

import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

tf.random.set_seed(42)

class LogSoftmaxModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(64)
        self.bn = tf.keras.layers.BatchNormalization()
        self.ln = tf.keras.layers.LayerNormalization()

    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.bn(x, training=training)
        x = tf.nn.elu(x)
        x = self.ln(x)
        # matmul then log_softmax (model_0136 pattern)
        gram = tf.linalg.matmul(x, tf.transpose(x))
        return tf.nn.log_softmax(gram)

model = LogSoftmaxModel()
x_init = tf.random.normal([4, 8])
_ = model(x_init, training=True)

x = tf.constant(np.random.RandomState(6).randn(4, 8).astype(np.float32))

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
