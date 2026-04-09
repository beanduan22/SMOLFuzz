# ROOT CAUSE: tf.linalg.eigh numerical precision difference eager vs XLA
# eigh computes eigenvalues of a symmetric matrix using the LAPACK dsyevd
# (eager, via cuBLAS/LAPACK) vs a custom XLA implementation. The ordering
# of floating-point operations differs, causing small disagreements in
# eigenvalues of near-degenerate matrices.
# Representative model: model_0129 (eigh(x@xT) -> reduce_prod, diff=1.78e-4)
# Also: model_0258 (eigh in used_apis, scale_small, diff=1.83e-4)
# model_0163 (eigh(x) combined with sin, diff=1.01e-5)

import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

tf.random.set_seed(42)

class EighModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(8)
        self.bn = tf.keras.layers.BatchNormalization()
        self.ln = tf.keras.layers.LayerNormalization()
        self.dense2 = tf.keras.layers.Dense(4)

    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.bn(x, training=training)
        x = tf.nn.softplus(x)
        x = self.ln(x)
        x = self.dense2(x)
        # Form symmetric matrix and get eigenvalues
        gram = tf.matmul(x, x, transpose_a=True)  # 4x4 symmetric
        eigenvalues = tf.linalg.eigh(gram)[0]
        return tf.reduce_prod(eigenvalues, axis=-1)

model = EighModel()
x_init = tf.random.normal([4, 8])
_ = model(x_init, training=True)

x = tf.constant(np.random.RandomState(5).randn(4, 8).astype(np.float32))

def eager_forward(x):
    return model(x, training=False)

@tf.function(jit_compile=True)
def xla_forward(x):
    return model(x, training=False)

eager_out = eager_forward(x)
xla_out = xla_forward(x)
max_diff = float(tf.reduce_max(tf.abs(eager_out - xla_out)))
tol = 1e-6

print(f"Eager eigenvalue product: {eager_out.numpy()}")
print(f"XLA   eigenvalue product: {xla_out.numpy()}")
print(f"Max diff = {max_diff:.3e}, tol = {tol:.0e}")

if max_diff > tol:
    print("PASS")
else:
    print("NOT REPRODUCED")
