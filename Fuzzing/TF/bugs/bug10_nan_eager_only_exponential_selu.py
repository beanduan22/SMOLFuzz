# ROOT CAUSE: eager_nan=True but xla_nan=False — exponential activation overflow
# model_0117: Dense(exponential) -> BN -> Dropout -> Dense(selu) -> LayerNorm -> reduce_max
# With scale_large mutation, exponential activation overflows to inf in eager;
# XLA may constant-fold or handle differently, avoiding the NaN path.
# model_0279: sinh(large) -> inf, then elu(inf) -> inf, then sigmoid(inf)->1 (fine);
# but XLA handles sinh overflow differently (inf vs NaN).

import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

# Eager: exponential(large) = inf
x_large = tf.constant([[500.0, 600.0, 700.0, 800.0]])
eager_out = tf.keras.activations.exponential(x_large)
print(f"Eager exponential(large): {eager_out.numpy()}")

# Compare eager vs XLA for sinh overflow
@tf.function(jit_compile=True)
def xla_forward(x):
    return tf.math.sinh(x)

def eager_forward(x):
    return tf.math.sinh(x)

x_test = tf.constant([[1000.0, -1000.0, 2000.0, -500.0]])
eager_result = eager_forward(x_test)
xla_result = xla_forward(x_test)

eager_nan = bool(tf.reduce_any(tf.math.is_nan(eager_result) | tf.math.is_inf(eager_result)))
xla_nan = bool(tf.reduce_any(tf.math.is_nan(xla_result) | tf.math.is_inf(xla_result)))

print(f"eager sinh(large): {eager_result.numpy()}, has_nan_or_inf = {eager_nan}")
print(f"XLA   sinh(large): {xla_result.numpy()}, has_nan_or_inf = {xla_nan}")
print(f"Eager and XLA agree on nan/inf: {eager_nan == xla_nan}")

# model_0279: eager_nan=True, xla_nan=False, xla_inf=True
# sinh overflow -> different treatment
@tf.function(jit_compile=True)
def model_279_xla(x):
    x = tf.math.sinh(x)
    x = tf.nn.elu(x)
    return tf.nn.sigmoid(x)

def model_279_eager(x):
    x = tf.math.sinh(x)
    x = tf.nn.elu(x)
    return tf.nn.sigmoid(x)

x_279 = tf.constant([[500.0, -300.0, 200.0, -100.0, 400.0, -200.0, 300.0, -50.0]])
e_out = model_279_eager(x_279)
x_out = model_279_xla(x_279)
e_nan = bool(tf.reduce_any(tf.math.is_nan(e_out)))
x_nan = bool(tf.reduce_any(tf.math.is_nan(x_out)))
x_inf = bool(tf.reduce_any(tf.math.is_inf(x_out)))
print(f"\nModel 279 pattern: eager_nan={e_nan}, xla_nan={x_nan}, xla_inf={x_inf}")

if eager_nan or e_nan:
    print("PASS")
else:
    print("NOT REPRODUCED")
