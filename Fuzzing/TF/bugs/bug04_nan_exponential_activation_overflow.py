# ROOT CAUSE: tf.keras.activations.exponential (exp) overflow producing Inf
# When scale_large mutation scales inputs up, exp(large_value) -> Inf.
# The NaN flag in this class is eager_inf=True, xla_inf=True.
# model_0105: cosh(linalg.norm(x)) with large x produces Inf.
# model_0117: exponential activation Dense layer with large input -> Inf.
# model_0209: expm1(large) -> Inf.

import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

# Case 1: exponential activation on large input produces inf
x_large = tf.constant([[100.0, 200.0, 150.0, 50.0]])
out_exp = tf.keras.activations.exponential(x_large)
has_inf = bool(tf.reduce_any(tf.math.is_inf(out_exp)))
print(f"exp(large input): {out_exp.numpy()}, has_inf = {has_inf}")

# Case 2: cosh(norm(large_x)) -> inf (model_0105 pattern)
x_large2 = tf.constant([[1e3, 2e3, 3e3]])
norm_val = tf.linalg.norm(x_large2)
out_cosh = tf.math.cosh(norm_val)
has_inf2 = bool(tf.math.is_inf(out_cosh))
print(f"cosh(norm(large)): {out_cosh.numpy()}, has_inf = {has_inf2}")

# Case 3: expm1(large) -> inf (model_0209 pattern)
x_large3 = tf.constant([100.0, 200.0])
out_expm1 = tf.math.expm1(x_large3)
has_inf3 = bool(tf.reduce_any(tf.math.is_inf(out_expm1)))
print(f"expm1(large): {out_expm1.numpy()}, has_inf = {has_inf3}")

if has_inf or has_inf2 or has_inf3:
    print("PASS")
else:
    print("NOT REPRODUCED")
