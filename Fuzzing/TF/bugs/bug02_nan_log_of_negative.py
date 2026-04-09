# ROOT CAUSE: tf.math.log / tf.math.log1p of non-positive values
# log(x) is undefined for x <= 0. Models feed log directly on layer output
# that may contain zeros or negatives. Both eager and XLA produce NaN.
# Representative model: model_0120 (tf.math.log(tf.reduce_prod(x, axis=1)))
# Also: model_0086 (log1p(x) where x can be < -1)

import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

# Case 1: log(reduce_prod) — product of small values can go zero/negative
x = tf.constant([[-0.5, 0.3, -0.2, 0.8, 0.1, -0.4, 0.9, -0.7]])
prod = tf.reduce_prod(x, axis=1, keepdims=True)
out = tf.math.log(prod)
has_nan_case1 = bool(tf.reduce_any(tf.math.is_nan(out)))
print(f"Case 1: reduce_prod = {prod.numpy()}, log(prod) = {out.numpy()}, has_nan = {has_nan_case1}")

# Case 2: log directly on values that include 0 or negatives
x2 = tf.constant([-1.0, 0.0, -0.5, 0.001])
out2 = tf.math.log(x2)
has_nan_case2 = bool(tf.reduce_any(tf.math.is_nan(out2) | tf.math.is_inf(out2)))
print(f"Case 2: log([-1,0,-0.5,0.001]) = {out2.numpy()}, has_nan_or_inf = {has_nan_case2}")

if has_nan_case1 or has_nan_case2:
    print("PASS")
else:
    print("NOT REPRODUCED")
