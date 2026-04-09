# ROOT CAUSE: tf.math.pow with negative base and non-integer exponent
# pow(x, 0.5) = sqrt(x) is NaN for x < 0.
# model_0182: pow(reduce_prod(x, axis=-1, keepdims=True), 0.5)
# reduce_prod of signed values can be negative; then pow(..., 0.5) = NaN.
# model_0094: sqrt(linalg.norm(...)) — norm is always >= 0, but the cumsum
# division pattern produces degenerate values.

import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

# pow(negative, 0.5) = NaN
x = tf.constant([-0.5, -1.0, 0.5, 1.0])
out = tf.math.pow(x, 0.5)
has_nan = bool(tf.reduce_any(tf.math.is_nan(out)))
print(f"pow([-0.5,-1.0,0.5,1.0], 0.5): {out.numpy()}, has_nan = {has_nan}")

# model_0182 pattern: pow(reduce_prod(signed_values), 0.5)
x2 = tf.constant([[-0.3, 0.8, -0.2, 0.5, -0.9, 0.1, -0.6, 0.4]])
# After ELU (makes some values negative for x < -1, but clips): still, product of all:
prod = tf.reduce_prod(x2, axis=-1, keepdims=True)
out2 = tf.math.pow(prod, 0.5)  # NaN when prod < 0
has_nan2 = bool(tf.reduce_any(tf.math.is_nan(out2)))
print(f"reduce_prod = {prod.numpy()}, pow(prod, 0.5) = {out2.numpy()}, has_nan = {has_nan2}")

if has_nan or has_nan2:
    print("PASS")
else:
    print("NOT REPRODUCED")
