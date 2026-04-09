# ROOT CAUSE: tf.math.lgamma produces NaN/inf at non-positive integers
# lgamma(x) has poles at x = 0, -1, -2, ...
# Models pass reduce_sum output (which may be near-zero or negative) into lgamma.
# Representative: model_0175 (lgamma(linspace(0.1,1.0,8)) — actually fine values here,
# but model_0222 uses lgamma(reduce_sum(stack([x,x]))) where sum can be negative.
# model_0229 uses lgamma(linalg.trace(x@xT)) — trace is always >= 0 but can be 0.

import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

# lgamma at non-positive integers/zero
x_bad = tf.constant([0.0, -1.0, -2.0, -3.0, 0.5, 1.0])
out = tf.math.lgamma(x_bad)
has_nan_inf = bool(tf.reduce_any(tf.math.is_nan(out) | tf.math.is_inf(out)))
print(f"lgamma([0,-1,-2,-3,0.5,1.0]): {out.numpy()}, has_nan_or_inf = {has_nan_inf}")

# model_0222 pattern: lgamma(reduce_sum(stacked x)) where x can be negative
x = tf.constant([[-0.3, 0.8, -0.6, 0.2, -0.9, 0.4, -0.1, 0.7]])
# stack doubles the tensor
stacked = tf.stack([x, x], axis=-1)
total = tf.reduce_sum(stacked)
out2 = tf.math.lgamma(total)
has_nan2 = bool(tf.math.is_nan(out2) | tf.math.is_inf(out2))
print(f"lgamma(reduce_sum(stacked_x)) = lgamma({total.numpy():.4f}) = {out2.numpy()}, has_issue = {has_nan2}")

if has_nan_inf or has_nan2:
    print("PASS")
else:
    print("NOT REPRODUCED")
