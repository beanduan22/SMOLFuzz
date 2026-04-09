# ROOT CAUSE: tf.math.sinh / tf.math.cosh overflow with large-scale inputs
# sinh(x) and cosh(x) grow as exp(|x|)/2. When inputs are scaled large
# (scale_large mutation), these return NaN/inf. BatchNorm then propagates NaN.
# Representative model: model_0017 (tf.math.sinh after dense+BN, scale_large)
# Also: model_0143, model_0151, model_0270, model_0279

import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

# Small values: fine
x_small = tf.constant([[1.0, -1.0, 0.5, -0.5]])
out_small = tf.math.sinh(x_small)
print(f"sinh(small): {out_small.numpy()}, has_nan = {bool(tf.reduce_any(tf.math.is_nan(out_small)))}")

# Large values (scale_large): overflow to NaN/inf
x_large = tf.constant([[1e3, -1e3, 5e2, -5e2]])
out_sinh = tf.math.sinh(x_large)
out_cosh = tf.math.cosh(x_large)
has_nan_sinh = bool(tf.reduce_any(tf.math.is_nan(out_sinh) | tf.math.is_inf(out_sinh)))
has_nan_cosh = bool(tf.reduce_any(tf.math.is_nan(out_cosh) | tf.math.is_inf(out_cosh)))
print(f"sinh(large): {out_sinh.numpy()}, has_nan_or_inf = {has_nan_sinh}")
print(f"cosh(large): {out_cosh.numpy()}, has_nan_or_inf = {has_nan_cosh}")

if has_nan_sinh or has_nan_cosh:
    print("PASS")
else:
    print("NOT REPRODUCED")
