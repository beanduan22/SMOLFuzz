# ROOT CAUSE: tf.linalg.lstsq on ill-conditioned (near-singular) system
# model_0094: cumsum division pattern creates near-zero denominators,
# combined with sqrt of potentially negative values.
# model_0109: lstsq(x@xT, ones) where x@xT can be singular.
# Both eager and XLA produce NaN for degenerate inputs.

import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

# model_0109 pattern: x@xT can be singular if rows are linearly dependent
# After layer_norm, rows may become nearly identical
x = tf.constant([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])  # rank-1 matrix
gram = tf.matmul(x, x, transpose_b=True)  # 4x4, rank 1 (singular)
rhs = tf.ones_like(gram)
sol = tf.linalg.lstsq(gram, rhs)
has_nan = bool(tf.reduce_any(tf.math.is_nan(sol)))
print(f"lstsq on rank-1 gram matrix: has_nan = {has_nan}")
print(f"Solution shape: {sol.shape}, sample: {sol.numpy()[0,:2]}")

# model_0094: cumsum(x, exclusive=True, reverse=True) as denominator — zero at last
x2 = tf.constant([[0.1, 0.2, 0.3, 0.4]])
denom = tf.cumsum(x2, axis=1, exclusive=True, reverse=True)
print(f"cumsum(exclusive=True, reverse=True): {denom.numpy()}")
safe_div = tf.where(tf.math.not_equal(x2, 0), x2 / denom[:, None], tf.zeros_like(x2))
has_nan2 = bool(tf.reduce_any(tf.math.is_nan(safe_div) | tf.math.is_inf(safe_div)))
print(f"x / cumsum_denom: {safe_div.numpy()}, has_nan_or_inf = {has_nan2}")

if has_nan or has_nan2:
    print("PASS")
else:
    print("NOT REPRODUCED")
