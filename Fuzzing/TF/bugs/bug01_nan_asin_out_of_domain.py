# ROOT CAUSE: tf.math.asin receives input outside [-1, 1] domain
# asin(x) is only defined for x in [-1, 1]. When cumsum+exp produces values
# much larger than 1.0, asin returns NaN. Both eager and XLA produce NaN.
# Representative model: model_0004 (used_apis: tf.math.asin, tf.math.exp, tf.math.cumsum)

import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

x = tf.constant([[1.0, 2.0, 3.0, 4.0]])
# cumsum(exp(x)) produces large values far outside [-1, 1]
out = tf.math.asin(tf.math.cumsum(tf.math.exp(x)))

has_nan = bool(tf.reduce_any(tf.math.is_nan(out)))
print(f"asin(cumsum(exp(x))) = {out.numpy()}")
print(f"has_nan = {has_nan}")

if has_nan:
    print("PASS")
else:
    print("NOT REPRODUCED")
