# ROOT CAUSE: tf.math.acos receives input outside [-1, 1] domain
# acos(x) is only defined for x in [-1, 1]. When applied directly to
# layer output (e.g., after Dense+BN), values can exceed this range.
# model_0214 applies log(acos(x)) — acos can also return 0, making log(0) = -inf.
# eager_nan=True but xla_nan=False (XLA may handle this differently).

import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

# acos is undefined outside [-1, 1]
x_out = tf.constant([0.5, 1.5, -0.3, 2.0, -1.5])
out_acos = tf.math.acos(x_out)
has_nan = bool(tf.reduce_any(tf.math.is_nan(out_acos)))
print(f"acos([0.5,1.5,-0.3,2.0,-1.5]): {out_acos.numpy()}, has_nan = {has_nan}")

# log(acos(x)) — model_0214 pattern: double domain violation
x_layer_output = tf.constant([[0.3, 1.2, -0.8, 0.9, 1.1, -1.3, 0.7, 0.0]])
out_acos2 = tf.math.acos(x_layer_output)
out_log_acos = tf.math.log(out_acos2)  # log(0) = -inf when acos(1.0)=0
has_nan_inf = bool(tf.reduce_any(tf.math.is_nan(out_log_acos) | tf.math.is_inf(out_log_acos)))
print(f"log(acos(x)): {out_log_acos.numpy()}, has_nan_or_inf = {has_nan_inf}")

if has_nan or has_nan_inf:
    print("PASS")
else:
    print("NOT REPRODUCED")
