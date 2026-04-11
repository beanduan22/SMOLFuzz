"""
Bug: tf.math.abs returns inconsistent results for complex128 inputs
containing inf and nan.

For complex(inf, nan) and complex(nan, inf), the mathematically correct
result is inf (since |inf + nan*j| = sqrt(inf^2 + nan^2) = inf under
IEEE 754 rules used by most C math libraries).

CPU returns inf (correct).
GPU returns nan (incorrect).

Related: https://github.com/tensorflow/tensorflow/issues/98410
"""
import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf

cases = [
    complex(np.inf, np.nan),   # |inf + nan*j| should be inf
    complex(np.nan, np.inf),   # |nan + inf*j| should be inf
]

for v in cases:
    x = tf.constant([v], dtype=tf.complex128)
    with tf.device('/CPU:0'):
        cpu = tf.math.abs(x).numpy()[0]
    with tf.device('/GPU:0'):
        gpu = tf.math.abs(x).numpy()[0]
    print(f"abs({v})")
    print(f"  CPU: {cpu}")
    print(f"  GPU: {gpu}")
    print(f"  Asymmetric: CPU=inf GPU=nan -> {'BUG' if np.isinf(cpu) and np.isnan(gpu) else 'ok'}")
    print()
