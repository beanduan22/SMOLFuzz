"""
Bug: tf.math.abs returns nan on GPU where CPU returns inf for complex64 inputs
containing inf and nan.

Same root cause as the complex128 bug: IEEE 754 mandates |inf + nan*j| = inf
because the magnitude is dominated by the infinite component. CPU (C++ hypot)
follows this rule. GPU CUDA kernel does not, returning nan instead.

This bug affects both complex64 and complex128 dtypes.
"""
import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf

cases = [
    complex(np.inf, np.nan),   # |inf + nan*j| should be inf
    complex(np.nan, np.inf),   # |nan + inf*j| should be inf
]

for v in cases:
    x = tf.constant([v], dtype=tf.complex64)
    with tf.device("/CPU:0"):
        cpu = tf.math.abs(x).numpy()[0]
    with tf.device("/GPU:0"):
        gpu = tf.math.abs(x).numpy()[0]
    print(f"abs({v})")
    print(f"  CPU: {cpu}")
    print(f"  GPU: {gpu}")
    print(f"  Asymmetric: CPU=inf GPU=nan -> {'BUG' if np.isinf(cpu) and np.isnan(gpu) else 'ok'}")
    print()
