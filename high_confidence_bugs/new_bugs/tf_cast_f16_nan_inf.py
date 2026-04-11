"""
Bug: Casting float16 NaN or +Inf to integer types gives different results
on CPU vs GPU in TensorFlow.

Root cause: Same undefined behavior as the float32 cast bug (from PaddlePaddle #72779).
The C and CUDA standards leave float-to-int conversion for NaN/Inf as undefined
behavior for ALL floating-point types. CPU x86 CVTTPH2SI / VCVTTSH2SI saturates
to INT_MIN for out-of-range values. GPU CUDA float16-to-int returns 0 for NaN
and INT_MAX for +Inf — opposite saturation semantics.
"""
import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf

cases = [
    (np.float16("nan"),  tf.int32,  "float16(nan)  -> int32"),
    (np.float16("inf"),  tf.int32,  "float16(inf)  -> int32"),
    (np.float16("inf"),  tf.int64,  "float16(inf)  -> int64"),
]

for val, dtype, name in cases:
    x = tf.constant([val], dtype=tf.float16)
    with tf.device("/CPU:0"):
        cpu_v = tf.cast(x, dtype).numpy()[0]
    with tf.device("/GPU:0"):
        gpu_v = tf.cast(x, dtype).numpy()[0]
    status = "BUG" if cpu_v != gpu_v else "ok"
    print(f"{name}:  CPU={cpu_v}  GPU={gpu_v}  -> {status}")
