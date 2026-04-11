"""
Bug: Casting float32 NaN or +Inf to integer types gives completely different results
on CPU vs GPU in TensorFlow.

Origin: Found in PaddlePaddle (issue #72779). The same asymmetry exists in TF.

Root cause: Same as PyTorch — C/CUDA standards both leave float-to-int conversion
for NaN/Inf as undefined behavior. CPU x86 CVTTSS2SI saturates to INT_MIN for any
out-of-range value. GPU CUDA conversion returns 0 for NaN and INT_MAX for +Inf,
producing completely different bit patterns than the CPU path.
"""
import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

cases = [
    (float("nan"),  tf.int32,  "float32(nan)  -> int32"),
    (float("inf"),  tf.int32,  "float32(inf)  -> int32"),
    (float("inf"),  tf.int64,  "float32(inf)  -> int64"),
]

for val, dtype, name in cases:
    x = tf.constant([val], dtype=tf.float32)
    with tf.device("/CPU:0"):
        cpu_v = tf.cast(x, dtype).numpy()[0]
    with tf.device("/GPU:0"):
        gpu_v = tf.cast(x, dtype).numpy()[0]
    print(f"{name}:  CPU={cpu_v}  GPU={gpu_v}  -> {'BUG' if cpu_v != gpu_v else 'ok'}")
