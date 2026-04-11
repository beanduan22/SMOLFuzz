"""
Bug: Casting bfloat16 NaN or +Inf to integer types gives different results
on CPU vs GPU in TensorFlow.

Root cause: Same undefined behavior as the float32 and float16 cast bugs
(from PaddlePaddle #72779). Confirmed in TF for bfloat16 inputs.
CPU x86 saturates to INT_MIN for any out-of-range value.
GPU CUDA bfloat16-to-int returns 0 for NaN and INT_MAX for +Inf.
"""
import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

cases = [
    (float("nan"), tf.int32, "bfloat16(nan) -> int32"),
    (float("inf"), tf.int32, "bfloat16(inf) -> int32"),
    (float("inf"), tf.int64, "bfloat16(inf) -> int64"),
]

for val, dtype, name in cases:
    x = tf.constant([val], dtype=tf.bfloat16)
    with tf.device("/CPU:0"):
        cpu_v = tf.cast(x, dtype).numpy()[0]
    with tf.device("/GPU:0"):
        gpu_v = tf.cast(x, dtype).numpy()[0]
    print(f"{name}:  CPU={cpu_v}  GPU={gpu_v}  -> {'BUG' if cpu_v != gpu_v else 'ok'}")
