# https://github.com/tensorflow/tensorflow/issues/86350
import tensorflow as tf

x = tf.cast(tf.random.normal((2, 3, 4, 5), seed=31), tf.bfloat16)
y = tf.cast(tf.random.normal((2, 3, 5, 6), seed=32), tf.bfloat16)
with tf.device("CPU:0"):
    cpu = tf.raw_ops.BatchMatMulV2(x=x, y=y)
with tf.device("GPU:0"):
    gpu = tf.raw_ops.BatchMatMulV2(x=x, y=y)
print("4D/bf16 diff:", tf.reduce_max(tf.abs(tf.cast(cpu, tf.float32) - tf.cast(gpu, tf.float32))).numpy())

x = tf.cast(tf.random.normal((2, 3, 4, 4, 5), seed=33), tf.bfloat16)
y = tf.cast(tf.random.normal((2, 3, 4, 5, 6), seed=34), tf.bfloat16)
with tf.device("CPU:0"):
    cpu = tf.raw_ops.BatchMatMulV2(x=x, y=y)
with tf.device("GPU:0"):
    gpu = tf.raw_ops.BatchMatMulV2(x=x, y=y)
print("5D/bf16 diff:", tf.reduce_max(tf.abs(tf.cast(cpu, tf.float32) - tf.cast(gpu, tf.float32))).numpy())

x = tf.cast(tf.random.normal((2, 5, 4), seed=35), tf.bfloat16)
y = tf.cast(tf.random.normal((2, 5, 6), seed=36), tf.bfloat16)
with tf.device("CPU:0"):
    cpu = tf.raw_ops.BatchMatMulV2(x=x, y=y, adj_x=True)
with tf.device("GPU:0"):
    gpu = tf.raw_ops.BatchMatMulV2(x=x, y=y, adj_x=True)
print("3D/adj_x diff:", tf.reduce_max(tf.abs(tf.cast(cpu, tf.float32) - tf.cast(gpu, tf.float32))).numpy())
