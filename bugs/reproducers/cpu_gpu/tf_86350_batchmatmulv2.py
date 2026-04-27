# https://github.com/tensorflow/tensorflow/issues/86350
import tensorflow as tf

x = tf.constant([
    [[[[ -1.3594, -0.3027], [-1.4141,  0.2969]],
      [[ -0.9141,  1.7812], [ 1.2266,  0.8594]]],
     [[[  0.8359, -0.9414], [-1.7969, -0.7461]],
      [[  0.3164,  0.3691], [ 0.7656,  0.2354]]]],
    [[[[ -0.5898,  1.3516], [ 0.4902, -0.1045]],
      [[ -0.1099,  1.5078], [ 0.2852, -0.0957]]],
     [[[-0.9883,  1.3203], [-0.2715, -1.7578]],
      [[ -0.1602, -0.4336], [-0.6875, -0.4492]]]],
], dtype=tf.bfloat16)
y = tf.constant([
    [[[  0.6836, -0.6562], [-0.5508, -0.8438]],
     [[  1.6094, -0.9883], [-0.1318,  1.1094]]],
    [[[  0.4062, -1.1094], [-0.7188, -1.7578]],
     [[ -1.0391, -0.6602], [ 0.8359, -0.6562]]],
], dtype=tf.bfloat16)
with tf.device("CPU:0"):
    cpu = tf.raw_ops.BatchMatMulV2(x=x, y=y)
with tf.device("GPU:0"):
    gpu = tf.raw_ops.BatchMatMulV2(x=x, y=y)
print("original:", "max|diff|=", tf.reduce_max(tf.abs(tf.cast(cpu, tf.float32) - tf.cast(gpu, tf.float32))).numpy())

x2 = tf.cast(tf.random.normal((2, 3, 4, 5), seed=31), tf.bfloat16)
y2 = tf.cast(tf.random.normal((2, 3, 5, 6), seed=32), tf.bfloat16)
with tf.device("CPU:0"):
    cpu = tf.raw_ops.BatchMatMulV2(x=x2, y=y2)
with tf.device("GPU:0"):
    gpu = tf.raw_ops.BatchMatMulV2(x=x2, y=y2)
print("4D/bf16:", "max|diff|=", tf.reduce_max(tf.abs(tf.cast(cpu, tf.float32) - tf.cast(gpu, tf.float32))).numpy())

x3 = tf.cast(tf.random.normal((2, 3, 4, 4, 5), seed=33), tf.bfloat16)
y3 = tf.cast(tf.random.normal((2, 3, 4, 5, 6), seed=34), tf.bfloat16)
with tf.device("CPU:0"):
    cpu = tf.raw_ops.BatchMatMulV2(x=x3, y=y3)
with tf.device("GPU:0"):
    gpu = tf.raw_ops.BatchMatMulV2(x=x3, y=y3)
print("5D/bf16:", "max|diff|=", tf.reduce_max(tf.abs(tf.cast(cpu, tf.float32) - tf.cast(gpu, tf.float32))).numpy())
