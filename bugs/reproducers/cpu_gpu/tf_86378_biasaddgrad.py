# https://github.com/tensorflow/tensorflow/issues/86378
import tensorflow as tf

x = tf.cast(tf.random.normal((2, 3, 2, 3, 2), seed=7), tf.bfloat16)
with tf.device("CPU:0"):
    cpu = tf.raw_ops.BiasAddGrad(out_backprop=x, data_format="NCHW")
with tf.device("GPU:0"):
    gpu = tf.raw_ops.BiasAddGrad(out_backprop=x, data_format="NCHW")
print("NCHW/bf16  diff:", tf.reduce_max(tf.abs(tf.cast(cpu, tf.float32) - tf.cast(gpu, tf.float32))).numpy())

x = tf.cast(tf.random.normal((2, 4, 4, 8), seed=11), tf.bfloat16)
with tf.device("CPU:0"):
    cpu = tf.raw_ops.BiasAddGrad(out_backprop=x, data_format="NHWC")
with tf.device("GPU:0"):
    gpu = tf.raw_ops.BiasAddGrad(out_backprop=x, data_format="NHWC")
print("NHWC/bf16  diff:", tf.reduce_max(tf.abs(tf.cast(cpu, tf.float32) - tf.cast(gpu, tf.float32))).numpy())

x = tf.cast(tf.random.normal((4, 16, 7, 11), seed=13), tf.float16)
with tf.device("CPU:0"):
    cpu = tf.raw_ops.BiasAddGrad(out_backprop=x, data_format="NCHW")
with tf.device("GPU:0"):
    gpu = tf.raw_ops.BiasAddGrad(out_backprop=x, data_format="NCHW")
print("NCHW/f16   diff:", tf.reduce_max(tf.abs(tf.cast(cpu, tf.float32) - tf.cast(gpu, tf.float32))).numpy())
