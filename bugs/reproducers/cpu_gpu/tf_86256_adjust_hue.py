# https://github.com/tensorflow/tensorflow/issues/86256
import tensorflow as tf

images = tf.constant([
    [[ 1.9720840,  2.1302242, -0.1902120],
     [ 0.6557856, -1.3016001,  1.1452782]],
    [[-2.2193234,  0.3198028,  0.9568117],
     [-0.3937407, -0.0503466, -0.3693791]],
], dtype=tf.float32)
delta = tf.constant(-0.7441734, dtype=tf.float32)
with tf.device("CPU:0"):
    cpu = tf.image.adjust_hue(images, delta)
with tf.device("GPU:0"):
    gpu = tf.image.adjust_hue(images, delta)
print("original   diff:", tf.reduce_max(tf.abs(cpu - gpu)).numpy())

x = tf.random.uniform((4, 16, 16, 3), 0.0, 1.0, seed=17)
with tf.device("CPU:0"):
    cpu = tf.image.adjust_hue(x, -0.5)
with tf.device("GPU:0"):
    gpu = tf.image.adjust_hue(x, -0.5)
print("4x16x16x3  diff:", tf.reduce_max(tf.abs(cpu - gpu)).numpy())

x = tf.cast(tf.random.uniform((6, 6, 3), 0.0, 1.0, seed=23), tf.float16)
with tf.device("CPU:0"):
    cpu = tf.image.adjust_hue(x, tf.cast(0.31, tf.float16))
with tf.device("GPU:0"):
    gpu = tf.image.adjust_hue(x, tf.cast(0.31, tf.float16))
print("6x6x3/f16  diff:", tf.reduce_max(tf.abs(tf.cast(cpu, tf.float32) - tf.cast(gpu, tf.float32))).numpy())
