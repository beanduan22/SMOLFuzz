# https://github.com/tensorflow/tensorflow/issues/76726
# expect: aborted (core dumped)
import tensorflow as tf

image = tf.cast(tf.tile([[[0, 0, 0, 1]], [[0, 0, 1, 0]]], [0, 0, 1]), tf.uint8)
tf.io.encode_png(image)
