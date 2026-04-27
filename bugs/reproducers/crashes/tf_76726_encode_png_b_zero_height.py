# https://github.com/tensorflow/tensorflow/issues/76726
# expect: aborted (core dumped)  (variant: explicit 0xWxC empty image)
import tensorflow as tf

image = tf.zeros((0, 4, 3), dtype=tf.uint8)
tf.io.encode_png(image)
