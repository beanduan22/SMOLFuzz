# https://github.com/tensorflow/tensorflow/issues/76726
# expect: aborted (core dumped)  (variant: explicit Hx0xC empty image)
import tensorflow as tf

image = tf.zeros((4, 0, 3), dtype=tf.uint8)
tf.io.encode_png(image)
