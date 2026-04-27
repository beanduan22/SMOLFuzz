# https://github.com/tensorflow/tensorflow/issues/97204
import numpy as np
import tensorflow as tf

np.random.seed(202)
x = tf.constant(np.random.uniform(-32767., 127., size=(4, 1)).astype(np.float32))
y = tf.constant(np.random.uniform(0., 89., size=(1, 28, 2, 3, 2)).astype(np.float32))
with tf.device("CPU:0"):
    print("cpu (4,1)x(1,28,2,3,2):", tf.raw_ops.NotEqual(x=x, y=y, incompatible_shape_error=False))
with tf.device("GPU:0"):
    try:
        print("gpu:", tf.raw_ops.NotEqual(x=x, y=y, incompatible_shape_error=False))
    except tf.errors.InvalidArgumentError as e:
        print("gpu:", "InvalidArgumentError")

x = tf.zeros((3, 5), dtype=tf.int32)
y = tf.zeros((4, 5), dtype=tf.int32)
with tf.device("CPU:0"):
    print("cpu (3,5)x(4,5)/int32:", tf.raw_ops.NotEqual(x=x, y=y, incompatible_shape_error=False))
with tf.device("GPU:0"):
    try:
        print("gpu:", tf.raw_ops.NotEqual(x=x, y=y, incompatible_shape_error=False))
    except tf.errors.InvalidArgumentError:
        print("gpu:", "InvalidArgumentError")
