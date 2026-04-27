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
        print("gpu: InvalidArgumentError:", str(e).splitlines()[0][:80])

x = tf.zeros((2, 3, 4), dtype=tf.float32)
y = tf.zeros((1, 5, 6, 7, 4), dtype=tf.float32)
with tf.device("CPU:0"):
    print("cpu (2,3,4)x(1,5,6,7,4):", tf.raw_ops.NotEqual(x=x, y=y, incompatible_shape_error=False))
with tf.device("GPU:0"):
    try:
        print("gpu:", tf.raw_ops.NotEqual(x=x, y=y, incompatible_shape_error=False))
    except tf.errors.InvalidArgumentError as e:
        print("gpu: InvalidArgumentError:", str(e).splitlines()[0][:80])
