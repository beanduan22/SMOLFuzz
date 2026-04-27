# https://github.com/tensorflow/tensorflow/issues/96180
import numpy as np
import tensorflow as tf

data = np.array([[0.0 + np.inf], [np.inf + 0.0], [np.inf + np.inf]], dtype="complex128")
with tf.device("CPU"):
    print("cpu pure-inf   :", tf.math.reciprocal(tf.constant(data)).numpy().flatten())
with tf.device("GPU"):
    print("gpu pure-inf   :", tf.math.reciprocal(tf.constant(data)).numpy().flatten())

data = np.array([[0.0 + np.inf], [np.inf + 0.0], [np.inf + np.inf], [np.nan + 0.0]], dtype="complex128")
with tf.device("CPU"):
    print("cpu inf+nan    :", tf.math.reciprocal(tf.constant(data)).numpy().flatten())
with tf.device("GPU"):
    print("gpu inf+nan    :", tf.math.reciprocal(tf.constant(data)).numpy().flatten())

data = np.array([np.inf + 1j, 1j * np.inf, np.inf + np.inf * 1j], dtype="complex64")
with tf.device("CPU"):
    print("cpu c64        :", tf.math.reciprocal(tf.constant(data)).numpy())
with tf.device("GPU"):
    print("gpu c64        :", tf.math.reciprocal(tf.constant(data)).numpy())
