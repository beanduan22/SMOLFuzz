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
print("original  :", "cpu=", cpu.numpy().flatten()[-3:], "gpu=", gpu.numpy().flatten()[-3:])
print("max|diff| :", tf.reduce_max(tf.abs(cpu - gpu)).numpy())

x = tf.random.uniform((2, 4, 4, 3), 0.0, 1.0, seed=17)
for d_val in (-0.99, -0.7441734, -0.5, -0.1, 0.5):
    d = tf.constant(d_val, dtype=tf.float32)
    with tf.device("CPU:0"):
        cpu = tf.image.adjust_hue(x, d)
    with tf.device("GPU:0"):
        gpu = tf.image.adjust_hue(x, d)
    print(f"delta={d_val:+.2f} max|diff|={tf.reduce_max(tf.abs(cpu - gpu)).numpy():.3e}")
