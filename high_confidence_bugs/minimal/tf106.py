import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf

class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(16)
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, x, training=False):
        x = self.bn(self.dense(x), training=training)
        x = tf.math.digamma(tf.cast(x, dtype=tf.float64))
        return tf.reduce_sum(x)

tf.random.set_seed(42)
x = np.array([
    [0.49671414, -0.13826430,  0.64768857,  1.52302980, -0.23415338, -0.23413695,  1.57921278,  0.76743472],
    [-0.46947438,  0.54256004, -0.46341768, -0.46572974,  0.24196227, -1.91328025, -1.72491789, -0.56228751],
    [-1.01283109,  0.31424734, -0.90802407, -1.41230369,  1.46564877, -0.22577630,  0.06752820, -1.42474818],
    [-0.54438275,  0.11092259, -1.15099359,  0.37569803, -0.60063869, -0.29169375, -0.60170662,  1.85227823],
], dtype=np.float32)

with tf.device("/CPU:0"):
    m_cpu = Model()
    _ = m_cpu(tf.constant(x))

with tf.device("/GPU:0"):
    m_gpu = Model()
    _ = m_gpu(tf.constant(x))
    for vc, vg in zip(m_cpu.variables, m_gpu.variables):
        vg.assign(tf.cast(vc, vg.dtype))

cpu = m_cpu(tf.constant(x), training=False).numpy()
gpu = m_gpu(tf.constant(x), training=False).numpy()

print("CPU:", cpu)
print("GPU:", gpu)
print(f"L2: {np.sqrt(np.sum((cpu - gpu)**2)):.4e}")
