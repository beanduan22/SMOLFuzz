import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf

class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(8, kernel_constraint=tf.keras.constraints.min_max_norm(-0.5, 0.5))
        self.bn = tf.keras.layers.BatchNormalization()
        self.dense2 = tf.keras.layers.Dense(1)

    def call(self, x, training=False):
        x = self.bn(self.dense1(x), training=training)
        return self.dense2(tf.signal.dct(x, type=2))

tf.random.set_seed(42)
x = np.array([
    [0.4967, -0.1383, 0.6477, 1.5230, -0.2342, -0.2341, 1.5792, 0.7674],
    [-0.4695,  0.5426, -0.4634, -0.4657,  0.2420, -1.9133, -1.7249, -0.5623],
    [-1.0128,  0.3142, -0.9080, -1.4123,  1.4656, -0.2258,  0.0675, -1.4247],
    [-0.5444,  0.1109, -1.1510,  0.3757, -0.6006, -0.2917, -0.6017,  1.8523],
], dtype=np.float32)

with tf.device("/CPU:0"):
    m_cpu = Model()
    _ = m_cpu(tf.constant(x))

with tf.device("/GPU:0"):
    m_gpu = Model()
    _ = m_gpu(tf.constant(x))
    for vc, vg in zip(m_cpu.variables, m_gpu.variables):
        vg.assign(tf.cast(vc, vg.dtype))

cpu = m_cpu(tf.constant(x), training=False).numpy().flatten()
gpu = m_gpu(tf.constant(x), training=False).numpy().flatten()

print("CPU:", cpu)
print("GPU:", gpu)
print(f"L2: {np.sqrt(np.sum((cpu - gpu)**2)):.4e}")
