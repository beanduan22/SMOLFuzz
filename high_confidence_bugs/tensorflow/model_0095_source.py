# SMOLFuzz TF model 95
# APIs: ['tf.keras.layers.ELU', 'tf.nn.avg_pool2d', 'tf.experimental.numpy.complex128', 'tf.experimental.numpy.tri', 'tf.experimental.numpy.zeros_like', 'tf.experimental.numpy.complex64', 'tf.keras.metrics.mean_absolute_percentage_error', 'tf.keras.layers.Dense', 'tf.experimental.numpy.asarray', 'tf.math.expm1', 'tf.keras.layers.Bidirectional', 'tf.keras.layers.GlobalAveragePooling2D']

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.dense1 = tf.keras.layers.Dense(8, input_dim=8)
        self.elu = tf.keras.layers.ELU()
        self.dense2 = tf.keras.layers.Dense(8)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.dense3 = tf.keras.layers.Dense(1)

    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.elu(x)
        x = self.dense2(x)
        x = self.batch_norm(x, training=training)
        x = self.dense3(x)
        return x
