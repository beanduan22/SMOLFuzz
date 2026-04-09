# SMOLFuzz TF model 29
# APIs: ['tf.keras.applications.densenet.DenseNet169', 'tf.experimental.numpy.divmod', 'tf.experimental.numpy.equal', 'tf.nn.moments', 'tf.keras.metrics.KLDivergence', 'tf.keras.layers.Dropout', 'tf.keras.layers.AdditiveAttention', 'tf.math.reduce_all', 'tf.keras.backend.epsilon', 'tf.experimental.numpy.arctanh', 'tf.cosh', 'tf.experimental.numpy.einsum']

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.dense1 = tf.keras.layers.Dense(8, input_dim=8)
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(8)
        self.dense3 = tf.keras.layers.Dense(1)

    def call(self, x, training=False):
        x = self.dense1(x)
        x = tf.cosh(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        x = self.dense3(x)
        return x
