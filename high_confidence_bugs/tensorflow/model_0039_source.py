# SMOLFuzz TF model 39
# APIs: ['tf.math.imag', 'tf.math.unsorted_segment_max', 'tf.linalg.logdet', 'tf.keras.layers.experimental.RandomFourierFeatures', 'tf.experimental.numpy.atleast_2d', 'tf.nn.embedding_lookup', 'tf.linalg.eye', 'tf.nn.log_softmax', 'tf.keras.layers.SeparableConvolution1D', 'tf.keras.metrics.CategoricalCrossentropy', 'tf.keras.layers.experimental.preprocessing.RandomFlip', 'tf.experimental.numpy.isfinite']

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.dense1 = tf.keras.layers.Dense(8)
        self.bn = tf.keras.layers.BatchNormalization()
        self.ln = tf.keras.layers.LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.rff = tf.keras.layers.experimental.RandomFourierFeatures(output_dim=8, kernel_initializer="gaussian", scale=1.0)

    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.bn(x, training=training)
        x = self.ln(x)
        x = tf.math.imag(tf.linalg.logdet(tf.linalg.eye(8) + tf.experimental.numpy.atleast_2d(x)))
        x = self.dropout(x, training=training)
        return x
