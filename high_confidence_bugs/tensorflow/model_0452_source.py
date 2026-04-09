# SMOLFuzz TF model 452
# APIs: ['tf.keras.applications.inception_resnet_v2.preprocess_input', 'tf.keras.losses.Loss', 'tf.keras.losses.kld', 'tf.nn.ctc_beam_search_decoder', 'tf.experimental.numpy.moveaxis', 'tf.math.reduce_logsumexp', 'tf.experimental.numpy.sinh', 'tf.experimental.numpy.remainder', 'tf.keras.losses.poisson', 'tf.experimental.numpy.reshape', 'tf.experimental.numpy.subtract', 'tf.keras.layers.AvgPool2D']

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.dense1 = tf.keras.layers.Dense(16, input_dim=8)
        self.bn = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.ln = tf.keras.layers.LayerNormalization()
        self.dense2 = tf.keras.layers.Dense(32, input_dim=16)

    def call(self, x, training=False):
        x = self.dense1(x)
        x = tf.experimental.numpy.sinh(x)
        x = self.bn(x, training=training)
        x = tf.math.reduce_logsumexp(x, axis=-1, keepdims=True)
        return tf.cast(tf.reduce_sum(x), dtype=tf.float32)
