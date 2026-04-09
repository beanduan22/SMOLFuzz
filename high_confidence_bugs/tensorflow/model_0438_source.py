# SMOLFuzz TF model 438
# APIs: ['tf.experimental.numpy.shape', 'tf.keras.activations.hard_sigmoid', 'tf.keras.regularizers.deserialize', 'tf.tan', 'tf.keras.optimizers.get', 'tf.experimental.numpy.heaviside', 'tf.signal.mfccs_from_log_mel_spectrograms', 'tf.math.special.bessel_y1', 'tf.experimental.numpy.minimum', 'tf.experimental.numpy.gcd', 'tf.keras.layers.subtract', 'tf.keras.activations.serialize']

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.dense1 = tf.keras.layers.Dense(16, input_dim=8)
        self.dense2 = tf.keras.layers.Dense(32, input_dim=16)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(0.2)

    def call(self, x, training=False):
        x = tf.keras.activations.hard_sigmoid(x)
        x = self.dense1(x)
        x = self.batch_norm(x, training=training)
        x = tf.tan(x)
        return tf.experimental.numpy.minimum(tf.math.reduce_sum(x), 10.0)
