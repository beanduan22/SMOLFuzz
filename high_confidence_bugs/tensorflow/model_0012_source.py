# SMOLFuzz TF model 12
# APIs: ['tf.linalg.tridiagonal_matmul', 'tf.keras.optimizers.Adadelta', 'tf.math.special.dawsn', 'tf.math.subtract', 'tf.signal.inverse_stft_window_fn', 'tf.keras.utils.serialize_keras_object', 'tf.keras.layers.RepeatVector', 'tf.keras.callbacks.CallbackList', 'tf.math.add_n', 'tf.experimental.numpy.divide', 'tf.keras.layers.InputSpec', 'tf.math.multiply']

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.dense1 = tf.keras.layers.Dense(16, input_dim=8)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(1, input_dim=16)

    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.batch_norm(x)
        x = self.layer_norm(x)
        x = tf.math.multiply(x, tf.experimental.numpy.divide(tf.math.add_n([x]), 2))
        return self.dense2(x)
