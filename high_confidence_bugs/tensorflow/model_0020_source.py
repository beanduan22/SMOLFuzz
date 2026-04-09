# SMOLFuzz TF model 20
# APIs: ['tf.keras.layers.GlobalMaxPool1D', 'tf.keras.activations.gelu', 'tf.math.sign', 'tf.keras.losses.MAE', 'tf.keras.constraints.serialize', 'tf.experimental.tensorrt.ConversionParams', 'tf.linalg.norm', 'tf.experimental.numpy.diag_indices', 'tf.experimental.numpy.subtract', 'tf.keras.utils.get_file', 'tf.math.rsqrt', 'tf.math.truediv']

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.dense1 = tf.keras.layers.Dense(16, input_dim=8)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.global_max_pool = tf.keras.layers.GlobalMaxPool1D()

    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.batch_norm(x)
        x = tf.keras.activations.gelu(x)
        x = self.dropout(x, training=training)
        x = self.global_max_pool(tf.expand_dims(x, axis=-1))
        return tf.math.rsqrt(tf.linalg.norm(x))
