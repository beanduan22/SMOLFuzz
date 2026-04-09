# SMOLFuzz TF model 158
# APIs: ['tf.keras.layers.GlobalAveragePooling1D', 'tf.experimental.numpy.string_', 'tf.sign', 'tf.math.special.bessel_j1', 'tf.math.reciprocal', 'tf.experimental.numpy.arccosh', 'tf.math.atan', 'tf.experimental.numpy.result_type', 'tf.keras.layers.experimental.preprocessing.RandomHeight', 'tf.keras.layers.DepthwiseConv1D', 'tf.math.cumsum', 'tf.math.lgamma']

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.dense1 = tf.keras.layers.Dense(16, input_dim=8)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling1D()

    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.batch_norm(x)
        x = tf.math.special.bessel_j1(tf.math.reciprocal(x))
        x = tf.sign(x) * tf.math.atan(x)
        return tf.reduce_mean(x)
