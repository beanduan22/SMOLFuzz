# SMOLFuzz TF model 90 | attempts=3
# APIS_SELECTED = ['tf.keras.optimizers.Adam', 'tf.keras.layers.Wrapper', 'tf.losses.categorical_crossentropy', 'tf.metrics.BinaryCrossentropy', 'tf.keras.activations.exponential', 'tf.keras.layers.experimental.preprocessing.StringLookup', 'tf.keras.metrics.mean_squared_logarithmic_error', 'tf.keras.backend.reset_uids', 'tf.keras.metrics.FalseNegatives', 'tf.keras.layers.experimental.SyncBatchNormalization', 'tf.nn.dropout', 'tf.math.is_finite', 'tf.slice', 'tf.TensorSpec', 'tf.signal.mdct', 'tf.math.floormod', 'tf.math.special.bessel_k0e', 'tf.signal.irfft2d', 'tf.math.expm1', 'tf.math.reduce_prod']
# USED_APIS = ['tf.keras.layers.Dense', 'tf.keras.layers.BatchNormalization', 'tf.keras.layers.LayerNormalization', 'tf.keras.activations.exponential', 'tf.keras.layers.StringLookup', 'tf.GradientTape', 'tf.slice']

class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(8)
        self.bn = tf.keras.layers.BatchNormalization()
        self.ln = tf.keras.layers.LayerNormalization()
        self.act1 = tf.keras.layers.Activation(tf.keras.activations.exponential)
        self.lookup = tf.keras.layers.StringLookup(vocabulary=['a', 'b', 'c', 'd'], output_mode='int')
        self.d2 = tf.keras.layers.Dense(8)

    def call(self, x, training=False):
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = self.d1(x)
            z = self.bn(y, training=training)
        g = tape.gradient(z, x)
        h = self.ln(z + g)
        i = self.act1(h)
        j = tf.slice(i, [0, 0], [-1, 8])
        l = self.d2(j)
        return l

USED_APIS = ["tf.keras.layers.Dense", "tf.keras.layers.BatchNormalization",
             "tf.keras.layers.LayerNormalization", "tf.keras.activations.exponential",
             "tf.keras.layers.StringLookup", "tf.GradientTape",
             "tf.slice"]
