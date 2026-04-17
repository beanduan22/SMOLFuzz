# SMOLFuzz TF model 100 | attempts=4
# APIS_SELECTED = ['tf.optimizers.get', 'tf.losses.mae', 'tf.keras.layers.experimental.preprocessing.RandomZoom', 'tf.keras.activations.gelu', 'tf.keras.layers.RandomRotation', 'tf.initializers.serialize', 'tf.metrics.deserialize', 'tf.keras.layers.GlobalAvgPool2D', 'tf.initializers.Initializer', 'tf.keras.metrics.Mean', 'tf.losses.SquaredHinge', 'tf.slice', 'tf.zeros', 'tf.linalg.pinv', 'tf.math.segment_min', 'tf.math.reduce_logsumexp', 'tf.signal.mdct', 'tf.bitwise.right_shift', 'tf.math.segment_max', 'tf.extract_volume_patches']
# USED_APIS = ['tf.keras.layers.Dense', 'tf.keras.layers.BatchNormalization', 'tf.keras.layers.LayerNormalization', 'tf.keras.activations.gelu', 'tf.GradientTape', 'tf.device']

class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(8)
        self.bn = tf.keras.layers.BatchNormalization()
        self.ln = tf.keras.layers.LayerNormalization()
        self.activation = tf.keras.layers.Activation(tf.keras.activations.gelu)
        self.d2 = tf.keras.layers.Dense(8)
        self.dense_out = tf.keras.layers.Dense(1)

    def call(self, x, training=False):
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = self.d1(x)
            y = self.bn(y, training=training)
            y = self.ln(y)
            y = self.activation(y)
            y = self.d2(y)
        g = tape.gradient(y, x)
        with tf.device('/CPU:0'):
            z = self.dense_out(g + y)
        return z

USED_APIS = ["tf.keras.layers.Dense", "tf.keras.layers.BatchNormalization",
             "tf.keras.layers.LayerNormalization", "tf.keras.activations.gelu",
             "tf.GradientTape", "tf.device"]
