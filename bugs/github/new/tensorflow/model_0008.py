# SMOLFuzz TF model 8 | attempts=2
# APIS_SELECTED = ['tf.optimizers.Adam', 'tf.keras.losses.logcosh', 'tf.keras.metrics.get', 'tf.keras.metrics.FalseNegatives', 'tf.keras.layers.experimental.preprocessing.RandomFlip', 'tf.keras.activations.softsign', 'tf.keras.layers.Average', 'tf.initializers.deserialize', 'tf.keras.activations.swish', 'tf.keras.losses.mean_absolute_error', 'tf.keras.layers.MaxPooling1D', 'tf.shape', 'tf.TensorSpec', 'tf.reduce_sum', 'tf.linalg.global_norm', 'tf.asin', 'tf.round', 'tf.math.add', 'tf.math.special.bessel_i1', 'tf.TensorShape']
# USED_APIS = ['tf.keras.layers.Dense', 'tf.keras.layers.BatchNormalization', 'tf.keras.layers.LayerNormalization', 'tf.keras.activations.swish', 'tf.keras.layers.Average', 'tf.GradientTape', 'tf.device', 'tf.keras.metrics.FalseNegatives', 'tf.keras.losses.LogCosh', 'tf.keras.losses.MeanAbsoluteError']

class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(8)
        self.bn = tf.keras.layers.BatchNormalization()
        self.ln = tf.keras.layers.LayerNormalization()
        self.dense2 = tf.keras.layers.Dense(8)
        self.swish_activation = tf.keras.activations.swish
        self.average_layer = tf.keras.layers.Average()
        self.dense3 = tf.keras.layers.Dense(8)
        self.false_negatives = tf.keras.metrics.FalseNegatives()
        self.logcosh_loss = tf.keras.losses.LogCosh()
        self.mean_absolute_error = tf.keras.losses.MeanAbsoluteError()
        self.optimzer = tf.optimizers.Adam()

    def call(self, x, training=False):
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = self.dense1(x)
            y = self.bn(y, training=training)
            y = self.ln(y)
            y = self.swish_activation(y)
            y = self.dense2(y)
            y = self.average_layer([y, x])
            y = self.dense3(y)
        g = tape.gradient(y, x)
        with tf.device('/CPU:0'):
            y = y + g
        return y

USED_APIS = ["tf.keras.layers.Dense", "tf.keras.layers.BatchNormalization",
             "tf.keras.layers.LayerNormalization", "tf.keras.activations.swish",
             "tf.keras.layers.Average", "tf.GradientTape", "tf.device",
             "tf.keras.metrics.FalseNegatives", "tf.keras.losses.LogCosh",
             "tf.keras.losses.MeanAbsoluteError"]
