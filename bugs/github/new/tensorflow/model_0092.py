# SMOLFuzz TF model 92 | attempts=1
# APIS_SELECTED = ['tf.optimizers.Optimizer', 'tf.metrics.SparseCategoricalAccuracy', 'tf.keras.layers.GlobalMaxPooling1D', 'tf.initializers.Orthogonal', 'tf.metrics.hinge', 'tf.keras.losses.MeanAbsolutePercentageError', 'tf.keras.losses.kld', 'tf.keras.activations.swish', 'tf.keras.layers.experimental.preprocessing.Discretization', 'tf.keras.layers.RandomHeight', 'tf.keras.layers.experimental.preprocessing.RandomHeight', 'tf.not_equal', 'tf.split', 'tf.name_scope', 'tf.saturate_cast', 'tf.fingerprint', 'tf.linalg.eigvals', 'tf.math.softsign', 'tf.sqrt', 'tf.math.special.bessel_y0']
# USED_APIS = ['tf.keras.layers.Dense', 'tf.initializers.Orthogonal', 'tf.keras.layers.BatchNormalization', 'tf.keras.layers.LayerNormalization', 'tf.keras.activations.swish', 'tf.GradientTape', 'tf.device', 'tf.keras.layers.GlobalMaxPooling1D']

class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(8, kernel_initializer=tf.initializers.Orthogonal())
        self.bn = tf.keras.layers.BatchNormalization()
        self.ln = tf.keras.layers.LayerNormalization()
        self.dense2 = tf.keras.layers.Dense(8)
        self.global_max_pooling = tf.keras.layers.GlobalMaxPooling1D(input_shape=(4, 8))
        self.activation = tf.keras.activations.swish
        self.dense3 = tf.keras.layers.Dense(4)

    def call(self, x, training=False):
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = self.dense1(x)
            y = self.bn(y, training=training)
            y = self.ln(y)
            y = self.activation(y)
            z = self.dense2(y)
        g = tape.gradient(z, x)
        h = z + g
        with tf.device('/CPU:0'):
            i = self.global_max_pooling(tf.expand_dims(h, axis=1))
        j = self.dense3(i)
        return j

USED_APIS = ["tf.keras.layers.Dense", "tf.initializers.Orthogonal",
             "tf.keras.layers.BatchNormalization", "tf.keras.layers.LayerNormalization",
             "tf.keras.activations.swish", "tf.GradientTape", "tf.device",
             "tf.keras.layers.GlobalMaxPooling1D"]
