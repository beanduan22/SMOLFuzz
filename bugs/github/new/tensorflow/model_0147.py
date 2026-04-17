# SMOLFuzz TF model 147 | attempts=2
# APIS_SELECTED = ['tf.keras.optimizers.Adagrad', 'tf.keras.experimental.CosineDecay', 'tf.keras.layers.LeakyReLU', 'tf.metrics.mape', 'tf.keras.losses.mean_squared_error', 'tf.keras.losses.LogCosh', 'tf.losses.deserialize', 'tf.losses.mean_squared_logarithmic_error', 'tf.keras.losses.KLD', 'tf.keras.metrics.PrecisionAtRecall', 'tf.keras.constraints.NonNeg', 'tf.repeat', 'tf.Module', 'tf.linalg.svd', 'tf.math.special.fresnel_cos', 'tf.tanh', 'tf.math.erfcinv', 'tf.foldr', 'tf.reduce_max', 'tf.signal.ifftshift']
# USED_APIS = ['tf.keras.layers.Dense', 'tf.keras.layers.LeakyReLU', 'tf.keras.layers.BatchNormalization', 'tf.keras.layers.LayerNormalization', 'tf.keras.constraints.NonNeg', 'tf.keras.optimizers.Adagrad', 'tf.GradientTape', 'tf.device']

class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(8)
        self.leaky_relu = tf.keras.layers.LeakyReLU()
        self.batch_norm = tf.keras.layers.BatchNormalization(input_shape=(8,))
        self.layer_norm = tf.keras.layers.LayerNormalization(input_shape=(8,))
        self.dense2 = tf.keras.layers.Dense(8, kernel_constraint=tf.keras.constraints.NonNeg())
        self.optimizer = tf.keras.optimizers.Adagrad()

    def call(self, x, training=False):
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = self.dense1(x)
            y = self.leaky_relu(y)
            y = self.batch_norm(y, training=training)
            y = self.layer_norm(y)
            z = self.dense2(y)
        
        g = tape.gradient(z, x)
        with tf.device('/CPU:0'):
            result = z + g
        
        return result

USED_APIS = ["tf.keras.layers.Dense", "tf.keras.layers.LeakyReLU", "tf.keras.layers.BatchNormalization",
             "tf.keras.layers.LayerNormalization", "tf.keras.constraints.NonNeg", "tf.keras.optimizers.Adagrad",
             "tf.GradientTape", "tf.device"]
