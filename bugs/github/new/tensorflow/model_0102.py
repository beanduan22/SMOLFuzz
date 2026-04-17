# SMOLFuzz TF model 102 | attempts=2
# APIS_SELECTED = ['tf.keras.optimizers.Optimizer', 'tf.keras.layers.RandomHeight', 'tf.keras.initializers.Ones', 'tf.losses.KLD', 'tf.losses.mean_absolute_percentage_error', 'tf.initializers.VarianceScaling', 'tf.nn.weighted_cross_entropy_with_logits', 'tf.initializers.HeNormal', 'tf.metrics.PrecisionAtRecall', 'tf.keras.layers.Permute', 'tf.keras.layers.experimental.preprocessing.Hashing', 'tf.sequence_mask', 'tf.as_dtype', 'tf.math.special.bessel_k0e', 'tf.acosh', 'tf.tanh', 'tf.histogram_fixed_width', 'tf.math.special.bessel_k0', 'tf.math.is_non_decreasing', 'tf.signal.fftshift']
# USED_APIS = ['tf.keras.layers.Dense', 'tf.keras.initializers.HeNormal', 'tf.tanh', 'tf.keras.layers.BatchNormalization', 'tf.keras.layers.LayerNormalization', 'tf.GradientTape']

class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(8, kernel_initializer=tf.initializers.HeNormal())
        self.dense2 = tf.keras.layers.Dense(8, kernel_initializer=tf.initializers.VarianceScaling())
        self.activation = tf.keras.layers.Activation(tf.tanh)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x, training=False):
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = self.dense1(x)
            y = self.dense2(y)
            y = self.activation(y)
        
        g = tape.gradient(y, x)
        y += g
        
        if training:
            y = self.batch_norm(y, training=training)
        else:
            y = self.layer_norm(y)
        
        return y

USED_APIS = ["tf.keras.layers.Dense", "tf.keras.initializers.HeNormal",
             "tf.tanh", "tf.keras.layers.BatchNormalization",
             "tf.keras.layers.LayerNormalization", "tf.GradientTape"]
