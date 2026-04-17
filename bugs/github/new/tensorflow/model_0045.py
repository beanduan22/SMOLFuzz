# SMOLFuzz TF model 45 | attempts=1
# APIS_SELECTED = ['tf.optimizers.Adam', 'tf.keras.constraints.min_max_norm', 'tf.keras.metrics.mape', 'tf.losses.kl_divergence', 'tf.keras.losses.mean_squared_logarithmic_error', 'tf.keras.layers.RandomRotation', 'tf.metrics.get', 'tf.keras.initializers.HeNormal', 'tf.keras.initializers.LecunNormal', 'tf.keras.metrics.mean_absolute_percentage_error', 'tf.losses.kld', 'tf.reverse_sequence', 'tf.ones_like', 'tf.math.greater', 'tf.linalg.LinearOperatorAdjoint', 'tf.math.unsorted_segment_mean', 'tf.signal.mfccs_from_log_mel_spectrograms', 'tf.math.greater_equal', 'tf.signal.inverse_stft_window_fn', 'tf.OptionalSpec']
# USED_APIS = ['tf.keras.layers.Dense', 'tf.keras.initializers.HeNormal', 'tf.keras.layers.BatchNormalization', 'tf.keras.layers.LayerNormalization', 'tf.keras.layers.Activation', 'tf.keras.constraints.min_max_norm', 'tf.GradientTape']

class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(8, kernel_initializer=tf.keras.initializers.HeNormal())
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.activation1 = tf.keras.layers.Activation('relu')
        self.dense2 = tf.keras.layers.Dense(8, kernel_constraint=tf.keras.constraints.min_max_norm(min_value=-0.5, max_value=0.5))
        self.activation2 = tf.keras.layers.Activation('sigmoid')

    def call(self, x, training=False):
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = self.dense1(x)
            z = self.batch_norm(y, training=training)
            w = self.layer_norm(z)
            u = self.activation1(w)
            v = self.dense2(u)
            t = self.activation2(v)
        g = tape.gradient(t, x)
        return t + g

USED_APIS = ["tf.keras.layers.Dense", "tf.keras.initializers.HeNormal",
             "tf.keras.layers.BatchNormalization", "tf.keras.layers.LayerNormalization",
             "tf.keras.layers.Activation", "tf.keras.constraints.min_max_norm",
             "tf.GradientTape"]
