# SMOLFuzz TF model 78 | attempts=3
# APIS_SELECTED = ['tf.keras.optimizers.Adamax', 'tf.keras.experimental.CosineDecayRestarts', 'tf.keras.losses.mean_absolute_percentage_error', 'tf.keras.layers.experimental.preprocessing.RandomFlip', 'tf.metrics.SensitivityAtSpecificity', 'tf.initializers.GlorotNormal', 'tf.keras.layers.Discretization', 'tf.initializers.lecun_uniform', 'tf.metrics.CosineSimilarity', 'tf.keras.layers.GlobalAvgPool1D', 'tf.keras.metrics.kld', 'tf.greater_equal', 'tf.unique_with_counts', 'tf.TensorSpec', 'tf.round', 'tf.signal.vorbis_window', 'tf.reduce_prod', 'tf.linalg.LinearOperatorAdjoint', 'tf.math.reciprocal', 'tf.signal.hann_window']
# USED_APIS = ['tf.keras.layers.Dense', 'tf.initializers.GlorotNormal', 'tf.keras.layers.BatchNormalization', 'tf.keras.layers.LayerNormalization', 'tf.keras.layers.Activation', 'tf.keras.layers.GlobalAvgPool1D', 'tf.metrics.CosineSimilarity', 'tf.GradientTape', 'tf.reduce_prod']

class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(8, kernel_initializer=tf.initializers.GlorotNormal())
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.dense2 = tf.keras.layers.Dense(4, kernel_initializer=tf.initializers.lecun_uniform())  # Adjusted units to match output shape
        self.activation = tf.keras.layers.Activation('relu')
        self.global_avg_pool = tf.keras.layers.GlobalAvgPool1D(input_shape=(None, 8))
        self.cosine_similarity = tf.metrics.CosineSimilarity()
        self.sensitivity_at_specificity = tf.metrics.SensitivityAtSpecificity(0.5)

    def call(self, x, training=False):
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = self.dense1(x)
            y = self.batch_norm(y, training=training)
            y = self.layer_norm(y)
            y = self.dense2(y)
            y = self.activation(y)
            y = self.global_avg_pool(tf.expand_dims(y, axis=1))
        grad = tape.gradient(y, x)
        y = y + tf.reduce_prod(grad, axis=1, keepdims=True)

        return y

USED_APIS = ["tf.keras.layers.Dense", "tf.initializers.GlorotNormal", "tf.keras.layers.BatchNormalization",
             "tf.keras.layers.LayerNormalization", "tf.keras.layers.Activation", "tf.keras.layers.GlobalAvgPool1D",
             "tf.metrics.CosineSimilarity", "tf.GradientTape", "tf.reduce_prod"]
