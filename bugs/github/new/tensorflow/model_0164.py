# SMOLFuzz TF model 164 | attempts=2
# APIS_SELECTED = ['tf.optimizers.get', 'tf.keras.experimental.SequenceFeatures', 'tf.keras.metrics.CategoricalAccuracy', 'tf.nn.tanh', 'tf.nn.safe_embedding_lookup_sparse', 'tf.losses.mean_squared_error', 'tf.losses.kl_divergence', 'tf.initializers.Identity', 'tf.losses.mse', 'tf.metrics.poisson', 'tf.keras.activations.exponential', 'tf.shape', 'tf.RaggedTensorSpec', 'tf.linalg.lu', 'tf.math.squared_difference', 'tf.signal.hamming_window', 'tf.math.segment_prod', 'tf.math.sqrt', 'tf.linalg.slogdet', 'tf.math.angle']
# USED_APIS = ['tf.keras.layers.Dense', 'tf.initializers.Identity', 'tf.keras.layers.BatchNormalization', 'tf.keras.layers.LayerNormalization', 'tf.keras.activations.exponential', 'tf.GradientTape', 'tf.math.squared_difference', 'tf.nn.tanh', 'tf.math.sqrt', 'tf.device']

class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(8, kernel_initializer=tf.initializers.Identity())
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.dense2 = tf.keras.layers.Dense(8)
        self.activation = tf.keras.activations.exponential
        self.metric = tf.keras.metrics.CategoricalAccuracy()

    def call(self, x, training=False):
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = self.dense1(x)
            y = self.batch_norm(y, training=training)
            y = self.layer_norm(y)
            z = self.dense2(y)
            z = self.activation(z)
        
        g = tape.gradient(z, x)
        y = tf.math.squared_difference(z, g)
        with tf.device('/CPU:0'):
            y = tf.nn.tanh(y)
            y = tf.math.sqrt(y)
        
        return y

USED_APIS = ["tf.keras.layers.Dense", "tf.initializers.Identity",
             "tf.keras.layers.BatchNormalization", "tf.keras.layers.LayerNormalization",
             "tf.keras.activations.exponential", "tf.GradientTape", "tf.math.squared_difference",
             "tf.nn.tanh", "tf.math.sqrt", "tf.device"]
