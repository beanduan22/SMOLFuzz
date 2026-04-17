# SMOLFuzz TF model 79 | attempts=4
# APIS_SELECTED = ['tf.optimizers.serialize', 'tf.keras.experimental.CosineDecayRestarts', 'tf.keras.layers.GlobalMaxPooling1D', 'tf.keras.metrics.sparse_top_k_categorical_accuracy', 'tf.keras.layers.RandomContrast', 'tf.keras.losses.Poisson', 'tf.metrics.sparse_categorical_accuracy', 'tf.nn.batch_normalization', 'tf.keras.activations.get', 'tf.metrics.deserialize', 'tf.nn.l2_normalize', 'tf.less_equal', 'tf.gather_nd', 'tf.RaggedTensorSpec', 'tf.cosh', 'tf.linalg.cholesky_solve', 'tf.math.special.bessel_k1e', 'tf.linalg.eig', 'tf.pow', 'tf.math.special.fresnel_cos']
# USED_APIS = ['tf.keras.layers.Dense', 'tf.keras.layers.BatchNormalization', 'tf.keras.activations.get', 'tf.keras.layers.LayerNormalization']

class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(8)
        self.bn = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.activations.get('relu')
        self.dense2 = tf.keras.layers.Dense(8)
        self.ln = tf.keras.layers.LayerNormalization()

    def call(self, x, training=False):
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = self.dense1(x)
            y = self.bn(y, training=training)
            y = self.activation(y)
            y = self.dense2(y)
            y = self.ln(y)
        g = tape.gradient(tf.reduce_sum(y), x)
        if g is not None:
            y = y + g
        return y

USED_APIS = ["tf.keras.layers.Dense", "tf.keras.layers.BatchNormalization",
             "tf.keras.activations.get", "tf.keras.layers.LayerNormalization"]
