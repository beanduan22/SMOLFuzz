# SMOLFuzz TF model 4 | attempts=2
# APIS_SELECTED = ['tf.keras.optimizers.Adadelta', 'tf.keras.regularizers.L2', 'tf.keras.metrics.poisson', 'tf.keras.losses.MAE', 'tf.keras.metrics.CategoricalHinge', 'tf.keras.losses.poisson', 'tf.nn.gelu', 'tf.keras.layers.experimental.EinsumDense', 'tf.keras.losses.MSE', 'tf.keras.metrics.BinaryCrossentropy', 'tf.metrics.MeanRelativeError', 'tf.tensor_scatter_nd_min', 'tf.numpy_function', 'tf.signal.ifftshift', 'tf.reduce_logsumexp', 'tf.linalg.svd', 'tf.signal.irfft', 'tf.exp', 'tf.math.special.bessel_i0e', 'tf.TensorShape']
# USED_APIS = ['tf.keras.layers.Dense', 'tf.keras.regularizers.L2', 'tf.keras.layers.BatchNormalization', 'tf.keras.layers.LayerNormalization', 'tf.nn.gelu', 'tf.keras.layers.EinsumDense', 'tf.GradientTape', 'tf.device']

class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(8, kernel_regularizer=tf.keras.regularizers.L2())
        self.bn = tf.keras.layers.BatchNormalization()
        self.ln = tf.keras.layers.LayerNormalization()
        self.activation = tf.keras.layers.Activation(tf.nn.gelu)
        self.einsum_dense = tf.keras.layers.EinsumDense("ij,jk->ik", output_shape=(8,))
        self.dense2 = tf.keras.layers.Dense(4)

    def call(self, x, training=False):
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = self.dense1(x)
            y = self.bn(y, training=training)
            y = self.ln(y)
            y = self.activation(y)
            y = self.einsum_dense(y)
        
        g = tape.gradient(y, x)
        z = y + g

        with tf.device('/CPU:0'):
            z = self.dense2(z)
        
        return z

USED_APIS = ["tf.keras.layers.Dense", "tf.keras.regularizers.L2", "tf.keras.layers.BatchNormalization",
             "tf.keras.layers.LayerNormalization", "tf.nn.gelu", "tf.keras.layers.EinsumDense",
             "tf.GradientTape", "tf.device"]
