# SMOLFuzz TF model 36 | attempts=4
# APIS_SELECTED = ['tf.keras.optimizers.SGD', 'tf.keras.initializers.RandomUniform', 'tf.keras.layers.Hashing', 'tf.losses.poisson', 'tf.metrics.mean_squared_error', 'tf.keras.layers.GlobalMaxPooling1D', 'tf.metrics.Precision', 'tf.losses.cosine_similarity', 'tf.keras.constraints.unit_norm', 'tf.nn.swish', 'tf.keras.metrics.Sum', 'tf.split', 'tf.py_function', 'tf.math.squared_difference', 'tf.maximum', 'tf.math.lgamma', 'tf.math.polygamma', 'tf.reduce_min', 'tf.linalg.LinearOperatorKronecker', 'tf.RegisterGradient']
# USED_APIS = ['tf.keras.layers.Dense', 'tf.keras.initializers.RandomUniform', 'tf.keras.layers.Hashing', 'tf.keras.constraints.unit_norm', 'tf.nn.swish', 'tf.GradientTape', 'tf.metrics.Sum', 'tf.maximum', 'tf.device', 'tf.zeros_like']

class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(8, kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1))
        self.hashing = tf.keras.layers.Hashing(num_bins=8, output_mode='int', sparse=False)
        self.dense2 = tf.keras.layers.Dense(8, kernel_constraint=tf.keras.constraints.unit_norm())
        self.activation = tf.keras.layers.Activation(tf.nn.swish)
        self.sum_metric = tf.metrics.Sum()

    def call(self, x, training=False):
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = self.dense1(x)
            y = self.hashing(y)
            z = self.dense2(y)
            z = self.activation(z)
        
        g = tape.gradient(z, x)
        if not training and g is not None:
            self.sum_metric.update_state(g)
        
        with tf.device('/CPU:0'):
            result = tf.maximum(z, tf.zeros_like(z) if g is None else g)
        
        return result

USED_APIS = ["tf.keras.layers.Dense", "tf.keras.initializers.RandomUniform", "tf.keras.layers.Hashing", 
             "tf.keras.constraints.unit_norm", "tf.nn.swish", "tf.GradientTape", 
             "tf.metrics.Sum", "tf.maximum", "tf.device", "tf.zeros_like"]
