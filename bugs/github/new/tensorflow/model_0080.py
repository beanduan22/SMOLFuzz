# SMOLFuzz TF model 80 | attempts=2
# APIS_SELECTED = ['tf.keras.optimizers.schedules.PolynomialDecay', 'tf.keras.models.Sequential', 'tf.keras.activations.selu', 'tf.losses.huber', 'tf.metrics.hinge', 'tf.losses.mean_absolute_percentage_error', 'tf.keras.layers.AbstractRNNCell', 'tf.nn.all_candidate_sampler', 'tf.keras.metrics.sparse_top_k_categorical_accuracy', 'tf.nn.zero_fraction', 'tf.keras.metrics.log_cosh', 'tf.less', 'tf.unique_with_counts', 'tf.py_function', 'tf.bitwise.left_shift', 'tf.math.greater_equal', 'tf.divide', 'tf.truncatemod', 'tf.math.add_n', 'tf.math.mod']
# USED_APIS = ['tf.keras.layers.Dense', 'tf.keras.layers.BatchNormalization', 'tf.keras.layers.LayerNormalization', 'tf.keras.activations.selu', 'tf.GradientTape', 'tf.device']

class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(8)
        self.bn = tf.keras.layers.BatchNormalization()
        self.ln = tf.keras.layers.LayerNormalization()
        self.dense2 = tf.keras.layers.Dense(4, activation=tf.keras.activations.selu)
        self.dense3 = tf.keras.layers.Dense(4)

    def call(self, x, training=False):
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = self.dense1(x)
            y = self.bn(y, training=training)
            y = self.ln(y)
            z = self.dense2(y)
        g = tape.gradient(z, x)
        z = z + g[:, :4]
        with tf.device('/CPU:0'):
            output = self.dense3(z)
        return output

USED_APIS = ["tf.keras.layers.Dense", "tf.keras.layers.BatchNormalization",
             "tf.keras.layers.LayerNormalization", "tf.keras.activations.selu",
             "tf.GradientTape", "tf.device"]
