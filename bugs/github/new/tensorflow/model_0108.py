# SMOLFuzz TF model 108 | attempts=3
# APIS_SELECTED = ['tf.keras.optimizers.Adagrad', 'tf.nn.gelu', 'tf.keras.layers.Discretization', 'tf.metrics.sparse_categorical_crossentropy', 'tf.losses.msle', 'tf.losses.mae', 'tf.keras.losses.cosine_similarity', 'tf.losses.huber', 'tf.keras.layers.experimental.preprocessing.RandomContrast', 'tf.keras.metrics.AUC', 'tf.keras.layers.MaxPooling3D', 'tf.scatter_nd', 'tf.type_spec_from_value', 'tf.signal.ifft', 'tf.linalg.eigh', 'tf.math.rint', 'tf.math.maximum', 'tf.linalg.lstsq', 'tf.math.logical_not', 'tf.linalg.tridiagonal_matmul']
# USED_APIS = ['tf.keras.layers.Dense', 'tf.keras.layers.BatchNormalization', 'tf.keras.layers.LayerNormalization', 'tf.nn.gelu', 'tf.GradientTape', 'tf.device', 'tf.math.maximum']

class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(8)
        self.bn = tf.keras.layers.BatchNormalization()
        self.ln = tf.keras.layers.LayerNormalization()
        self.dense2 = tf.keras.layers.Dense(8)  # Changed units to 8
        self.activation = tf.keras.layers.Activation(tf.nn.gelu)
        self.dense3 = tf.keras.layers.Dense(8)  # Changed units to 8

    def call(self, x, training=False):
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = self.dense1(x)
            y = self.bn(y, training=training)
            y = self.ln(y)
            y = self.dense2(y)  # Output shape is now [4, 8]
            y = self.activation(y)
            z = self.dense3(y)
        g = tape.gradient(z, x)
        
        with tf.device('/CPU:0'):
            h = z + g
            i = tf.math.maximum(h, 0.0)
        
        return i

USED_APIS = ["tf.keras.layers.Dense", "tf.keras.layers.BatchNormalization",
             "tf.keras.layers.LayerNormalization", "tf.nn.gelu",
             "tf.GradientTape", "tf.device", "tf.math.maximum"]
