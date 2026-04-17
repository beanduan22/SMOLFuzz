# SMOLFuzz TF model 149 | attempts=1
# APIS_SELECTED = ['tf.keras.optimizers.Nadam', 'tf.keras.experimental.SequenceFeatures', 'tf.keras.layers.experimental.preprocessing.TextVectorization', 'tf.metrics.RootMeanSquaredError', 'tf.keras.layers.AvgPool3D', 'tf.keras.layers.RandomContrast', 'tf.keras.metrics.binary_crossentropy', 'tf.initializers.RandomNormal', 'tf.keras.layers.Concatenate', 'tf.keras.layers.SeparableConvolution1D', 'tf.keras.metrics.CategoricalHinge', 'tf.shape', 'tf.Variable.SaveSliceInfo', 'tf.add', 'tf.acosh', 'tf.math.segment_mean', 'tf.eigvals', 'tf.signal.linear_to_mel_weight_matrix', 'tf.linalg.cross', 'tf.signal.rfft2d']
# USED_APIS = ['tf.keras.layers.Dense', 'tf.initializers.RandomNormal', 'tf.keras.layers.BatchNormalization', 'tf.keras.layers.LayerNormalization', 'tf.keras.layers.Concatenate', 'tf.keras.layers.Activation', 'tf.keras.optimizers.Nadam', 'tf.metrics.RootMeanSquaredError', 'tf.GradientTape', 'tf.device']

class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(8, kernel_initializer=tf.initializers.RandomNormal())
        self.bn = tf.keras.layers.BatchNormalization()
        self.ln = tf.keras.layers.LayerNormalization()
        self.dense2 = tf.keras.layers.Dense(4)
        self.concat = tf.keras.layers.Concatenate()
        self.dense3 = tf.keras.layers.Dense(8)
        self.activation = tf.keras.layers.Activation('relu')
        self.optimizer = tf.keras.optimizers.Nadam()
        self.rmse = tf.metrics.RootMeanSquaredError()

    def call(self, x, training=False):
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = self.dense1(x)
            y = self.bn(y, training=training)
            y = self.ln(y)
            z = self.dense2(y)
            z = self.concat([y, z])
            z = self.dense3(z)
            z = self.activation(z)
        g = tape.gradient(z, x)
        with tf.device('/CPU:0'):
            result = z + g
        return result

USED_APIS = ["tf.keras.layers.Dense", "tf.initializers.RandomNormal",
             "tf.keras.layers.BatchNormalization", "tf.keras.layers.LayerNormalization",
             "tf.keras.layers.Concatenate", "tf.keras.layers.Activation",
             "tf.keras.optimizers.Nadam", "tf.metrics.RootMeanSquaredError",
             "tf.GradientTape", "tf.device"]
