# SMOLFuzz TF model 157 | attempts=1
# APIS_SELECTED = ['tf.optimizers.schedules.CosineDecay', 'tf.keras.layers.Convolution1DTranspose', 'tf.losses.mape', 'tf.metrics.PrecisionAtRecall', 'tf.losses.cosine_similarity', 'tf.losses.Hinge', 'tf.losses.categorical_hinge', 'tf.losses.MSE', 'tf.losses.mean_squared_logarithmic_error', 'tf.keras.metrics.SparseTopKCategoricalAccuracy', 'tf.shape_n', 'tf.Variable.SaveSliceInfo', 'tf.linalg.sqrtm', 'tf.math.less_equal', 'tf.math.log_softmax', 'tf.floor', 'tf.linalg.tensor_diag_part', 'tf.math.angle', 'tf.math.cos', 'tf.init_scope']
# USED_APIS = ['tf.keras.layers.Dense', 'tf.keras.layers.BatchNormalization', 'tf.keras.layers.LayerNormalization', 'tf.GradientTape', 'tf.keras.layers.Convolution1DTranspose', 'tf.math.cos', 'tf.device']

class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(8)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.dense2 = tf.keras.layers.Dense(8)
        self.conv_transpose = tf.keras.layers.Convolution1DTranspose(filters=8, kernel_size=3, padding='same')
        self.dense3 = tf.keras.layers.Dense(8)
        self.act = tf.keras.layers.Activation(tf.math.cos)

    def call(self, x, training=False):
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = self.dense1(x)
            y = self.batch_norm(y, training=training)
            y = self.layer_norm(y)
            z = self.dense2(y)
            z = self.conv_transpose(tf.expand_dims(z, 1))
            z = tf.squeeze(z, 1)
            w = self.dense3(z)
        g = tape.gradient(w, x)
        h = w + g
        with tf.device('/CPU:0'):
            i = self.act(h)
        return i

USED_APIS = ["tf.keras.layers.Dense", "tf.keras.layers.BatchNormalization",
             "tf.keras.layers.LayerNormalization", "tf.GradientTape",
             "tf.keras.layers.Convolution1DTranspose", "tf.math.cos", "tf.device"]
