# SMOLFuzz TF model 82 | attempts=2
# APIS_SELECTED = ['tf.optimizers.Optimizer', 'tf.keras.experimental.CosineDecayRestarts', 'tf.keras.metrics.sparse_top_k_categorical_accuracy', 'tf.keras.initializers.lecun_normal', 'tf.keras.losses.MAPE', 'tf.keras.layers.MaxPooling3D', 'tf.metrics.SparseCategoricalAccuracy', 'tf.keras.layers.Layer', 'tf.metrics.MeanSquaredLogarithmicError', 'tf.metrics.Hinge', 'tf.keras.layers.experimental.SyncBatchNormalization', 'tf.keras.layers.experimental.preprocessing.RandomZoom', 'tf.split', 'tf.constant', 'tf.sinh', 'tf.math.polygamma', 'tf.signal.inverse_mdct', 'tf.meshgrid', 'tf.signal.frame', 'tf.math.reduce_variance']
# USED_APIS = ['tf.keras.layers.Dense', 'tf.keras.initializers.lecun_normal', 'tf.keras.layers.BatchNormalization', 'tf.keras.layers.LayerNormalization', 'tf.keras.layers.Activation', 'tf.sinh', 'tf.GradientTape', 'tf.device']

class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(8, kernel_initializer=tf.keras.initializers.lecun_normal())
        self.bn = tf.keras.layers.BatchNormalization()
        self.dense2 = tf.keras.layers.Dense(8)
        self.ln = tf.keras.layers.LayerNormalization()
        self.dense3 = tf.keras.layers.Dense(8)
        self.act = tf.keras.layers.Activation(tf.sinh)

    def call(self, x, training=False):
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = self.dense1(x)
            y = self.bn(y, training=training)
            y = self.dense2(y)
            y = self.ln(y)
            y = self.dense3(y)
            z = self.act(y)
        g = tape.gradient(z, x)
        with tf.device('/CPU:0'):
            h = z + g
        return h

USED_APIS = ["tf.keras.layers.Dense", "tf.keras.initializers.lecun_normal",
             "tf.keras.layers.BatchNormalization", "tf.keras.layers.LayerNormalization", 
             "tf.keras.layers.Activation", "tf.sinh", "tf.GradientTape", "tf.device"]
