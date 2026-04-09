# SMOLFuzz TF model 246
# APIs: ['tf.keras.constraints.max_norm', 'tf.keras.metrics.poisson', 'tf.experimental.numpy.arcsinh', 'tf.experimental.numpy.fabs', 'tf.experimental.numpy.sinc', 'tf.keras.preprocessing.image_dataset_from_directory', 'tf.linalg.triangular_solve', 'tf.math.bessel_i0', 'tf.keras.applications.mobilenet_v2.MobileNetV2', 'tf.keras.layers.LocallyConnected2D', 'tf.keras.layers.ZeroPadding2D', 'tf.experimental.numpy.zeros_like']

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.dense1 = tf.keras.layers.Dense(16, input_dim=8)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.dense2 = tf.keras.layers.Dense(1, kernel_constraint=tf.keras.constraints.max_norm(max_value=1))

    def call(self, x, training=False):
        x = self.dense1(x)
        x = tf.experimental.numpy.arcsinh(tf.experimental.numpy.fabs(x))
        x = self.batch_norm(x, training=training)
        x = self.layer_norm(x)
        return self.dense2(x)
