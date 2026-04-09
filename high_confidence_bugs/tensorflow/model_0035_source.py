# SMOLFuzz TF model 35
# APIs: ['tf.keras.metrics.CategoricalCrossentropy', 'tf.experimental.dlpack.from_dlpack', 'tf.keras.layers.ELU', 'tf.keras.layers.DepthwiseConv1D', 'tf.keras.applications.mobilenet_v2.MobileNetV2', 'tf.math.greater_equal', 'tf.linalg.cross', 'tf.nn.bias_add', 'tf.math.is_strictly_increasing', 'tf.math.special.spence', 'tf.experimental.numpy.atleast_3d', 'tf.linalg.LinearOperatorCirculant2D']

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.dense1 = tf.keras.layers.Dense(8)
        self.elu = tf.keras.layers.ELU()
        self.dense2 = tf.keras.layers.Dense(8)
        self.batchnorm = tf.keras.layers.BatchNormalization()
        self.dense3 = tf.keras.layers.Dense(1)

    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.elu(x)
        x = self.dense2(x)
        x = self.batchnorm(x, training=training)
        return self.dense3(x)
