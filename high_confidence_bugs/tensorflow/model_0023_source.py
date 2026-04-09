# SMOLFuzz TF model 23
# APIs: ['tf.keras.metrics.BinaryAccuracy', 'tf.keras.activations.elu', 'tf.keras.layers.RandomZoom', 'tf.sigmoid', 'tf.experimental.tensorrt.ConversionParams', 'tf.tan', 'tf.experimental.numpy.swapaxes', 'tf.math.lbeta', 'tf.keras.layers.GlobalAvgPool2D', 'tf.signal.frame', 'tf.keras.layers.Conv3DTranspose', 'tf.math.unsorted_segment_max']

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.dense1 = tf.keras.layers.Dense(8, activation=tf.keras.activations.elu)
        self.batchnorm = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(8, activation=tf.tan)
        self.dense3 = tf.keras.layers.Dense(1)

    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.batchnorm(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        x = self.dense3(x)
        return tf.sigmoid(x)
