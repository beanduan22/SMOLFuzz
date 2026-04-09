# SMOLFuzz TF model 265
# APIs: ['tf.experimental.numpy.nonzero', 'tf.keras.layers.RandomFlip', 'tf.linalg.LinearOperatorFullMatrix', 'tf.keras.utils.custom_object_scope', 'tf.experimental.numpy.nansum', 'tf.keras.constraints.max_norm', 'tf.keras.applications.MobileNet', 'tf.keras.optimizers.Adam', 'tf.keras.preprocessing.image.random_channel_shift', 'tf.keras.metrics.TrueNegatives', 'tf.keras.layers.Convolution1DTranspose', 'tf.keras.metrics.serialize']

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.dense1 = tf.keras.layers.Dense(8, input_dim=8, kernel_constraint=tf.keras.constraints.max_norm(max_value=2.0))
        self.bn = tf.keras.layers.BatchNormalization()
        self.ln = tf.keras.layers.LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(1, input_dim=8)

    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.bn(x, training=training)
        x = self.ln(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        return tf.experimental.numpy.nansum(x)
