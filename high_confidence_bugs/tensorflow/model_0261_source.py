# SMOLFuzz TF model 261
# APIs: ['tf.keras.layers.SpatialDropout3D', 'tf.keras.losses.sparse_categorical_crossentropy', 'tf.keras.layers.subtract', 'tf.keras.layers.experimental.preprocessing.RandomRotation', 'tf.math.reduce_sum', 'tf.linalg.svd', 'tf.keras.layers.UpSampling2D', 'tf.keras.metrics.FalsePositives', 'tf.experimental.async_clear_error', 'tf.keras.metrics.SquaredHinge', 'tf.keras.models.model_from_config', 'tf.experimental.numpy.logical_xor']

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.dense1 = tf.keras.layers.Dense(8)
        self.bn = tf.keras.layers.BatchNormalization()
        self.ln = tf.keras.layers.LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(1)

    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.bn(x, training=training)
        x = self.ln(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        return tf.math.reduce_sum(x)
