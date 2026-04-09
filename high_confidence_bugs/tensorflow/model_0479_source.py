# SMOLFuzz TF model 479
# APIs: ['tf.keras.applications.densenet.decode_predictions', 'tf.keras.layers.experimental.preprocessing.RandomZoom', 'tf.experimental.numpy.concatenate', 'tf.keras.layers.Cropping2D', 'tf.keras.metrics.RecallAtPrecision', 'tf.math.tan', 'tf.keras.metrics.MeanSquaredError', 'tf.sort', 'tf.math.special.fresnel_sin', 'tf.keras.utils.set_random_seed', 'tf.experimental.numpy.compress', 'tf.experimental.numpy.hsplit']

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.dense1 = tf.keras.layers.Dense(8)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(1)

    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.batch_norm(x, training=training)
        x = self.layer_norm(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        return tf.math.tan(x)
