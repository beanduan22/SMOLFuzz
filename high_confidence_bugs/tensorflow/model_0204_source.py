# SMOLFuzz TF model 204
# APIs: ['tf.keras.applications.densenet.DenseNet201', 'tf.nn.conv1d', 'tf.keras.metrics.MAE', 'tf.math.is_non_decreasing', 'tf.math.logical_not', 'tf.experimental.numpy.deg2rad', 'tf.keras.regularizers.deserialize', 'tf.keras.layers.Minimum', 'tf.keras.optimizers.schedules.CosineDecay', 'tf.experimental.numpy.random.standard_normal', 'tf.exp', 'tf.math.erf']

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.dense1 = tf.keras.layers.Dense(16, input_dim=8)
        self.bn = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.dense2 = tf.keras.layers.Dense(1)

    def call(self, x, training=False):
        x = tf.exp(tf.math.erf(x))
        x = self.dense1(x)
        x = self.bn(x, training=training)
        x = self.dropout(x, training=training)
        return tf.squeeze(self.dense2(x))
