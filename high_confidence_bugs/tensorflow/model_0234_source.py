# SMOLFuzz TF model 234
# APIs: ['tf.keras.initializers.GlorotUniform', 'tf.keras.optimizers.schedules.CosineDecayRestarts', 'tf.keras.losses.MeanAbsolutePercentageError', 'tf.math.reduce_sum', 'tf.keras.layers.GlobalAveragePooling2D', 'tf.keras.metrics.MeanSquaredLogarithmicError', 'tf.keras.layers.experimental.SyncBatchNormalization', 'tf.experimental.numpy.broadcast_to', 'tf.signal.ifftshift', 'tf.keras.layers.experimental.preprocessing.RandomTranslation', 'tf.keras.layers.SeparableConv2D', 'tf.experimental.numpy.argmin']

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.dense1 = tf.keras.layers.Dense(16, input_dim=8)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.dense2 = tf.keras.layers.Dense(8)

    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.batch_norm(x, training=training)
        x = self.dropout(x, training=training)
        x = self.layer_norm(x)
        x = self.dense2(x)
        return tf.math.reduce_sum(x)
