# SMOLFuzz TF model 478
# APIs: ['tf.nn.nce_loss', 'tf.keras.initializers.TruncatedNormal', 'tf.signal.mdct', 'tf.keras.metrics.mean_squared_error', 'tf.math.l2_normalize', 'tf.keras.layers.AveragePooling3D', 'tf.keras.losses.MeanSquaredError', 'tf.experimental.numpy.count_nonzero', 'tf.experimental.numpy.empty', 'tf.keras.optimizers.schedules.PiecewiseConstantDecay', 'tf.keras.layers.RandomTranslation', 'tf.keras.optimizers.schedules.CosineDecayRestarts']

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.dense1 = tf.keras.layers.Dense(16, input_shape=(8,), kernel_initializer=tf.keras.initializers.TruncatedNormal())
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.dense2 = tf.keras.layers.Dense(1, input_dim=16)

    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.batch_norm(x)
        x = self.dropout(x, training=training)
        x = self.layer_norm(x)
        x = self.dense2(x)
        return tf.math.reduce_sum(x)
