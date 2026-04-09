# SMOLFuzz TF model 202
# APIs: ['tf.keras.applications.DenseNet169', 'tf.signal.ifft', 'tf.experimental.dispatch_for_unary_elementwise_apis', 'tf.experimental.numpy.max', 'tf.linalg.LinearOperatorDiag', 'tf.keras.utils.timeseries_dataset_from_array', 'tf.experimental.numpy.flip', 'tf.linalg.LinearOperatorHouseholder', 'tf.keras.layers.experimental.preprocessing.RandomContrast', 'tf.keras.layers.Concatenate', 'tf.keras.layers.Conv3DTranspose', 'tf.math.add']

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.dense1 = tf.keras.layers.Dense(16, input_dim=8)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.concat = tf.keras.layers.Concatenate()
        self.dense2 = tf.keras.layers.Dense(1)

    def call(self, x, training=False):
        x = tf.math.add(x, 0.5)
        x = self.dense1(x)
        x = self.batch_norm(x, training=training)
        x = self.dropout(x, training=training)
        x = self.layer_norm(x)
        x = self.concat([x[:, :4], x[:, 4:]])
        return tf.math.reduce_sum(self.dense2(x))
