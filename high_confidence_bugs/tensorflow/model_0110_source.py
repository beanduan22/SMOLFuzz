# SMOLFuzz TF model 110
# APIs: ['tf.experimental.numpy.select', 'tf.experimental.numpy.flipud', 'tf.experimental.numpy.outer', 'tf.math.scalar_mul', 'tf.keras.layers.Add', 'tf.keras.losses.mean_squared_logarithmic_error', 'tf.experimental.async_clear_error', 'tf.experimental.numpy.inner', 'tf.keras.applications.inception_resnet_v2.decode_predictions', 'tf.keras.estimator.model_to_estimator', 'tf.keras.datasets.boston_housing.load_data', 'tf.keras.layers.Reshape']

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.dense1 = tf.keras.layers.Dense(16, input_shape=(8,))
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.add = tf.keras.layers.Add()
        self.reshape = tf.keras.layers.Reshape((4, 4))
        self.dense2 = tf.keras.layers.Dense(1)

    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.batch_norm(x)
        x1 = tf.experimental.numpy.select([x > 0], [tf.math.scalar_mul(2.0, x)])
        x2 = tf.math.scalar_mul(-1.0, x)
        x = self.add([x1, x2])
        x = self.dropout(x, training=training)
        x = self.reshape(x)
        x = tf.experimental.numpy.outer(tf.reduce_mean(x, axis=1), tf.reduce_mean(x, axis=1))
        x = tf.reduce_sum(x)
        return x
