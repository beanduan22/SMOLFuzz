# SMOLFuzz TF model 51
# APIs: ['tf.experimental.numpy.uint64', 'tf.keras.losses.BinaryCrossentropy', 'tf.math.real', 'tf.acosh', 'tf.linalg.lu_solve', 'tf.experimental.numpy.divide', 'tf.math.reduce_max', 'tf.experimental.numpy.signbit', 'tf.experimental.numpy.not_equal', 'tf.math.reduce_prod', 'tf.keras.utils.GeneratorEnqueuer', 'tf.keras.Model']

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.dense1 = tf.keras.layers.Dense(8)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(1)

    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.batch_norm(x, training=training)
        x = self.layer_norm(x)
        x = tf.math.acosh(tf.experimental.numpy.divide(x, tf.experimental.numpy.signbit(x) + 1))
        x = self.dropout(x, training=training)
        return self.dense2(x)
