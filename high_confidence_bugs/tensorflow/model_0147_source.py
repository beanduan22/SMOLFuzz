# SMOLFuzz TF model 147
# APIs: ['tf.keras.layers.RandomContrast', 'tf.keras.losses.mean_absolute_percentage_error', 'tf.experimental.numpy.repeat', 'tf.nn.conv1d_transpose', 'tf.math.greater', 'tf.nn.swish', 'tf.keras.initializers.LecunNormal', 'tf.experimental.numpy.logspace', 'tf.experimental.numpy.random.seed', 'tf.keras.metrics.TrueNegatives', 'tf.experimental.numpy.deg2rad', 'tf.linalg.matrix_rank']

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.dense1 = tf.keras.layers.Dense(8, activation=tf.nn.swish, kernel_initializer=tf.keras.initializers.LecunNormal())
        self.bn = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(4, activation='relu', kernel_initializer=tf.keras.initializers.LecunNormal())
        self.dense3 = tf.keras.layers.Dense(1)

    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.bn(x, training=training)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        x = self.dense3(x)
        return tf.reduce_sum(x)
