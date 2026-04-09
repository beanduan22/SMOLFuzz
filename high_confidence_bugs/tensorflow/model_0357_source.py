# SMOLFuzz TF model 357
# APIs: ['tf.math.atan2', 'tf.experimental.numpy.compress', 'tf.math.expm1', 'tf.keras.metrics.msle', 'tf.keras.applications.MobileNet', 'tf.keras.backend.set_floatx', 'tf.math.squared_difference', 'tf.experimental.numpy.float_', 'tf.math.equal', 'tf.math.reduce_logsumexp', 'tf.keras.constraints.max_norm', 'tf.nn.all_candidate_sampler']

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.dense1 = tf.keras.layers.Dense(8, input_dim=8, kernel_constraint=tf.keras.constraints.max_norm(max_value=2.0))
        self.bn = tf.keras.layers.BatchNormalization()
        self.ln = tf.keras.layers.LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(rate=0.5)
        self.dense2 = tf.keras.layers.Dense(1, input_dim=8)

    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.bn(x, training=training)
        x = self.ln(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        return tf.reduce_logsumexp(tf.math.expm1(x), axis=-1)
