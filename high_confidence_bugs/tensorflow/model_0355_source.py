# SMOLFuzz TF model 355
# APIs: ['tf.experimental.numpy.triu', 'tf.keras.utils.unpack_x_y_sample_weight', 'tf.keras.activations.swish', 'tf.keras.applications.Xception', 'tf.keras.layers.SeparableConv1D', 'tf.math.real', 'tf.nn.all_candidate_sampler', 'tf.keras.layers.SimpleRNN', 'tf.keras.losses.CategoricalCrossentropy', 'tf.experimental.numpy.zeros_like', 'tf.math.softsign', 'tf.keras.losses.mse']

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.dense1 = tf.keras.layers.Dense(8)
        self.bn = tf.keras.layers.BatchNormalization()
        self.ln = tf.keras.layers.LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(1)

    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.bn(x, training=training)
        x = tf.math.softsign(x)
        x = self.ln(x)
        x = self.dropout(x, training=training)
        return self.dense2(x)
