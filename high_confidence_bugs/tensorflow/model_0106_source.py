# SMOLFuzz TF model 106
# APIs: ['tf.math.digamma', 'tf.keras.layers.SeparableConv2D', 'tf.linalg.inv', 'tf.signal.mdct', 'tf.linalg.matmul', 'tf.keras.layers.SeparableConvolution2D', 'tf.nn.conv2d_transpose', 'tf.keras.applications.resnet50.preprocess_input', 'tf.reshape', 'tf.keras.initializers.he_uniform', 'tf.nn.conv1d_transpose', 'tf.keras.losses.CategoricalHinge']

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.dense1 = tf.keras.layers.Dense(16, input_dim=8)
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.dropout1 = tf.keras.layers.Dropout(0.2)
        self.layer_norm1 = tf.keras.layers.LayerNormalization()

    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.batch_norm1(x, training=training)
        x = tf.math.digamma(tf.cast(x, dtype=tf.float64))
        x = tf.reshape(x, [-1])
        return tf.reduce_sum(x)
