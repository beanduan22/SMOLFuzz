# SMOLFuzz TF model 53
# APIs: ['tf.keras.applications.densenet.preprocess_input', 'tf.experimental.numpy.atleast_2d', 'tf.atanh', 'tf.experimental.numpy.minimum', 'tf.keras.applications.ResNet152', 'tf.keras.applications.densenet.DenseNet201', 'tf.maximum', 'tf.keras.optimizers.Adam', 'tf.keras.losses.Poisson', 'tf.experimental.numpy.flip', 'tf.math.exp', 'tf.nn.RNNCellDropoutWrapper']

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.dense1 = tf.keras.layers.Dense(8)
        self.bn = tf.keras.layers.BatchNormalization()
        self.dense2 = tf.keras.layers.Dense(8)
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense3 = tf.keras.layers.Dense(1)

    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.bn(x, training=training)
        x = tf.atanh(tf.maximum(x, -0.99))
        x = self.dense2(x)
        x = self.dropout(x, training=training)
        x = self.dense3(x)
        return x
