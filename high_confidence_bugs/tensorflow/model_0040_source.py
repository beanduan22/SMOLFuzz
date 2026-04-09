# SMOLFuzz TF model 40
# APIs: ['tf.keras.initializers.LecunUniform', 'tf.nn.elu', 'tf.keras.layers.MaxPool2D', 'tf.keras.experimental.SequenceFeatures', 'tf.keras.metrics.TruePositives', 'tf.math.atanh', 'tf.keras.activations.get', 'tf.keras.metrics.KLD', 'tf.keras.layers.Masking', 'tf.linalg.normalize', 'tf.experimental.numpy.einsum', 'tf.keras.applications.xception.Xception']

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.dense1 = tf.keras.layers.Dense(16, input_dim=8, kernel_initializer=tf.keras.initializers.LecunUniform())
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.dense2 = tf.keras.layers.Dense(1, input_dim=16)

    def call(self, x, training=False):
        x = tf.nn.elu(self.dense1(x))
        x = self.batch_norm(x)
        x = self.dropout(x, training=training)
        x = self.layer_norm(x)
        return self.dense2(x)
