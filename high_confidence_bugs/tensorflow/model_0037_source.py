# SMOLFuzz TF model 37
# APIs: ['tf.math.imag', 'tf.keras.losses.cosine_similarity', 'tf.keras.metrics.CategoricalHinge', 'tf.keras.metrics.CategoricalAccuracy', 'tf.keras.losses.CategoricalHinge', 'tf.experimental.numpy.random.poisson', 'tf.keras.losses.Huber', 'tf.keras.metrics.mse', 'tf.keras.metrics.FalsePositives', 'tf.experimental.numpy.isrealobj', 'tf.experimental.numpy.abs', 'tf.keras.applications.efficientnet.EfficientNetB1']

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.dense1 = tf.keras.layers.Dense(16, activation='relu', input_dim=8)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.dense2 = tf.keras.layers.Dense(8, activation='tanh')
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense3 = tf.keras.layers.Dense(1, activation='linear')

    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.batch_norm(x)
        x = self.dense2(x)
        x = self.layer_norm(x)
        x = self.dropout(x, training=training)
        return self.dense3(x)
