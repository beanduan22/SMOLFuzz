# SMOLFuzz TF model 225
# APIs: ['tf.keras.layers.Cropping3D', 'tf.keras.layers.CategoryEncoding', 'tf.experimental.numpy.tril', 'tf.experimental.numpy.ndim', 'tf.experimental.numpy.fabs', 'tf.keras.applications.resnet.ResNet152', 'tf.keras.losses.CategoricalCrossentropy', 'tf.math.betainc', 'tf.experimental.numpy.logaddexp2', 'tf.keras.losses.cosine_similarity', 'tf.keras.metrics.CategoricalHinge', 'tf.keras.preprocessing.sequence.skipgrams']

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.dense1 = tf.keras.layers.Dense(8)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(1)

    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.batch_norm(x, training=training)
        x = self.layer_norm(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        return tf.experimental.numpy.fabs(x)
