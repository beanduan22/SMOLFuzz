# SMOLFuzz TF model 383
# APIs: ['tf.math.special.fresnel_cos', 'tf.keras.applications.ResNet50V2', 'tf.keras.utils.SequenceEnqueuer', 'tf.keras.layers.Conv3DTranspose', 'tf.keras.callbacks.ProgbarLogger', 'tf.keras.losses.MSE', 'tf.keras.layers.experimental.preprocessing.RandomZoom', 'tf.math.ndtri', 'tf.experimental.numpy.trace', 'tf.keras.layers.Resizing', 'tf.experimental.numpy.complex128', 'tf.experimental.numpy.eye']

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.dense1 = tf.keras.layers.Dense(8)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(4)
        self.dense3 = tf.keras.layers.Dense(1)

    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.batch_norm(x, training=training)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        x = self.dense3(x)
        return tf.math.ndtri(x)
