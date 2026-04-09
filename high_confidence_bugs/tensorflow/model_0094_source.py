# SMOLFuzz TF model 94
# APIs: ['tf.keras.metrics.squared_hinge', 'tf.nn.max_pool2d', 'tf.square', 'tf.keras.layers.experimental.preprocessing.Discretization', 'tf.keras.losses.CosineSimilarity', 'tf.experimental.numpy.compress', 'tf.keras.optimizers.schedules.CosineDecayRestarts', 'tf.keras.layers.experimental.preprocessing.RandomFlip', 'tf.keras.applications.MobileNetV3Large', 'tf.math.special.bessel_k0e', 'tf.keras.activations.softsign', 'tf.sin']

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.dense1 = tf.keras.layers.Dense(16, input_dim=8)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.dense2 = tf.keras.layers.Dense(32)

    def call(self, x, training=False):
        x = self.dense1(x)
        x = tf.sin(tf.square(x))
        x = self.batch_norm(x)
        x = self.dropout(x, training=training)
        x = self.layer_norm(x)
        return tf.reduce_sum(self.dense2(x))
