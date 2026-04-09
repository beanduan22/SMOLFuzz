# SMOLFuzz TF model 208
# APIs: ['tf.nn.max_pool2d', 'tf.keras.applications.VGG19', 'tf.keras.layers.experimental.preprocessing.RandomFlip', 'tf.math.segment_mean', 'tf.keras.layers.experimental.preprocessing.Resizing', 'tf.linalg.expm', 'tf.keras.optimizers.Optimizer', 'tf.linalg.banded_triangular_solve', 'tf.nn.conv_transpose', 'tf.keras.initializers.variance_scaling', 'tf.keras.applications.nasnet.NASNetLarge', 'tf.experimental.numpy.int8']

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.dense1 = tf.keras.layers.Dense(16, input_dim=8)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.dense2 = tf.keras.layers.Dense(1, input_dim=16)

    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.batch_norm(x, training=training)
        x = self.dropout(x, training=training)
        x = self.layer_norm(x)
        return self.dense2(x)
