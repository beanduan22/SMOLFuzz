# SMOLFuzz TF model 84
# APIs: ['tf.math.floor', 'tf.keras.utils.experimental.DatasetCreator', 'tf.keras.applications.imagenet_utils.preprocess_input', 'tf.math.confusion_matrix', 'tf.experimental.numpy.exp2', 'tf.linalg.norm', 'tf.keras.losses.Huber', 'tf.math.abs', 'tf.keras.layers.AveragePooling2D', 'tf.experimental.numpy.tanh', 'tf.keras.applications.VGG16', 'tf.experimental.numpy.sort']

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
        x = tf.math.abs(x)
        x = self.batch_norm(x, training=training)
        x = self.dropout(x, training=training)
        x = self.layer_norm(x)
        return tf.experimental.numpy.exp2(tf.linalg.norm(self.dense2(x)))
