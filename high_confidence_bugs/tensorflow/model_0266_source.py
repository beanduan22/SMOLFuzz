# SMOLFuzz TF model 266
# APIs: ['tf.keras.utils.custom_object_scope', 'tf.math.tanh', 'tf.keras.layers.ZeroPadding3D', 'tf.keras.layers.Lambda', 'tf.keras.preprocessing.text_dataset_from_directory', 'tf.keras.metrics.binary_accuracy', 'tf.tan', 'tf.experimental.numpy.float32', 'tf.math.cumulative_logsumexp', 'tf.keras.metrics.MSLE', 'tf.keras.metrics.MeanAbsoluteError', 'tf.linalg.matmul']

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.dense1 = tf.keras.layers.Dense(16, input_dim=8)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.lambda_layer = tf.keras.layers.Lambda(lambda x: tf.math.tanh(x))
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.batch_norm(x)
        x = self.dropout(x, training=training)
        x = self.lambda_layer(x)
        x = self.layer_norm(x)
        return tf.reduce_mean(tf.math.tan(x))
