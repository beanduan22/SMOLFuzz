# SMOLFuzz TF model 9
# APIs: ['tf.keras.metrics.BinaryCrossentropy', 'tf.keras.metrics.FalsePositives', 'tf.experimental.numpy.tan', 'tf.keras.initializers.glorot_uniform', 'tf.keras.applications.mobilenet_v3.decode_predictions', 'tf.math.acos', 'tf.keras.layers.subtract', 'tf.keras.losses.BinaryCrossentropy', 'tf.math.squared_difference', 'tf.nn.lrn', 'tf.keras.losses.msle', 'tf.keras.initializers.TruncatedNormal']

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.dense1 = tf.keras.layers.Dense(8, activation='relu', input_dim=8, kernel_initializer=tf.keras.initializers.glorot_uniform())
        self.bn = tf.keras.layers.BatchNormalization()
        self.ln = tf.keras.layers.LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=tf.keras.initializers.TruncatedNormal())

    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.bn(x)
        x = self.ln(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        return tf.experimental.numpy.tan(tf.math.acos(x))
