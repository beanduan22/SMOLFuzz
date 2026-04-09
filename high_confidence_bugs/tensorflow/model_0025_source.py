# SMOLFuzz TF model 25
# APIs: ['tf.nn.dropout', 'tf.keras.activations.selu', 'tf.keras.metrics.sparse_top_k_categorical_accuracy', 'tf.math.polyval', 'tf.keras.layers.AvgPool3D', 'tf.keras.preprocessing.image.apply_channel_shift', 'tf.experimental.numpy.tan', 'tf.keras.applications.resnet.preprocess_input', 'tf.keras.constraints.unit_norm', 'tf.keras.applications.NASNetLarge', 'tf.keras.layers.average', 'tf.keras.applications.EfficientNetB1']

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.dense1 = tf.keras.layers.Dense(8, activation=tf.keras.activations.selu)
        self.dense2 = tf.keras.layers.Dense(8, kernel_constraint=tf.keras.constraints.unit_norm())
        self.bn = tf.keras.layers.BatchNormalization()
        self.ln = tf.keras.layers.LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(0.5)

    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.bn(x, training=training)
        x = self.ln(x)
        x = tf.math.polyval([1., 2.], x)
        x = self.dropout(x, training=training)
        return tf.reduce_sum(x, axis=-1)
