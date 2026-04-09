# SMOLFuzz TF model 381
# APIs: ['tf.keras.applications.resnet50.ResNet50', 'tf.keras.metrics.PrecisionAtRecall', 'tf.experimental.numpy.concatenate', 'tf.pow', 'tf.keras.layers.AveragePooling2D', 'tf.keras.applications.mobilenet_v2.MobileNetV2', 'tf.keras.optimizers.Adam', 'tf.experimental.numpy.arccosh', 'tf.keras.metrics.MeanSquaredLogarithmicError', 'tf.nn.max_pool3d', 'tf.clip_by_value', 'tf.experimental.numpy.diag_indices']

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.dense1 = tf.keras.layers.Dense(8, activation='relu', input_dim=8)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.dense2 = tf.keras.layers.Dense(4, activation='relu')
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense3 = tf.keras.layers.Dense(1)

    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.batch_norm(x)
        x = self.dense2(x)
        x = self.layer_norm(x)
        x = self.dropout(x, training=training)
        x = self.dense3(x)
        return tf.pow(tf.clip_by_value(x, -1.0, 1.0), 2)
