# SMOLFuzz TF model 482
# APIs: ['tf.keras.layers.MaxPooling3D', 'tf.experimental.numpy.uint32', 'tf.keras.callbacks.ModelCheckpoint', 'tf.keras.applications.efficientnet.EfficientNetB1', 'tf.keras.preprocessing.image.smart_resize', 'tf.keras.initializers.HeNormal', 'tf.math.asinh', 'tf.experimental.numpy.expm1', 'tf.experimental.numpy.log', 'tf.keras.backend.reset_uids', 'tf.keras.initializers.lecun_normal', 'tf.nn.ctc_beam_search_decoder']

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.dense1 = tf.keras.layers.Dense(16, input_dim=8, kernel_initializer=tf.keras.initializers.HeNormal())
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(32, input_dim=16, kernel_initializer=tf.keras.initializers.lecun_normal())
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.batch_norm(x, training=training)
        x = tf.math.asinh(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        return tf.reduce_sum(x)
