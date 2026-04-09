# SMOLFuzz TF model 490
# APIs: ['tf.math.top_k', 'tf.keras.regularizers.l2', 'tf.keras.layers.Conv2D', 'tf.keras.initializers.LecunNormal', 'tf.exp', 'tf.math.imag', 'tf.experimental.numpy.tri', 'tf.experimental.numpy.issubdtype', 'tf.experimental.numpy.arctanh', 'tf.nn.max_pool1d', 'tf.linalg.LinearOperatorTridiag', 'tf.experimental.numpy.expand_dims']

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.dense1 = tf.keras.layers.Dense(16, input_shape=(8,), kernel_initializer=tf.keras.initializers.LecunNormal(), kernel_regularizer=tf.keras.regularizers.l2())
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.dropout1 = tf.keras.layers.Dropout(0.2)
        self.layer_norm1 = tf.keras.layers.LayerNormalization()
        self.dense2 = tf.keras.layers.Dense(8, input_dim=16)

    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.batch_norm1(x, training=training)
        x = tf.exp(tf.math.top_k(x, k=4)[0])
        x = self.dropout1(x, training=training)
        return tf.reduce_mean(x)
