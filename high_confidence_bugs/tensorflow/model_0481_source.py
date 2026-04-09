# SMOLFuzz TF model 481
# APIs: ['tf.keras.activations.relu', 'tf.keras.initializers.GlorotUniform', 'tf.keras.optimizers.deserialize', 'tf.math.not_equal', 'tf.experimental.numpy.full', 'tf.keras.applications.InceptionResNetV2', 'tf.nn.ctc_unique_labels', 'tf.math.top_k', 'tf.keras.layers.SimpleRNNCell', 'tf.math.bincount', 'tf.keras.layers.GlobalMaxPool1D', 'tf.math.real']

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.dense1 = tf.keras.layers.Dense(8, activation=tf.keras.activations.relu, kernel_initializer=tf.keras.initializers.GlorotUniform())
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.dense2 = tf.keras.layers.Dense(8, activation=tf.keras.activations.relu, kernel_initializer=tf.keras.initializers.GlorotUniform())
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.global_max_pool = tf.keras.layers.GlobalMaxPool1D()

    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.batch_norm(x, training=training)
        x = self.dense2(x)
        x = self.layer_norm(x)
        x = self.dropout(x, training=training)
        x = tf.expand_dims(x, axis=1)  # Reshape for GlobalMaxPool1D
        x = self.global_max_pool(x)
        return x
