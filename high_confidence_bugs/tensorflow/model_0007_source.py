# SMOLFuzz TF model 7
# APIs: ['tf.experimental.numpy.exp2', 'tf.experimental.numpy.int8', 'tf.experimental.numpy.lcm', 'tf.experimental.numpy.asanyarray', 'tf.keras.layers.Conv1DTranspose', 'tf.keras.layers.Concatenate', 'tf.keras.utils.timeseries_dataset_from_array', 'tf.keras.layers.Input', 'tf.math.special.expint', 'tf.keras.layers.MultiHeadAttention', 'tf.math.squared_difference', 'tf.keras.applications.DenseNet121']

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.dense1 = tf.keras.layers.Dense(8, input_dim=8)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(8)
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=4)

    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.batch_norm(x, training=training)
        x = tf.math.special.expint(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        x = self.layer_norm(x)
        return tf.reduce_sum(x)
