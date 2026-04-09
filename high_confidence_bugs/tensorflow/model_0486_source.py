# SMOLFuzz TF model 486
# APIs: ['tf.math.unsorted_segment_max', 'tf.math.scalar_mul', 'tf.keras.layers.Conv1D', 'tf.experimental.numpy.broadcast_arrays', 'tf.math.sigmoid', 'tf.keras.losses.mean_absolute_error', 'tf.experimental.numpy.greater', 'tf.experimental.numpy.finfo', 'tf.math.reduce_prod', 'tf.signal.linear_to_mel_weight_matrix', 'tf.experimental.numpy.amax', 'tf.keras.layers.DepthwiseConv2D']

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.dense1 = tf.keras.layers.Dense(16, input_dim=8)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.dense2 = tf.keras.layers.Dense(32)

    def call(self, x, training=False):
        x = tf.math.sigmoid(self.dense1(x))
        x = tf.math.scalar_mul(2.0, x)
        x = self.batch_norm(x, training=training)
        x = tf.math.reduce_prod(x, axis=-1, keepdims=True)
        return tf.experimental.numpy.amax(tf.math.unsorted_segment_max(x, segment_ids=[0]*4, num_segments=1))
