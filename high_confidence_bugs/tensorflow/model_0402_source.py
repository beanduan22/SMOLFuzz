# SMOLFuzz TF model 402
# APIs: ['tf.experimental.numpy.sum', 'tf.linalg.einsum', 'tf.keras.constraints.NonNeg', 'tf.experimental.numpy.exp', 'tf.experimental.numpy.log2', 'tf.math.equal', 'tf.keras.losses.categorical_hinge', 'tf.keras.layers.ThresholdedReLU', 'tf.keras.layers.MaxPool1D', 'tf.keras.layers.AbstractRNNCell', 'tf.keras.losses.CategoricalCrossentropy', 'tf.keras.backend.reset_uids']

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.dense1 = tf.keras.layers.Dense(16, input_shape=(8,), kernel_constraint=tf.keras.constraints.NonNeg())
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.thresholded_relu = tf.keras.layers.ThresholdedReLU(theta=0.5)
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(1, input_dim=16)

    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.batch_norm(x, training=training)
        x = self.thresholded_relu(x)
        x = self.dropout(x, training=training)
        x = tf.experimental.numpy.exp(tf.linalg.einsum('ij,j->i', x, tf.constant([0.5]*16)))
        return tf.math.reduce_sum(x)
