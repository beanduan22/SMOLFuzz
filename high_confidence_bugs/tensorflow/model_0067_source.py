# SMOLFuzz TF model 67
# APIs: ['tf.linalg.tensor_diag', 'tf.keras.backend.set_floatx', 'tf.keras.applications.mobilenet_v3.preprocess_input', 'tf.keras.constraints.min_max_norm', 'tf.keras.losses.MeanAbsoluteError', 'tf.keras.optimizers.schedules.PiecewiseConstantDecay', 'tf.keras.optimizers.Nadam', 'tf.experimental.numpy.experimental_enable_numpy_behavior', 'tf.keras.applications.resnet_v2.preprocess_input', 'tf.math.count_nonzero', 'tf.keras.regularizers.l1_l2', 'tf.signal.dct']

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        tf.keras.backend.set_floatx('float32')
        self.dense1 = tf.keras.layers.Dense(8, input_dim=8, kernel_constraint=tf.keras.constraints.min_max_norm(min_value=-0.5, max_value=0.5))
        self.bn = tf.keras.layers.BatchNormalization()
        self.ln = tf.keras.layers.LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(1, input_dim=8, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01))

    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.bn(x)
        x = tf.signal.dct(x, type=2)
        x = self.dropout(x, training=training)
        return self.dense2(x)
