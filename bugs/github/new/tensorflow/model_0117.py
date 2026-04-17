# SMOLFuzz TF model 117 | attempts=1
# APIS_SELECTED = ['tf.optimizers.Nadam', 'tf.metrics.MeanAbsolutePercentageError', 'tf.metrics.MeanAbsoluteError', 'tf.metrics.logcosh', 'tf.metrics.TopKCategoricalAccuracy', 'tf.nn.batch_norm_with_global_normalization', 'tf.keras.losses.KLD', 'tf.keras.activations.swish', 'tf.losses.KLD', 'tf.keras.layers.experimental.preprocessing.TextVectorization', 'tf.metrics.MeanMetricWrapper', 'tf.broadcast_to', 'tf.SparseTensorSpec', 'tf.math.floormod', 'tf.linalg.LinearOperatorLowRankUpdate', 'tf.math.logical_not', 'tf.signal.rfft3d', 'tf.math.special.bessel_i1', 'tf.atan', 'tf.get_logger']
# USED_APIS = ['tf.keras.layers.Dense', 'tf.keras.layers.BatchNormalization', 'tf.keras.layers.LayerNormalization', 'tf.keras.activations.swish', 'tf.metrics.MeanMetricWrapper', 'tf.keras.losses.logcosh', 'tf.metrics.MeanAbsoluteError', 'tf.GradientTape', 'tf.device']

class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(8)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.dense2 = tf.keras.layers.Dense(8, activation=tf.keras.activations.swish)
        self.logcosh_metric = tf.metrics.MeanMetricWrapper(tf.keras.losses.logcosh, name='logcosh')
        self.mean_abs_error_metric = tf.metrics.MeanAbsoluteError(name='mean_abs_error')

    def call(self, x, training=False):
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = self.dense1(x)
            y = self.batch_norm(y, training=training)
            y = self.layer_norm(y)
            z = self.dense2(y)
        
        g = tape.gradient(z, x)
        z = z + g
        
        with tf.device('/CPU:0'):
            logcosh_loss = self.logcosh_metric(x, z)
            mean_abs_error = self.mean_abs_error_metric(x, z)
        
        return z

USED_APIS = ["tf.keras.layers.Dense", "tf.keras.layers.BatchNormalization",
             "tf.keras.layers.LayerNormalization", "tf.keras.activations.swish",
             "tf.metrics.MeanMetricWrapper", "tf.keras.losses.logcosh",
             "tf.metrics.MeanAbsoluteError", "tf.GradientTape", "tf.device"]
