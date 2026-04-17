# SMOLFuzz TF model 163 | attempts=4
# APIS_SELECTED = ['tf.optimizers.schedules.deserialize', 'tf.keras.experimental.SidecarEvaluator', 'tf.losses.MeanAbsolutePercentageError', 'tf.keras.layers.average', 'tf.metrics.mae', 'tf.keras.regularizers.l2', 'tf.metrics.MAPE', 'tf.losses.Poisson', 'tf.keras.metrics.poisson', 'tf.keras.regularizers.L1L2', 'tf.metrics.kullback_leibler_divergence', 'tf.space_to_batch', 'tf.zeros', 'tf.math.special.fresnel_cos', 'tf.eigvals', 'tf.linalg.diag_part', 'tf.math.multiply', 'tf.truediv', 'tf.linalg.normalize', 'tf.math.log']
# USED_APIS = ['tf.keras.layers.Dense', 'tf.keras.regularizers.L1L2', 'tf.keras.layers.BatchNormalization', 'tf.keras.layers.LayerNormalization', 'tf.keras.layers.Activation', 'tf.math.multiply', 'tf.GradientTape', 'tf.device']

class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(8, kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.01))
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.dense2 = tf.keras.layers.Dense(8)
        self.layer_norm = tf.keras.layers.LayerNormalization(input_shape=(8,))
        self.activation = tf.keras.layers.Activation('relu')
        self.dense3 = tf.keras.layers.Dense(8)

    def call(self, x, training=False):
        with tf.device('/CPU:0'):
            y = self.dense1(x)
            y = self.batch_norm(y, training=training)
            y = self.activation(y)
        
        z = self.dense2(y)
        z = self.layer_norm(z)
        z = tf.math.multiply(z, z)
        
        with tf.GradientTape() as tape:
            tape.watch(y)
            z = self.dense2(y)
            z = self.layer_norm(z)
            z = tf.math.multiply(z, z)
        g = tape.gradient(z, y)
        
        output = self.dense3(z + g)
        return output

USED_APIS = ["tf.keras.layers.Dense", "tf.keras.regularizers.L1L2", 
             "tf.keras.layers.BatchNormalization", "tf.keras.layers.LayerNormalization", 
             "tf.keras.layers.Activation", "tf.math.multiply", "tf.GradientTape", "tf.device"]
