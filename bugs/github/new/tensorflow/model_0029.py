# SMOLFuzz TF model 29 | attempts=2
# APIS_SELECTED = ['tf.optimizers.Adadelta', 'tf.keras.initializers.GlorotUniform', 'tf.keras.layers.dot', 'tf.keras.metrics.get', 'tf.keras.layers.experimental.preprocessing.RandomHeight', 'tf.keras.losses.kullback_leibler_divergence', 'tf.losses.Poisson', 'tf.initializers.Identity', 'tf.keras.metrics.SensitivityAtSpecificity', 'tf.keras.metrics.FalsePositives', 'tf.where', 'tf.dynamic_stitch', 'tf.IndexedSlices', 'tf.linalg.LinearOperatorIdentity', 'tf.cos', 'tf.linalg.expm', 'tf.math.cosh', 'tf.floor', 'tf.fingerprint', 'tf.while_loop']
# USED_APIS = ['tf.keras.layers.Dense', 'tf.keras.initializers.GlorotUniform', 'tf.keras.layers.BatchNormalization', 'tf.keras.layers.LayerNormalization', 'tf.keras.layers.Activation', 'tf.cos', 'tf.math.cosh', 'tf.GradientTape', 'tf.device', 'tf.optimizers.Adadelta']

class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(8, kernel_initializer=tf.keras.initializers.GlorotUniform())
        self.bn = tf.keras.layers.BatchNormalization()
        self.ln = tf.keras.layers.LayerNormalization()
        self.dense2 = tf.keras.layers.Dense(8)
        self.dense3 = tf.keras.layers.Dense(8)
        self.activation1 = tf.keras.layers.Activation(tf.cos)
        self.activation2 = tf.keras.layers.Activation(tf.math.cosh)
        self.metric1 = tf.keras.metrics.SensitivityAtSpecificity(0.5)
        self.metric2 = tf.keras.metrics.FalsePositives()
        self.optimizer = tf.optimizers.Adadelta()

    def call(self, x, training=False):
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = self.dense1(x)
            y = self.bn(y, training=training)
            y = self.ln(y)
            z = self.dense2(y)
            z = self.activation1(z)
        g = tape.gradient(z, x)
        with tf.device('/CPU:0'):
            h = self.dense3(z + g)
            h = self.activation2(h)
        return h

USED_APIS = ["tf.keras.layers.Dense", "tf.keras.initializers.GlorotUniform", 
             "tf.keras.layers.BatchNormalization", "tf.keras.layers.LayerNormalization", 
             "tf.keras.layers.Activation", "tf.cos", "tf.math.cosh", 
             "tf.GradientTape", "tf.device", "tf.optimizers.Adadelta"]
