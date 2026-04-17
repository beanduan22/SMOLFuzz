# SMOLFuzz TF model 154 | attempts=2
# APIS_SELECTED = ['tf.keras.optimizers.SGD', 'tf.keras.layers.experimental.preprocessing.CenterCrop', 'tf.initializers.TruncatedNormal', 'tf.initializers.Identity', 'tf.losses.SparseCategoricalCrossentropy', 'tf.keras.losses.hinge', 'tf.keras.metrics.MeanAbsolutePercentageError', 'tf.keras.losses.poisson', 'tf.metrics.deserialize', 'tf.keras.initializers.GlorotUniform', 'tf.transpose', 'tf.ones_like', 'tf.bitwise.bitwise_xor', 'tf.math.reduce_sum', 'tf.sinh', 'tf.math.digamma', 'tf.math.erfinv', 'tf.math.less', 'tf.linalg.diag', 'tf.guarantee_const']
# USED_APIS = ['tf.keras.layers.Dense', 'tf.initializers.TruncatedNormal', 'tf.keras.layers.BatchNormalization', 'tf.keras.layers.LayerNormalization', 'tf.keras.layers.Activation', 'tf.losses.SparseCategoricalCrossentropy', 'tf.keras.metrics.MeanAbsolutePercentageError', 'tf.GradientTape', 'tf.sinh', 'tf.math.digamma', 'tf.math.erfinv', 'tf.transpose', 'tf.ones_like', 'tf.bitwise.bitwise_xor', 'tf.math.reduce_sum', 'tf.device']

class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(8, kernel_initializer=tf.initializers.TruncatedNormal())
        self.bn = tf.keras.layers.BatchNormalization()
        self.ln = tf.keras.layers.LayerNormalization()
        self.dense2 = tf.keras.layers.Dense(8, kernel_initializer=tf.initializers.GlorotUniform())
        self.activation = tf.keras.layers.Activation('relu')
        self.loss_fn = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.metric = tf.keras.metrics.MeanAbsolutePercentageError()
    
    def call(self, x, training=False):
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = self.dense1(x)
            y = self.bn(y, training=training)
            y = self.ln(y)
            y = self.activation(y)
            z = self.dense2(y)
        
        g = tape.gradient(z, x)
        with tf.device('/CPU:0'):
            h = z + g
            i = tf.sinh(h)
            j = tf.math.digamma(i)
            k = tf.math.erfinv(j)
            l = tf.transpose(k)
            m = tf.ones_like(l)
            n = tf.bitwise.bitwise_xor(tf.cast(l, tf.int32), tf.cast(m, tf.int32))
            o = tf.math.reduce_sum(n, axis=1, keepdims=True)
        
        return o

USED_APIS = ["tf.keras.layers.Dense", "tf.initializers.TruncatedNormal", 
             "tf.keras.layers.BatchNormalization", "tf.keras.layers.LayerNormalization",
             "tf.keras.layers.Activation", "tf.losses.SparseCategoricalCrossentropy",
             "tf.keras.metrics.MeanAbsolutePercentageError", "tf.GradientTape", 
             "tf.sinh", "tf.math.digamma", "tf.math.erfinv", "tf.transpose", 
             "tf.ones_like", "tf.bitwise.bitwise_xor", "tf.math.reduce_sum", "tf.device"]
