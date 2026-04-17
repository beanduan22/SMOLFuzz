# SMOLFuzz TF model 126 | attempts=1
# APIS_SELECTED = ['tf.keras.optimizers.schedules.InverseTimeDecay', 'tf.keras.metrics.squared_hinge', 'tf.keras.layers.Maximum', 'tf.keras.layers.RandomCrop', 'tf.keras.metrics.poisson', 'tf.keras.metrics.SparseCategoricalAccuracy', 'tf.keras.metrics.AUC', 'tf.keras.metrics.Mean', 'tf.keras.losses.mape', 'tf.keras.losses.MeanAbsolutePercentageError', 'tf.equal', 'tf.broadcast_dynamic_shape', 'tf.numpy_function', 'tf.math.reduce_sum', 'tf.linalg.LinearOperatorTridiag', 'tf.linalg.eigh', 'tf.math.special.dawsn', 'tf.linalg.normalize', 'tf.linalg.logdet', 'tf.linalg.tridiagonal_solve']
# USED_APIS = ['tf.keras.layers.Dense', 'tf.keras.layers.BatchNormalization', 'tf.keras.layers.LayerNormalization', 'tf.GradientTape', 'tf.keras.layers.Maximum', 'tf.math.reduce_sum', 'tf.keras.metrics.Mean', 'tf.keras.metrics.SparseCategoricalAccuracy', 'tf.device']

class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(8)
        self.bn = tf.keras.layers.BatchNormalization()
        self.ln = tf.keras.layers.LayerNormalization()
        self.dense2 = tf.keras.layers.Dense(8)
        self.max_layer = tf.keras.layers.Maximum()
        self.mean_metric = tf.keras.metrics.Mean(name='mean')
        self.sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='sparse_cat_acc')

    def call(self, x, training=False):
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = self.dense1(x)
            y = self.bn(y, training=training)
            y = self.ln(y)
            z = self.dense2(y)
        g = tape.gradient(z, x)
        
        with tf.device('/CPU:0'):
            h = self.max_layer([y, z])
        
        i = h + g
        j = self.mean_metric(i)
        k = self.sparse_categorical_accuracy(tf.argmax(y, axis=1), tf.argmax(z, axis=1))
        
        return i

USED_APIS = ["tf.keras.layers.Dense", "tf.keras.layers.BatchNormalization", 
             "tf.keras.layers.LayerNormalization", "tf.GradientTape", 
             "tf.keras.layers.Maximum", "tf.math.reduce_sum", 
             "tf.keras.metrics.Mean", "tf.keras.metrics.SparseCategoricalAccuracy", 
             "tf.device"]
