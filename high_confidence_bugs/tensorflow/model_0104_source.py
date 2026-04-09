# SMOLFuzz TF model 104
# APIs: ['tf.keras.callbacks.EarlyStopping', 'tf.keras.constraints.MinMaxNorm', 'tf.experimental.numpy.hypot', 'tf.keras.initializers.lecun_uniform', 'tf.linalg.expm', 'tf.experimental.numpy.matmul', 'tf.keras.applications.resnet50.ResNet50', 'tf.keras.backend.image_data_format', 'tf.experimental.numpy.imag', 'tf.keras.losses.poisson', 'tf.keras.callbacks.Callback', 'tf.nn.relu']

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.dense1 = tf.keras.layers.Dense(16, input_dim=8, kernel_initializer=tf.keras.initializers.lecun_uniform())
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.dense2 = tf.keras.layers.Dense(1, input_dim=16, kernel_constraint=tf.keras.constraints.MinMaxNorm(min_value=-5, max_value=5))

    def call(self, x, training=False):
        x = tf.nn.relu(self.dense1(x))
        x = self.batch_norm(x)
        x = self.dropout(x, training=training)
        x = self.layer_norm(x)
        return self.dense2(x)
