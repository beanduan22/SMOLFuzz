# SMOLFuzz TF model 68 | attempts=4
# APIS_SELECTED = ['tf.optimizers.schedules.deserialize', 'tf.initializers.Initializer', 'tf.losses.mean_absolute_percentage_error', 'tf.keras.layers.CenterCrop', 'tf.keras.losses.MSE', 'tf.losses.msle', 'tf.keras.activations.hard_sigmoid', 'tf.keras.layers.PReLU', 'tf.keras.regularizers.L2', 'tf.losses.mape', 'tf.math.is_inf', 'tf.concat', 'tf.zeros_like', 'tf.clip_by_value', 'tf.histogram_fixed_width_bins', 'tf.math.special.dawsn', 'tf.signal.irfft', 'tf.math.abs', 'tf.signal.inverse_mdct', 'tf.group']
# USED_APIS = ['tf.keras.layers.Dense', 'tf.keras.layers.PReLU', 'tf.keras.layers.BatchNormalization', 'tf.keras.losses.MeanAbsoluteError', 'tf.GradientTape', 'tf.zeros_like', 'tf.clip_by_value', 'tf.device']

class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(8, kernel_regularizer=tf.keras.regularizers.L2())
        self.prelu = tf.keras.layers.PReLU()
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.dense2 = tf.keras.layers.Dense(8)
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x, training=False):
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = self.dense1(x)
            z = self.prelu(y)
            w = self.batch_norm(z, training=training)
            u = self.dense2(w)
            v = self.layer_norm(u)
            mae_loss = tf.keras.losses.MeanAbsoluteError()(tf.zeros_like(v), v)
        grad = tape.gradient(mae_loss, x)
        clipped_grad = tf.clip_by_value(grad, -1.0, 1.0)
        with tf.device('/CPU:0'):
            final_output = v + clipped_grad
        return final_output

USED_APIS = ["tf.keras.layers.Dense", "tf.keras.layers.PReLU", "tf.keras.layers.BatchNormalization",
             "tf.keras.losses.MeanAbsoluteError", "tf.GradientTape", "tf.zeros_like", "tf.clip_by_value", "tf.device"]
