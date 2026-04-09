# SMOLFuzz TF model 421
# APIs: ['tf.nn.sparse_softmax_cross_entropy_with_logits', 'tf.keras.models.clone_model', 'tf.keras.losses.cosine_similarity', 'tf.experimental.numpy.allclose', 'tf.experimental.numpy.where', 'tf.linalg.LinearOperatorInversion', 'tf.keras.applications.inception_resnet_v2.decode_predictions', 'tf.sort', 'tf.keras.metrics.TruePositives', 'tf.keras.applications.mobilenet_v3.decode_predictions', 'tf.experimental.numpy.random.rand', 'tf.keras.losses.CategoricalHinge']

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.dense1 = tf.keras.layers.Dense(8, input_dim=8)
        self.bn = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.ln = tf.keras.layers.LayerNormalization()
        self.dense2 = tf.keras.layers.Dense(1)

    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        x = self.dropout(x, training=training)
        x = self.ln(x)
        return self.dense2(x)
