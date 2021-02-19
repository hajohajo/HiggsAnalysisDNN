from tensorflow.keras.metrics import AUC, Metric
from NeuralNetwork.disco_tf import distance_corr
import tensorflow as tf

class CustomAUC(AUC):
    def __init__(self, name="auc", **kwargs):
        super(CustomAUC, self).__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true[:, 0], (-1, 1)), y_pred.dtype)
        super(CustomAUC, self).update_state(y_true, y_pred)
        return super(CustomAUC, self).result()


class DiscoMetric(Metric):
    def __init__(self, name="disco", **kwargs):
        super(DiscoMetric, self).__init__(name=name, **kwargs)
        self.disco_value = self.add_weight(name="disco", initializer="zeros")
        self.n_update = self.add_weight("n_update", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.convert_to_tensor(y_pred)
        sample_weights = tf.cast(tf.reshape(y_true[:, 2], (-1, 1)), y_pred.dtype)
        mt = tf.cast(tf.reshape(y_true[:, 1], (-1, 1)), y_pred.dtype)
#        y_true = tf.cast(tf.reshape(y_true[:, 0], (-1, 1)), y_pred.dtype)

        dc_pred = tf.reshape(y_pred, [tf.size(y_pred)])
        dc_mt = tf.reshape(mt, [tf.size(mt)])
        dc_weights = tf.cast(tf.reshape(sample_weights, [tf.size(sample_weights)]), y_pred.dtype)
        value = distance_corr(dc_mt, dc_pred, normedweight=dc_weights, power=1)
        self.disco_value.assign_add(value)
        self.n_update.assign_add(1)

    def result(self):
        return self.disco_value/self.n_update

    def reset_states(self):
        self.disco_value.assign(0.0)
        self.n_update.assign(0.0)
