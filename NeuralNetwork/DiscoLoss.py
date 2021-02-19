import tensorflow as tf
from tensorflow.keras.losses import Loss
from tensorflow.python.keras.utils import losses_utils
from NeuralNetwork.disco_tf import distance_corr

class DiscoLoss(Loss):
    '''
    Loss that includes a decorrelating term based on the Distance Decorrelation method
    '''
    def __init__(self, factor=15.0, reduction=losses_utils.ReductionV2.AUTO, name="DiscoLoss"):
        super().__init__(reduction=reduction, name=name)
        self.factor = factor

    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        sample_weights = tf.cast(tf.reshape(y_true[:, 2], (-1, 1)), y_pred.dtype)
        mt = tf.cast(tf.reshape(y_true[:, 1], (-1, 1)), y_pred.dtype)
        y_true = tf.cast(tf.reshape(y_true[:, 0], (-1, 1)), y_pred.dtype)

        dc_pred = tf.reshape(y_pred, [tf.size(y_pred)])
        dc_mt = tf.reshape(mt, [tf.size(mt)])
        dc_weights = tf.cast(tf.reshape(sample_weights, [tf.size(sample_weights)]), y_pred.dtype)

        if self.factor == 0.0:
            custom_loss = tf.losses.binary_crossentropy(y_true, y_pred, label_smoothing=0.0)
        else:
            custom_loss = tf.losses.binary_crossentropy(y_true, y_pred, label_smoothing=0.0) \
                                + self.factor * distance_corr(dc_mt, dc_pred, normedweight=dc_weights, power=1)

        return custom_loss

