from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import Constant
from NeuralNetwork import DiscoLoss
import tensorflow as tf
from tensorflow.keras.mixed_precision import LossScaleOptimizer

class Classifier():
    def __init__(self,
                 n_inputs,
                 neurons=128,
                 activation='relu',
                 regularizer_magnitude=1e-5,
                 optimizer='adam',
                 lr=1e-4,
                 layers=3,
                 init_bias=0.5,
                 disco_factor=10.0):
        self._n_inputs = n_inputs
        self._neurons = neurons
        self._activation = activation
        self._regularizer_magnitude = regularizer_magnitude
        self._optimizer = optimizer
        self._lr = lr
        self._layers = layers
        self._init_bias = init_bias
        self._disco_factor = disco_factor

    def get_model(self):
        try:
            self._neurons/pow(2,self._layers)==0
        except:
            ValueError("Make sure n_neurons is divisible by 2^n_layers")
        if self._optimizer == 'adam':
            _optimizer = optimizers.Adam(lr=self._lr, amsgrad=True)
        else:
            ValueError("Use 'adam' optimizer instead")
        _initializer = 'lecun_normal'

        # input_layer = layers.Input(self._n_inputs, name="input_layer")
        # dense_1 = layers.Dense(4096, activation=self._activation, kernel_initializer=_initializer)(input_layer)
        # x = layers.Dense(2048, activation=self._activation, kernel_initializer=_initializer)(dense_1)
        # x = layers.Dense(1024, activation=self._activation, kernel_initializer=_initializer)(x)
        # x = layers.Dense(512, activation=self._activation, kernel_initializer=_initializer)(x)

        input_layer = layers.Input(self._n_inputs, dtype=tf.float16, name="input_layer")
        x = input_layer
        for i in range(self._layers):
            x = layers.Dense(int(self._neurons/pow(2,i)),
                             activation=self._activation,
                             kernel_initializer=_initializer,
     #                        activity_regularizer=l2(self._regularizer_magnitude),
                             name="dense_{}".format((i+1)))(x)
            x = layers.BatchNormalization(name="batchnorm_{}".format((i+1)))(x)
#
        output_layer = layers.Dense(1, activation='sigmoid',
                                    kernel_initializer=_initializer,
                                    bias_initializer=Constant(self._init_bias))(x)

        # x = layers.Dense(1, kernel_initializer=_initializer,bias_initializer=Constant(self._init_bias), name="out_dense")(x)
        # output_layer = layers.Activation('sigmoid', dtype=tf.float32, name='predictions')(x)

        model = Model(input_layer, output_layer)
        loss = DiscoLoss(factor=self._disco_factor)
        model.compile(optimizer=LossScaleOptimizer(_optimizer),
#                      loss='binary_crossentropy')
                      loss=loss)
        return model

