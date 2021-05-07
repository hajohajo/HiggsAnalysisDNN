import tensorflow as tf

class InputSanitizerLayer(tf.keras.layers.Layer):

    def __init__(self, min_values, max_values, **kwargs):
        super(InputSanitizerLayer, self).__init__(**kwargs)
        self._min_values = min_values
        self._max_values = max_values
        self._min_values_tensor = tf.constant(self._min_values, shape=(1, self._min_values.shape[-1]))
        self._max_values_tensor = tf.constant(self._max_values, shape=(1, self._max_values.shape[-1]))

    def build(self, input_shape):
        super(InputSanitizerLayer, self).build(input_shape)

    @tf.function
    def call(self, input):
        _min_tensor = tf.cast(self._min_values_tensor, dtype=input.dtype)
        _max_tensor = tf.cast(self._max_values_tensor, dtype=input.dtype)

        #abs values
        _abs = tf.math.abs(input)

        #log values
        _log = tf.math.log1p(_abs[:, 4:])
        _rest = _abs[:, :4]
        _log_out = tf.concat([_rest, _log], axis=1)

        #clipped
        _clipped = tf.clip_by_value(_log_out, clip_value_min=_min_tensor, clip_value_max=_max_tensor)

        #scaling
        # _sanitized = tf.subtract(tf.multiply(tf.constant(2.0, dtype=input.dtype), tf.math.divide(tf.subtract(_clipped, _min_tensor), tf.subtract(_max_tensor, _min_tensor))),
        #              tf.constant(1.0, dtype=input.dtype))
        _sanitized = tf.math.divide(tf.subtract(_clipped, _min_tensor), tf.subtract(_max_tensor, _min_tensor))

        return _sanitized

    def get_config(self):
        config = super().get_config()
        config.update({
            "min_values": self._min_values,
            "max_values": self._max_values
        })
        return config