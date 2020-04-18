import tensorflow as tf
import numpy as np

class C_Concat(tf.keras.layers.Layer):
    def __init__(self, axis):
        super(C_Concat, self).__init__()
        self.axis = axis

    def call(self, inputs1, inputs2):
        out = tf.concat([inputs1, inputs2], self.axis)
        return out

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'axis': self.axis
        })
        return config

