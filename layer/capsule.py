from typing import Union, Optional, Tuple

import tensorflow as tf
from einops import rearrange, einsum
from tensorflow import keras

from layer.activation import Squash


class PrimaryCapsule(keras.layers.Layer):
    def __init__(self, capsules: int, units: int, kernel_size: Optional[Union[int, Tuple[int, int]]] = None,
                 strides: Optional[Union[int, Tuple[int, int]]] = (1, 1), padding: Optional[Union[str, int]] = 'valid',
                 depthwise: Optional[bool] = False, activation: Optional[str] = 'linear',
                 use_bias: Optional[bool] = True, kernel_initializer: Optional[str] = 'glorot_uniform',
                 bias_initializer: Optional[str] = 'zeros',
                 kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
                 bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
                 activity_regularizer: Optional[keras.regularizers.Regularizer] = None,
                 kernel_constraint: Optional[keras.constraints.Constraint] = None,
                 bias_constraint: Optional[keras.constraints.Constraint] = None, **kwargs):
        super().__init__(**kwargs)
        self.capsules = capsules
        self.units = units
        self.kernel_size = kernel_size
        self.padding = padding
        self.strides = strides
        self.depthwise = depthwise
        self.use_bias = use_bias
        self.activation = Squash(activation)
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.pattern = '... h w (n d) -> ... (h w n) d'

    def build(self, input_shape: Tuple[None, int, int, int]) -> None:
        b, h, w, c = input_shape

        filters = self.units * self.capsules

        if self.kernel_size is None:
            self.kernel_size = [h, w]

        groups = 1
        if self.depthwise:
            groups = filters

        self.convolve = keras.layers.Conv2D(filters, self.kernel_size, self.strides, self.padding, groups=groups,
                                            use_bias=self.use_bias, kernel_initializer=self.kernel_initializer,
                                            bias_initializer=self.bias_initializer,
                                            kernel_regularizer=self.kernel_regularizer,
                                            bias_regularizer=self.bias_regularizer,
                                            activity_regularizer=self.activity_regularizer,
                                            kernel_constraint=self.kernel_constraint,
                                            bias_constraint=self.bias_constraint)

    def to_capsule(self, inputs: tf.Tensor) -> tf.Tensor:
        return rearrange(inputs, self.pattern, n=self.capsules, d=self.units)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        outputs = self.convolve(inputs)
        outputs = self.to_capsule(outputs)
        outputs = self.activation(outputs)
        return outputs


class CapsuleTransform(keras.layers.Layer):
    def __init__(self, capsules: int, units: int, kernel_initializer: Optional[str] = 'glorot_uniform',
                 kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
                 kernel_constraint: Optional[keras.regularizers.Regularizer] = None, **kwargs):
        super().__init__(**kwargs)
        self.capsules = capsules
        self.units = units
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint
        self.pattern = '... n_in d_in, n_in d_in n_out d_out -> ... n_in n_out d_out'

    def build(self, input_shape: Tuple[None, int, int]) -> None:
        _, in_capsules, in_units = input_shape

        self.kernel = self.add_weight(name='kernel', shape=(in_capsules, in_units, self.capsules, self.units),
                                      initializer=self.kernel_initializer, regularizer=self.kernel_regularizer,
                                      trainable=True, constraint=self.kernel_constraint)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return einsum(inputs, self.kernel, self.pattern)
