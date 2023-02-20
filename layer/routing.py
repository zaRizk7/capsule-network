from typing import Optional

import tensorflow as tf
from einops import einsum, reduce, rearrange
from tensorflow import keras

from layer.activation import Squash


class Router(keras.layers.Layer):
    def __init__(self, activation: Optional[str] = 'linear', bias_initializer: Optional[str] = 'zeros',
                 bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
                 bias_constraint: Optional[keras.regularizers.Regularizer] = None, **kwargs):
        super().__init__(**kwargs)
        self.activation = Squash(activation)
        self.bias_initializer = bias_initializer
        self.bias_regularizer = bias_regularizer
        self.bias_constraint = bias_constraint

    def build(self, input_shape) -> None:
        capsules, units = input_shape[2:]
        self.bias = self.add_weight(name='bias', shape=(capsules, units),
                                    initializer=self.bias_initializer, regularizer=self.bias_regularizer,
                                    trainable=True, constraint=self.bias_constraint)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return self.activation(inputs + self.bias)


class DynamicRouter(Router):
    def __init__(self, num_routing: Optional[int] = 3, activation: Optional[str] = 'linear',
                 bias_initializer: Optional[str] = 'zeros',
                 bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
                 bias_constraint: Optional[keras.regularizers.Regularizer] = None, **kwargs):
        super().__init__(activation, bias_initializer, bias_regularizer, bias_constraint, **kwargs)
        self.num_routing = 1 if num_routing <= 0 or num_routing is None else num_routing
        self.pattern = '... n_in n_out d_out -> ... n_out d_out'

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        priors = tf.zeros_like(inputs)
        for i in range(self.num_routing):
            agreements = tf.nn.softmax(priors, axis=2)
            outputs = tf.reduce_sum(inputs * agreements, axis=1, keepdims=True)
            outputs = outputs + self.bias
            outputs = self.activation(outputs)
            if i < self.num_routing - 1:
                priors = priors + tf.reduce_sum(inputs * outputs, axis=-1, keepdims=True)
        return reduce(outputs, self.pattern, reduction='sum')


class SelfAttentionRouter(Router):
    def __init__(self, activation: Optional[str] = 'linear',
                 bias_initializer: Optional[str] = 'zeros',
                 bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
                 bias_constraint: Optional[keras.regularizers.Regularizer] = None, **kwargs):
        super().__init__(activation, bias_initializer, bias_regularizer, bias_constraint, **kwargs)
        self.pattern_sym = '... n_in n_out1 d_out, ... n_in n_out2 d_out -> ... n_in n_out1'
        self.pattern_reduce = '... n_in n_out d_out -> ... n_out d_out'

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        outputs = einsum(inputs, inputs, self.pattern_sym)
        outputs = outputs[..., None]
        outputs = outputs / inputs.shape[-1] ** 0.5
        outputs = tf.nn.softmax(outputs, axis=2)
        outputs = outputs + self.bias
        outputs = reduce(inputs * outputs, self.pattern_reduce, reduction='sum')
        outputs = self.activation(outputs)
        return outputs


if __name__ == '__main__':
    inputs = keras.layers.Input((32, 10, 16))
    router = SelfAttentionRouter()
    keras.Sequential([inputs, router]).summary()
