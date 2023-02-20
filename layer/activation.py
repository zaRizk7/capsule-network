from typing import Optional, Union

import tensorflow as tf
from tensorflow import keras

__all__ = ['Squash']

VALID_ACTIVATIONS = ['linear', 'dr', 'sa']


class Squash(keras.layers.Layer):
    def __init__(self, activation: Optional[str] = 'linear', ord: Optional[Union[str, int]] = 'euclidean',
                 axis: Optional[int] = -1,
                 **kwargs):
        super().__init__(**kwargs)
        if not activation in VALID_ACTIVATIONS:
            raise ValueError(f"Activation can only be either {VALID_ACTIVATIONS}")

        self.activation = activation
        self.ord = ord
        self.axis = axis

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        if self.activation == 'linear':
            return inputs

        norm = tf.norm(inputs, ord=self.ord, axis=self.axis, keepdims=True)

        if self.activation == 'dr':
            return squash_dr(inputs, norm)

        return squash_sa(inputs, norm)


def squash_dr(inputs: tf.Tensor, norm: tf.Tensor) -> tf.Tensor:
    return inputs * norm / (1 + norm ** 2)


def squash_sa(inputs: tf.Tensor, norm: tf.Tensor) -> tf.Tensor:
    return (1 - 1 / tf.exp(norm)) * inputs / norm
