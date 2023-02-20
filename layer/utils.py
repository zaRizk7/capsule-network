from typing import Optional, Union

import tensorflow as tf
from tensorflow import keras


class Length(keras.layers.Layer):
    def __init__(self, ord: Optional[Union[str, int]] = 'euclidean', axis: Optional[int] = -1, **kwargs):
        super().__init__(**kwargs)
        self.ord = ord
        self.axis = axis

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return tf.norm(inputs, ord=self.ord, axis=self.axis)
