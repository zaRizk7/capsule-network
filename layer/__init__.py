from typing import Optional

import tensorflow as tf
from tensorflow import keras

from .capsule import PrimaryCapsule, CapsuleTransform
from .router import DynamicRouter, SelfAttentionRouter
from .utils import Length


class DRCapsule(keras.layers.Layer):
    def __init__(self, capsules: int, units: int, num_routing: Optional[int] = 3, activation: Optional[str] = 'linear',
                 kernel_initializer: Optional[str] = 'glorot_uniform', bias_initializer: Optional[str] = 'zeros',
                 kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
                 bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
                 kernel_constraint: Optional[keras.regularizers.Regularizer] = None,
                 bias_constraint: Optional[keras.regularizers.Regularizer] = None, **kwargs):
        super().__init__(**kwargs)
        self.transform = CapsuleTransform(capsules, units, kernel_initializer, kernel_regularizer, kernel_constraint)
        self.route = DynamicRouter(num_routing, activation, bias_initializer, bias_regularizer, bias_constraint)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        outputs = self.transform(inputs)
        outputs = self.route(outputs)
        return outputs


class SACapsule(keras.layers.Layer):
    def __init__(self, capsules: int, units: int, activation: Optional[str] = 'linear',
                 kernel_initializer: Optional[str] = 'glorot_uniform', bias_initializer: Optional[str] = 'zeros',
                 kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
                 bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
                 kernel_constraint: Optional[keras.regularizers.Regularizer] = None,
                 bias_constraint: Optional[keras.regularizers.Regularizer] = None, **kwargs):
        super().__init__(**kwargs)
        self.transform = CapsuleTransform(capsules, units, kernel_initializer, kernel_regularizer, kernel_constraint)
        self.route = SelfAttentionRouter(activation, bias_initializer, bias_regularizer, bias_constraint)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        outputs = self.transform(inputs)
        outputs = self.route(outputs)
        return outputs
