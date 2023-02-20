from typing import Optional

import tensorflow as tf
from tensorflow import keras


class MarginLoss(keras.losses.Loss):
    def __init__(self, pos_weight: Optional[float] = 0.9, lambda_: Optional[float] = 0.5,
                 reduction: Optional[keras.losses.Reduction] = keras.losses.Reduction.AUTO,
                 name: Optional[str] = 'margin_loss'):
        super().__init__(reduction=reduction, name=name)
        self.pos_weight = pos_weight
        self.lambda_ = lambda_

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = keras.utils.to_categorical(y_true, y_pred.shape[-1])
        loss_positive = y_true * tf.nn.relu(self.pos_weight - y_pred)
        loss_negative = (1 - y_true) * tf.nn.relu(y_pred - self.pos_weight + 1)
        loss = tf.reduce_sum(loss_positive ** 2 + self.lambda_ * loss_negative ** 2, axis=-1)
        return loss


class SpreadLoss(keras.losses.Loss):
    def __init__(self, margin: Optional[float] = 0.2,
                 reduction: Optional[keras.losses.Reduction] = keras.losses.Reduction.AUTO,
                 name: Optional[str] = 'spread_loss'):
        super().__init__(reduction=reduction, name=name)
        self.margin = margin

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        print(y_true)
        y_true = keras.utils.to_categorical(y_true, y_pred.shape[-1])
        mask = y_true != 1
        loss = y_true - y_pred
        loss = tf.nn.relu(self.margin - loss) ** 2
        loss = tf.reduce_sum(loss * mask, axis=-1)
        return loss
