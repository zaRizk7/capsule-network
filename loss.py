from typing import Optional

import tensorflow as tf
from tensorflow import keras


class MarginLoss(keras.losses.Loss):
    def __init__(self, pos_margin: Optional[float] = 0.9, neg_margin: Optional[float] = 0.1, lambda_: Optional[float] = 0.5,
                 reduction: Optional[keras.losses.Reduction] = keras.losses.Reduction.AUTO,
                 name: Optional[str] = 'margin_loss'):
        super().__init__(reduction=reduction, name=name)
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.lambda_ = lambda_

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.one_hot(y_true, y_pred.shape[-1], dtype=y_pred.dtype)
        loss_positive = y_true * tf.nn.relu(self.pos_margin - y_pred)
        loss_negative = (1 - y_true) * tf.nn.relu(y_pred - self.neg_margin)
        loss = loss_positive ** 2 + self.lambda_ * loss_negative ** 2
        loss = tf.reduce_sum(loss, axis=-1)
        return loss


class SpreadLoss(keras.losses.Loss):
    def __init__(self, margin: Optional[float] = 0.2,
                 reduction: Optional[keras.losses.Reduction] = keras.losses.Reduction.AUTO,
                 name: Optional[str] = 'spread_loss'):
        super().__init__(reduction=reduction, name=name)
        self.margin = margin

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.one_hot(y_true, y_pred.shape[-1], dtype=y_pred.dtype)
        mask = tf.cast(y_true != 1, y_pred.dtype)
        loss = y_true - y_pred
        loss = tf.nn.relu(self.margin - loss) ** 2
        loss = tf.reduce_sum(loss * mask, axis=-1)
        return loss
