# import tensorflow as tf
# import layer
# import loss
# import tensorflow_datasets as tfds
#
# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Input((28, 28, 1)))
# model.add(tf.keras.layers.Conv2D(256, 9))
# model.add(tf.keras.layers.Activation('relu'))
# model.add(layer.PrimaryCapsule(32, 8, 9, 2, activation='dr'))
# model.add(layer.DRCapsule(10, 16, activation='dr'))
# model.add(tf.keras.layers.Lambda(lambda inputs: tf.norm(inputs, 2, -1)))
# model.compile(loss=loss.MarginLoss(), metrics='categorical_accuracy', optimizer='adam', run_eagerly=True)
# model.summary()
#
# (ds_train, ds_test), ds_info = tfds.load(
#     'mnist',
#     split=['train', 'test'],
#     shuffle_files=True,
#     as_supervised=True,
#     with_info=True,
# )
#
#
# def normalize_img(image, label):
#     """Normalizes images: `uint8` -> `float32`."""
#     return tf.cast(image, tf.float32) / 255., label
#
#
# ds_train = ds_train.map(
#     normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
# ds_train = ds_train.cache()
# ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
# ds_train = ds_train.batch(128)
# ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
#
# ds_test = ds_test.map(
#     normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
# ds_test = ds_test.batch(128)
# ds_test = ds_test.cache()
# ds_test = ds_test.prefetch(tf.data.AUTOTUNE)
#
# model.fit(
#     ds_train,
#     epochs=6,
#     validation_data=ds_test,
# )

import tensorflow as tf

cifar = tf.keras.datasets.cifar100
(x_train, y_train), (x_test, y_test) = cifar.load_data()
model = tf.keras.applications.ResNet50(
    include_top=True,
    weights=None,
    input_shape=(32, 32, 3),
    classes=100,)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
model.fit(x_train, y_train, epochs=5, batch_size=64)