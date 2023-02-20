import layer
import loss
import tensorflow as tf
import tensorflow_datasets as tfds

tf.random.set_seed(0)

# tf.keras.mixed_precision.set_global_policy(
#     tf.keras.mixed_precision.Policy("mixed_float16")
# )

model = tf.keras.Sequential()
model.add(tf.keras.layers.Input((28, 28, 1)))
model.add(tf.keras.layers.Conv2D(256, 9, activation="relu"))
model.add(layer.PrimaryCapsule(32, 8, 9, 2, activation="dr"))
model.add(layer.DRCapsule(10, 16, activation="dr"))
model.add(layer.Length())
model.compile(
    loss=loss.MarginLoss(),
    metrics="accuracy",
    # optimizer=tf.keras.optimizers.Adam(
    #     tf.keras.optimizers.schedules.ExponentialDecay(
    #         1e-3, 60000 // 32 * 100, 0.98
    #     )
    # ),
    optimizer="adam",
)
model.summary()

(ds_train, ds_test), ds_info = tfds.load(
    "mnist",
    split=["train", "test"],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255.0, label


ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
ds_train = ds_train.batch(32)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

model.fit(
    ds_train,
    epochs=100,
    validation_data=ds_test,
)
