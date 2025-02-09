import tensorflow as tf
import tensorflow_datasets as tfds
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


# Use an enum for datasets  to avoid mistyping
class Dsets(str, Enum):
    mnist = "mnist"
    fmnist = "fashion_mnist"


# Use an enum for D1,D2 pair to avoid mistyping
class Dset_pairs(tuple[Dsets, Dsets], Enum):
    mnist_fmnist = (Dsets.mnist, Dsets.fmnist)


# Just a regular dataclass to handle the config logic
@dataclass
class Config:
    buffer_size: int = 10000
    seed_value: int = 1
    batch_size: int = 64
    epochs: int = 5
    loss: str = "sparse_categorical_crossentropy"
    optimizer: str = "adam"
    metrics: list[str] = field(default_factory=lambda: ["accuracy"])

    dset_pair: Dset_pairs = Dset_pairs.mnist_fmnist
    input_shape: tuple[int, int, int] = field(default_factory=lambda: (28, 28, 1))


# Define loading/preprocess functions for each available datasets
def load_mnist(
    config: Config,
) -> tuple[tf.data.Dataset, tf.data.Dataset, int, dict[int, str]]:
    (mnist_train, mnist_test), mnist_info = tfds.load(
        "mnist", split=["train", "test"], as_supervised=True, with_info=True
    )

    # To allow for [0, 1] normalization
    def preprocess(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    mnist_train = (
        mnist_train.map(preprocess).batch(config.batch_size).prefetch(tf.data.AUTOTUNE)
    )
    mnist_test = (
        mnist_test.map(preprocess).batch(config.batch_size).prefetch(tf.data.AUTOTUNE)
    )

    class_names = {
        i: mnist_info.features["label"].int2str(i)
        for i in range(mnist_info.features["label"].num_classes)
    }
    return (
        mnist_train,
        mnist_test,
        mnist_info.features["label"].num_classes,
        class_names,
    )


def load_fmnist(
    config: Config,
) -> tuple[tf.data.Dataset, tf.data.Dataset, int, dict[int, str]]:
    (fmnist_train, fmnist_test), fmnist_info = tfds.load(
        "fashion_mnist", split=["train", "test"], as_supervised=True, with_info=True
    )

    # To allow for [0, 1] normalization
    def preprocess(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    fmnist_train = (
        fmnist_train.map(preprocess).batch(config.batch_size).prefetch(tf.data.AUTOTUNE)
    )
    fmnist_test = (
        fmnist_test.map(preprocess).batch(config.batch_size).prefetch(tf.data.AUTOTUNE)
    )
    class_names = {
        i: fmnist_info.features["label"].int2str(i)
        for i in range(fmnist_info.features["label"].num_classes)
    }
    return (
        fmnist_train,
        fmnist_test,
        fmnist_info.features["label"].num_classes,
        class_names,
    )


LOAD_MAPPER: dict[
    Dsets : Callable[
        Config, tuple[tf.data.Dataset, tf.data.Dataset, int, dict[int, str]]
    ]
] = {
    Dsets.mnist: load_mnist,
    Dsets.fmnist: load_fmnist,
}


# Function to merge labels from a dataset pairs by shifting their index
def merge_datasets(
    config: Config,
) -> tuple[
    tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, dict[int, str]
]:
    global LOAD_MAPPER

    D1_train, D1_test, D1_classes, D1_names = LOAD_MAPPER[config.dset_pair[0]](config)
    D2_train, D2_test, _, D2_names = LOAD_MAPPER[config.dset_pair[1]](config)

    offset_D1_label = lambda img, lbl: (img, lbl)
    offset_D2_label = lambda img, lbl: (
        img,
        lbl + D1_classes,
    )  # Shift labels

    D1_train = D1_train.map(offset_D1_label)
    D2_train = D2_train.map(offset_D2_label)

    D1_test = D1_test.map(offset_D1_label)
    D2_test = D2_test.map(offset_D2_label)

    merged_class_names = {
        **D1_names,
        **{i + D1_classes: name for i, name in D2_names.items()},
    }

    return (
        D1_train,
        D2_train,
        D1_test,
        D2_test,
        merged_class_names,
    )


if __name__ == "__main__":
    config = Config()
    d1_train, _, d2_train, _, merged_class_names = merge_datasets(config)

    for images, labels in d1_train.take(100):
        print(images.shape, labels.shape)
