import tensorflow as tf
import tensorflow_datasets as tfds
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable
import rich


# Use an enum for datasets to avoid mistyping
class Dsets(str, Enum):
    mnist = "mnist"
    fmnist = "fashion_mnist"
    cifar10 = "cifar10"
    cifar100 = "cifar100"


# Use an enum for D1,D2 pair to avoid mistyping
class Dset_pairs(tuple[Dsets, Dsets], Enum):
    mnist_fmnist = (Dsets.mnist, Dsets.fmnist)
    half_mnist = (Dsets.mnist, Dsets.mnist)
    half_cifar10 = (Dsets.cifar10, Dsets.cifar10)
    half_cifar100 = (Dsets.cifar100, Dsets.cifar100)


# Just a regular dataclass to handle the config logic
@dataclass
class Config:
    batch_size: int = 64
    epochs: int = 5
    loss: str = "sparse_categorical_crossentropy"
    optimizer: str = "adam"
    metrics: list[str] = field(default_factory=lambda: ["accuracy"])

    dset_pair: Dset_pairs = Dset_pairs.mnist_fmnist
    p: float = 1.0

    @property
    def input_shape(self) -> tuple[int, int, int]:
        if self.dset_pair in [
            Dset_pairs.half_cifar10,
            Dset_pairs.half_cifar100,
        ]:
            return (32, 32, 3)

        return (28, 28, 1)

    @property
    def half_dset(self) -> bool:
        return self.dset_pair in [
            Dset_pairs.half_cifar10,
            Dset_pairs.half_cifar100,
            Dset_pairs.half_mnist,
        ]


# Define loading/preprocess functions for each available datasets


# To allow for [0, 1] normalization
def preprocess(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def load_mnist() -> tuple[tf.data.Dataset, tf.data.Dataset, int, dict[int, str]]:
    (mnist_train, mnist_test), mnist_info = tfds.load(
        "mnist", split=["train", "test"], as_supervised=True, with_info=True
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


def load_fmnist() -> tuple[tf.data.Dataset, tf.data.Dataset, int, dict[int, str]]:
    (fmnist_train, fmnist_test), fmnist_info = tfds.load(
        "fashion_mnist", split=["train", "test"], as_supervised=True, with_info=True
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


def load_cifar10() -> tuple[tf.data.Dataset, tf.data.Dataset, int, dict[int, str]]:
    (cifar10_train, cifar10_test), cifar10_info = tfds.load(
        "cifar10", split=["train", "test"], as_supervised=True, with_info=True
    )

    class_names = {
        i: cifar10_info.features["label"].int2str(i)
        for i in range(cifar10_info.features["label"].num_classes)
    }
    return (
        cifar10_train,
        cifar10_test,
        cifar10_info.features["label"].num_classes,
        class_names,
    )


def load_cifar100() -> tuple[tf.data.Dataset, tf.data.Dataset, int, dict[int, str]]:
    (cifar100_train, cifar100_test), cifar100_info = tfds.load(
        "cifar100", split=["train", "test"], as_supervised=True, with_info=True
    )

    class_names = {
        i: cifar100_info.features["label"].int2str(i)
        for i in range(cifar100_info.features["label"].num_classes)
    }
    return (
        cifar100_train,
        cifar100_test,
        cifar100_info.features["label"].num_classes,
        class_names,
    )


LOAD_MAPPER: dict[
    Dsets : Callable[None, tuple[tf.data.Dataset, tf.data.Dataset, int, dict[int, str]]]
] = {
    Dsets.mnist: load_mnist,
    Dsets.fmnist: load_fmnist,
    Dsets.cifar10: load_cifar10,
    Dsets.cifar100: load_cifar100,
}


def split_dset_in_half(
    train_dset: tf.data.Dataset,
    test_dset: tf.data.Dataset,
    class_names: dict[int, str],
    config: Config,
) -> tuple[
    tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, dict[int, str]
]:
    num_classes = len(class_names)
    D1_classes = tf.constant(list(range(num_classes // 2)), dtype=tf.int64)
    D2_classes = tf.constant(list(range(num_classes // 2, num_classes)), dtype=tf.int64)

    # This is really annoying but we relied on the filter method on TF that is done dynamically
    # which means that the cardinality is in fact unknown where we splitted dataset into 2 sub-datasets,
    # in that situation we need to manually compute the size of the dataset by iterating over it...
    def reset_cardinality(dset):
        cardinality = dset.reduce(0, lambda x, _: x + 1).numpy()
        return dset.apply(tf.data.experimental.assert_cardinality(cardinality))

    D1_train = (
        reset_cardinality(
            train_dset.filter(
                lambda _, label: tf.math.reduce_any(tf.equal(label, D1_classes))
            )
        )
        .map(preprocess)
        .batch(config.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    D2_train = (
        reset_cardinality(
            train_dset.filter(
                lambda _, label: tf.math.reduce_any(tf.equal(label, D2_classes))
            )
        )
        .map(preprocess)
        .batch(config.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    D1_test = (
        reset_cardinality(
            test_dset.filter(
                lambda _, label: tf.math.reduce_any(tf.equal(label, D1_classes))
            )
        )
        .map(preprocess)
        .batch(config.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    D2_test = (
        reset_cardinality(
            test_dset.filter(
                lambda _, label: tf.math.reduce_any(tf.equal(label, D2_classes))
            )
        )
        .map(preprocess)
        .batch(config.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return (
        D1_train,
        D2_train,
        D1_test,
        D2_test,
        class_names,
    )


# Function to merge labels from a dataset pairs by shifting their index
def merge_datasets(
    config: Config,
) -> tuple[
    tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, dict[int, str]
]:
    global LOAD_MAPPER

    D1_train, D1_test, D1_classes, D1_names = LOAD_MAPPER[config.dset_pair[0]]()
    if config.half_dset:
        return split_dset_in_half(D1_train, D1_test, D1_names, config)

    D2_train, D2_test, _, D2_names = LOAD_MAPPER[config.dset_pair[1]]()

    offset_D1_label = lambda img, lbl: (img, lbl)
    offset_D2_label = lambda img, lbl: (
        img,
        lbl + D1_classes,
    )  # Shift labels

    D1_train = (
        D1_train.map(preprocess)
        .batch(config.batch_size)
        .prefetch(tf.data.AUTOTUNE)
        .map(offset_D1_label)
    )
    D2_train = (
        D2_train.map(preprocess)
        .batch(config.batch_size)
        .prefetch(tf.data.AUTOTUNE)
        .map(offset_D2_label)
    )

    D1_test = (
        D1_test.map(preprocess)
        .batch(config.batch_size)
        .prefetch(tf.data.AUTOTUNE)
        .map(offset_D1_label)
    )
    D2_test = (
        D2_test.map(preprocess)
        .batch(config.batch_size)
        .prefetch(tf.data.AUTOTUNE)
        .map(offset_D2_label)
    )

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


# This function is used for experience replay and just create a new dataset by adding a fraction
# of the other dataset in the first one. We assume that the same batch size is used for the two dsets
def experience_replay(
    new_dset: tf.data.Dataset, old_dset: tf.data.Dataset, p: float
) -> tf.data.Dataset:
    cardinality = old_dset.cardinality().numpy()

    sample_size = int(p * cardinality)
    samples = old_dset.take(sample_size)

    er_dset = new_dset.concatenate(samples)

    return er_dset.shuffle(buffer_size=er_dset.cardinality().numpy()).prefetch(
        tf.data.AUTOTUNE
    )


if __name__ == "__main__":
    # Half cifar10
    config = Config(dset_pair=Dset_pairs.half_cifar10)
    d1_train, d2_train, d1_test, d2_test, merged_class_names = merge_datasets(config)
    for images, labels in d1_train.take(1):
        rich.print(images.shape, labels.shape)

    # Half cifar100
    config = Config(dset_pair=Dset_pairs.half_cifar100)
    d1_train, d2_train, d1_test, d2_test, merged_class_names = merge_datasets(config)
    for images, labels in d1_train.take(1):
        rich.print(images.shape, labels.shape)

    # Half mnist
    config = Config(dset_pair=Dset_pairs.half_mnist)
    d1_train, d2_train, d1_test, d2_test, merged_class_names = merge_datasets(config)
    for images, labels in d1_train.take(1):
        rich.print(images.shape, labels.shape)

    # Mnsit_fmnist
    config = Config(dset_pair=Dset_pairs.mnist_fmnist)
    d1_train, d2_train, d1_test, d2_test, merged_class_names = merge_datasets(config)
    for images, labels in d1_train.take(1):
        rich.print(images.shape, labels.shape)

    er_train = experience_replay(d2_train, d1_train, 0.2)
    for images, labels in er_train.take(1):
        rich.print(images.shape, labels.shape)
