import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datasets import Dset_pairs, merge_datasets, Config, experience_replay


def visualize_images(
    dataset: tf.data.Dataset, class_names: dict[int, str], cmap="gray", title="D1 train"
) -> None:
    images, labels = next(dataset.shuffle(1000).take(1).as_numpy_iterator())

    fig, ax = plt.subplots(3, 3, figsize=(8, 8))
    for i in range(9):
        row, col = i // 3, i % 3
        ax[row, col].imshow(images[i], cmap=cmap)
        ax[row, col].set_title(class_names.get(labels[i], f"Class {labels[i]}"))
        ax[row, col].axis("off")

    fig.tight_layout()
    fig.suptitle(title, fontsize=16)
    fig.subplots_adjust(top=0.85)
    plt.show()


def plot_class_distribution(
    dataset: tf.data.Dataset, num_classes: int, title: str = "D1 train"
) -> None:
    class_counts = np.zeros(num_classes, dtype=int)

    for _, labels in dataset.as_numpy_iterator():
        labels = labels.flatten()  # Ensure it's a 1D array
        unique, counts = np.unique(labels, return_counts=True)
        class_counts[unique] += counts  # Efficiently update class counts

    plt.figure(figsize=(8, 5))
    plt.bar(
        range(num_classes),
        class_counts,
        tick_label=[str(i) for i in range(num_classes)],
        color="skyblue",
    )
    plt.xlabel("Class Index")
    plt.ylabel("Frequency")
    plt.title(f"Class Distribution in Dataset {title}", fontsize=16)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


if __name__ == "__main__":
    # Fmnist-Mnist
    config = Config()
    d1_train, d2_train, d1_test, d2_test, class_names = merge_datasets(config)

    visualize_images(d1_train, class_names, title="D1 train")
    visualize_images(d2_train, class_names, title="D2 train")

    visualize_images(d1_test, class_names, title="D1 test")
    visualize_images(d2_test, class_names, title="D2 test")

    plot_class_distribution(d1_train, num_classes=len(class_names), title="D1 train")
    plot_class_distribution(d2_train, num_classes=len(class_names), title="D2 train")

    plot_class_distribution(d1_test, num_classes=len(class_names), title="D1 test")
    plot_class_distribution(d2_test, num_classes=len(class_names), title="D2 test")

    # Half-mnist
    config = Config(dset_pair=Dset_pairs.half_mnist)
    d1_train, d2_train, d1_test, d2_test, class_names = merge_datasets(config)

    visualize_images(d1_train, class_names, title="D1 train")
    visualize_images(d2_train, class_names, title="D2 train")

    visualize_images(d1_test, class_names, title="D1 test")
    visualize_images(d2_test, class_names, title="D2 test")

    plot_class_distribution(d1_train, num_classes=len(class_names), title="D1 train")
    plot_class_distribution(d2_train, num_classes=len(class_names), title="D2 train")

    plot_class_distribution(d1_test, num_classes=len(class_names), title="D1 test")
    plot_class_distribution(d2_test, num_classes=len(class_names), title="D2 test")

    # Half-cifar10
    config = Config(dset_pair=Dset_pairs.half_cifar10)
    d1_train, d2_train, d1_test, d2_test, class_names = merge_datasets(config)

    visualize_images(d1_train, class_names, title="D1 train")
    visualize_images(d2_train, class_names, title="D2 train")

    visualize_images(d1_test, class_names, title="D1 test")
    visualize_images(d2_test, class_names, title="D2 test")

    plot_class_distribution(d1_train, num_classes=len(class_names), title="D1 train")
    plot_class_distribution(d2_train, num_classes=len(class_names), title="D2 train")

    plot_class_distribution(d1_test, num_classes=len(class_names), title="D1 test")
    plot_class_distribution(d2_test, num_classes=len(class_names), title="D2 test")

    # Half-cifar100
    config = Config(dset_pair=Dset_pairs.half_cifar100)
    d1_train, d2_train, d1_test, d2_test, class_names = merge_datasets(config)

    visualize_images(d1_train, class_names, title="D1 train")
    visualize_images(d2_train, class_names, title="D2 train")

    visualize_images(d1_test, class_names, title="D1 test")
    visualize_images(d2_test, class_names, title="D2 test")

    plot_class_distribution(d1_train, num_classes=len(class_names), title="D1 train")
    plot_class_distribution(d2_train, num_classes=len(class_names), title="D2 train")

    plot_class_distribution(d1_test, num_classes=len(class_names), title="D1 test")
    plot_class_distribution(d2_test, num_classes=len(class_names), title="D2 test")

    # Try experience replay with Half-cifar100
    er_train = experience_replay(d2_train, d1_train, 0.2)
    visualize_images(er_train, class_names, title="D2 test")
    plot_class_distribution(
        er_train, num_classes=len(class_names), title="Experience replay"
    )
