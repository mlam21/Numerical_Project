import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datasets import Dset_pairs, merge_datasets, Config
import rich


def visualize_images(
    dataset: tf.data.Dataset, class_names: dict[int, str], cmap="gray"
) -> None:
    dataset_iter = iter(dataset.take(1))

    # Remember that images are loaded by batch
    images, labels = next(dataset_iter)
    images = images.numpy()
    labels = labels.numpy() if isinstance(labels, tf.Tensor) else labels

    fig, ax = plt.subplots(3, 3)
    for i in range(9):
        row, col = i // 3, i % 3
        ax[row, col].imshow(images[i], cmap=cmap)
        ax[row, col].set_title(class_names[labels[i]])

        ax[row, col].axis("off")

    fig.tight_layout()
    plt.show()


def plot_class_distribution(dataset: tf.data.Dataset, num_classes: int) -> None:
    class_counts = np.zeros(num_classes, dtype=int)
    total = 0

    for _, labels in dataset:
        labels = labels.numpy().flatten()
        for label in labels:
            class_counts[label] += 1
            total += 1

    rich.print(total)

    plt.figure(figsize=(8, 5))
    plt.bar(
        range(num_classes),
        class_counts,
        tick_label=[str(i) for i in range(num_classes)],
        color="skyblue",
    )
    plt.xlabel("Class Index")
    plt.ylabel("Frequency")
    plt.title("Class Distribution in Dataset")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


if __name__ == "__main__":
    config = Config()
    d1_train, d2_train, d1_test, d2_test, class_names = merge_datasets(config)

    visualize_images(d1_train, class_names)
    visualize_images(d2_train, class_names)

    visualize_images(d1_test, class_names)
    visualize_images(d2_test, class_names)

    plot_class_distribution(d1_train, num_classes=len(class_names))
    plot_class_distribution(d2_train, num_classes=len(class_names))

    plot_class_distribution(d1_test, num_classes=len(class_names))
    plot_class_distribution(d2_test, num_classes=len(class_names))
