from datasets import Config, merge_datasets
from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models


class CNNModel:
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        num_classes: int,
    ) -> None:
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model: tf.keras.Model | None = None
        self.history: tf.keras.callbacks.History = None

    def build_model(self, config: Config) -> None:
        self.model = models.Sequential(
            [
                layers.InputLayer(input_shape=self.input_shape),
                layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
                layers.MaxPooling2D((2, 2)),
                layers.Flatten(),
                layers.Dense(128, activation="relu"),
                layers.Dense(self.num_classes, activation="softmax"),
            ]
        )
        self.model.compile(
            optimizer=config.optimizer,
            loss=config.loss,
            metrics=config.metrics,
        )

    def save_weights(self, filepath: Path) -> None:
        if self.model:
            self.model.save_weights(filepath)
        else:
            raise ValueError("Model is not built. Call 'build_model()' first.")

    def load_weights(self, filepath: Path) -> None:
        if self.model:
            self.model.load_weights(filepath)
        else:
            raise ValueError("Model is not built. Call 'build_model()' first.")

    def plot_training_curves(self, save_path: Path | None = None) -> None:
        if self.history is None:
            raise ValueError("Please train the model before!")

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(self.history.history["accuracy"], label="Train Accuracy")
        plt.plot(self.history.history["val_accuracy"], label="Val Accuracy")
        plt.title("Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.history.history["loss"], label="Train Loss")
        plt.plot(self.history.history["val_loss"], label="Val Loss")
        plt.title("Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

        if save_path is not None:
            plt.savefig(save_path)

        plt.show()

    def train(
        self, train_dset: tf.data.Dataset, val_dset: tf.data.Dataset, epochs: int
    ) -> None:
        if self.model:
            self.history = self.model.fit(
                train_dset, epochs=epochs, validation_data=val_dset
            )
        else:
            raise ValueError("Model is not built. Call 'build_model()' first.")

    def test_model(self, test_data: tf.data.Dataset) -> tuple[float, float]:
        if self.model:
            test_loss, test_accuracy = self.model.evaluate(test_data)
            print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
            return test_loss, test_accuracy
        else:
            raise ValueError("Model is not built. Call 'build_model()' first.")


if __name__ == "__main__":
    config = Config()
    d1_train, d2_train, d1_test, d2_test, class_names = merge_datasets(config)

    cnn = CNNModel(config.input_shape, num_classes=len(class_names))
    cnn.build_model(config)

    cnn.train(train_dset=d1_train, val_dset=d1_test, epochs=config.epochs)
    cnn.plot_training_curves()
