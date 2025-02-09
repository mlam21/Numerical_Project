import os
import random
import tensorflow as tf
import json
import numpy as np
from datasets import Config, merge_datasets
from pathlib import Path
from trainer import CNNModel
from dataclasses import dataclass, asdict


# To ensure reproductiblity
def set_seed(seed: int):
    random.seed(seed)

    np.random.seed(seed)

    tf.random.set_seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"

    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)


# There's 2 trainings, the first one in 100/0 proportion, loss and accuracy
# are with respect to validation set
@dataclass
class Results:
    save_dir: Path
    config: Config

    training_1_d1_accuracy: float | None = None
    training_1_d1_loss: float | None = None
    training_1_d2_accuracy: float | None = None
    training_1_d2_loss: float | None = None

    training_2_d1_accuracy: float | None = None
    training_2_d1_loss: float | None = None
    training_2_d2_accuracy: float | None = None
    training_2_d2_loss: float | None = None

    # As I am using config that has a tuple of an enum in its field turning it
    # into a dict causes issue so I just convert it to a regular tuple of str before saving it
    def save(self) -> None:
        save_output = results.save_dir / "results.json"
        self.config.dset_pair = tuple(item.value for item in self.config.dset_pair)  # type: ignore
        self.save_dir = self.save_dir.as_posix()  # type: ignore
        with open(save_output, "w") as file:
            json.dump(asdict(results), file, indent=1)


# Load config, dsets and and the model
config = Config()
set_seed(config.seed_value)
d1_train, d2_train, d1_test, d2_test, merged_class_names = merge_datasets(config)

results = Results(config=config, save_dir=Path("data/mnist-fmnist"))
os.makedirs(results.save_dir, exist_ok=True)

cnn = CNNModel(
    config.input_shape,
    len(merged_class_names),
)
cnn.build_model(config)

# Train the model on the D1 (with D1 val)
cnn.train(train_dset=d1_train, val_dset=d1_test, epochs=config.epochs)

# Save the model, nomenclature for weights D1-D2-proportion in the dset
cnn.save_weights(results.save_dir / "mnist-fmnist-100-0.weights.h5")
cnn.plot_training_curves(save_path=results.save_dir / "training_1_curves.png")

# Test model on D1-test and D2-test, obviously catastrophic performance on D2
d1_test_loss, d1_test_accuracy = cnn.test_model(test_data=d1_test)
d2_test_loss, d2_test_accuracy = cnn.test_model(test_data=d2_test)

results.training_1_d1_accuracy = d1_test_accuracy
results.training_1_d1_loss = d1_test_loss
results.training_1_d2_accuracy = d2_test_accuracy
results.training_1_d2_loss = d2_test_loss

# Retrain it on D2 this time (with D2-val)
cnn.train(train_dset=d2_train, val_dset=d2_test, epochs=config.epochs)

# Save the model, nomenclature for weights D1-D2-D1-proportion in the dset
cnn.save_weights(results.save_dir / "mnist-fmnist-0-100.weights.h5")
cnn.plot_training_curves(save_path=results.save_dir / "training_2_curves.png")

# Test model on D1-test and D2-test
d1_test_loss, d1_test_accuracy = cnn.test_model(test_data=d1_test)
d2_test_loss, d2_test_accuracy = cnn.test_model(test_data=d2_test)

results.training_2_d1_accuracy = d1_test_accuracy
results.training_2_d1_loss = d1_test_loss
results.training_2_d2_accuracy = d2_test_accuracy
results.training_2_d2_loss = d2_test_loss

# Saving results
results.save()
