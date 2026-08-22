"""PartitionMetricTracker callback (workstream W5).

Contract (ROADMAP.md §4.6): on every ``on_epoch_end``, append train/val
metrics from ``logs`` and holdout-test metrics from
``self.model.evaluate(self.test_dataset, verbose=0)`` to ``self.records``
under the frozen six-key schema.
"""

import tensorflow as tf


class PartitionMetricTracker(tf.keras.callbacks.Callback):
    """Tracks training, validation, and holdout test metrics at each epoch."""

    def __init__(self, test_dataset: tf.data.Dataset) -> None:
        super().__init__()
        self.test_dataset = test_dataset
        self.records: dict[str, list[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "test_loss": [],
            "test_acc": [],
        }

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        logs = logs or {}
        self.records["train_loss"].append(float(logs.get("loss")))
        self.records["train_acc"].append(float(logs.get("accuracy")))
        self.records["val_loss"].append(float(logs.get("val_loss")))
        self.records["val_acc"].append(float(logs.get("val_accuracy")))

        results = self.model.evaluate(self.test_dataset, verbose=0)
        if isinstance(results, float):
            t_loss, t_acc = results, float("nan")
        else:
            t_loss, t_acc = float(results[0]), float(results[1])
        self.records["test_loss"].append(t_loss)
        self.records["test_acc"].append(t_acc)
