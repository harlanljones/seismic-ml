"""W5 contract tests: mixed precision, layer alignment, shapes, tracker schema."""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import pytest
import tensorflow as tf

from src.config import DROPOUT_RATE, LEARNING_RATE
from src.model.arch import build_4070ti_model
from src.model.tracker import PartitionMetricTracker
from src.pipeline.datasets import make_tf_dataset


@pytest.fixture(autouse=True)
def _mixed_float16_policy():
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    yield
    tf.keras.mixed_precision.set_global_policy("float32")


def _synthetic_arrays(n: int = 512, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6)).astype(np.float32)
    y = rng.integers(0, 2, size=n).astype(np.int64)
    return X, y


def test_mixed_precision_contract():
    assert tf.keras.mixed_precision.global_policy().name == "mixed_float16"
    model = build_4070ti_model(6)

    head = model.layers[-1]
    assert isinstance(head, tf.keras.layers.Dense)
    assert head.dtype == "float32"

    assert model.loss == "binary_crossentropy"
    assert model.jit_compile is True

    compile_cfg = model._compile_config.config
    assert compile_cfg["jit_compile"] is True


def test_layer_alignment():
    model = build_4070ti_model(6)
    dense_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense)]
    hidden = dense_layers[:-1]
    head = dense_layers[-1]

    units = [l.units for l in hidden]
    assert units == [128, 64, 64]
    assert all(u % 64 == 0 for u in units)
    assert head.units == 1
    assert head.activation is tf.keras.activations.sigmoid

    dropout = [
        l for l in model.layers if isinstance(l, tf.keras.layers.Dropout)
    ]
    assert dropout[0].rate == DROPOUT_RATE


def test_forward_shape():
    model = build_4070ti_model(6)
    X = np.random.default_rng(1).normal(size=(8, 6)).astype(np.float32)
    out = model.predict(X, verbose=0)
    assert out.shape == (8, 1)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_tracker_records_schema():
    X, y = _synthetic_arrays(n=512)
    split = 384
    train_ds = make_tf_dataset(X[:split], y[:split], is_train=True)
    val_ds = make_tf_dataset(X[split:], y[split:])
    test_ds = make_tf_dataset(X[split:], y[split:])

    model = build_4070ti_model(6)
    tracker = PartitionMetricTracker(test_dataset=test_ds)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=2,
        callbacks=[tracker],
        verbose=0,
    )

    expected_keys = {
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
        "test_loss",
        "test_acc",
    }
    assert set(tracker.records.keys()) == expected_keys
    lengths = {len(v) for v in tracker.records.values()}
    assert lengths == {2}
    for values in tracker.records.values():
        for v in values:
            assert isinstance(v, float)
            assert np.isfinite(v)


def test_optimizer_learning_rate():
    model = build_4070ti_model(6)
    lr = model.optimizer.learning_rate
    assert float(tf.keras.backend.get_value(lr)) == pytest.approx(LEARNING_RATE)
