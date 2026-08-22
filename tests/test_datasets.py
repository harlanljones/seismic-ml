"""Tests for src.pipeline.datasets (workstream W4). CPU-only, synthetic data."""

import math
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf

from src.config import BATCH_SIZE
from src.pipeline.datasets import make_tf_dataset


def _synthetic(n: int, f: int = 6) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    features = rng.normal(size=(n, f))
    labels = rng.integers(0, 2, size=(n,))
    return features, labels


def test_batching_partial_final_batch() -> None:
    n = 300  # not divisible by BATCH_SIZE
    features, labels = _synthetic(n)
    ds = make_tf_dataset(features, labels)
    expected = math.ceil(n / BATCH_SIZE)
    assert len(ds) == expected
    batches = list(ds)
    assert len(batches) == expected
    assert int(sum(b.shape[0] for b, _ in batches)) == n


def test_element_shapes_and_cardinality() -> None:
    n, f = 256, 6
    features, labels = _synthetic(n, f)
    ds = make_tf_dataset(features, labels)
    feat_spec, label_spec = ds.element_spec
    assert feat_spec.shape == (None, f)
    assert label_spec.shape == (None,)
    assert ds.cardinality() != tf.data.INFINITE_CARDINALITY
    for x, y in ds.take(1):
        assert x.shape[1] == f
        assert y.shape == (x.shape[0],)


def test_shuffle_only_when_is_train_and_label_multiset_preserved() -> None:
    n = 500
    features, labels = _synthetic(n)
    train_ds = make_tf_dataset(features, labels, is_train=True)
    eval_ds = make_tf_dataset(features, labels, is_train=False)

    train_labels = np.concatenate([y.numpy() for _, y in train_ds])
    np.testing.assert_array_equal(np.sort(train_labels), np.sort(labels))

    eval_labels = np.concatenate([y.numpy() for _, y in eval_ds])
    np.testing.assert_array_equal(eval_labels, labels)


def test_eval_dataset_preserves_label_order_through_batching() -> None:
    n = 300
    features, labels = _synthetic(n)
    ds = make_tf_dataset(features, labels, is_train=False)
    collected = []
    for batch_features, batch_labels in ds:
        assert batch_labels.dtype == labels.dtype
        assert batch_features.dtype == features.dtype
        collected.append(batch_labels.numpy())
    np.testing.assert_array_equal(np.concatenate(collected), labels)


def test_both_variants_iterate_without_error() -> None:
    features, labels = _synthetic(128 * 2 + 7)
    for is_train in (False, True):
        ds = make_tf_dataset(features, labels, is_train=is_train)
        for _ in ds:
            pass
