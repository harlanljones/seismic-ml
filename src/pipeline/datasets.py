"""tf.data input pipelines for SeismicML (workstream W4)."""

import numpy as np
import tensorflow as tf

from src.config import BATCH_SIZE


def make_tf_dataset(
    features: np.ndarray, labels: np.ndarray, is_train: bool = False
) -> tf.data.Dataset:
    """Build a batched, prefetched tf.data.Dataset from feature/label arrays.

    Contract (ROADMAP.md §4.4):
    - ``from_tensor_slices((features, labels))``
    - if ``is_train``: ``shuffle(buffer_size=4096)``
    - always: ``batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)``
    - Yields shapes ``(None, F)`` float features and ``(None,)`` labels,
      with dtypes as inferred by ``from_tensor_slices``.
    """
    ds = tf.data.Dataset.from_tensor_slices((features, labels))
    if is_train:
        ds = ds.shuffle(buffer_size=4096)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
