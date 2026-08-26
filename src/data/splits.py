"""Chronological walk-forward splitting and leakage-free scaling (W3)."""

import numpy as np
from sklearn.preprocessing import StandardScaler


def walk_forward_split(
    n_samples: int,
    train_end: float = 0.70,
    val_end: float = 0.85,
) -> tuple[slice, slice, slice]:
    """Chronological walk-forward index slices. No shuffling ever.

    Splits the first ``n_samples`` indices into contiguous train/val/test
    partitions strictly by position:

    - train: ``[0, train_end * n_samples)``
    - val:   ``[train_end * n_samples, val_end * n_samples)``
    - test:  ``[val_end * n_samples, n_samples)``

    Defaults ``train_end=0.70`` and ``val_end=0.85`` reproduce the original
    70/15/15 behavior (0–70%, 70–85%, 85–100%). Retraining can pass shifted
    ``train_end`` / ``val_end`` (e.g. ``0.90`` / ``0.95``) to use a later
    chronological window without altering existing callers.
    """
    i1 = int(n_samples * train_end)
    i2 = int(n_samples * val_end)
    return (slice(0, i1), slice(i1, i2), slice(i2, n_samples))


def scale_splits(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Fit StandardScaler on train ONLY; transform val/test.

    Returns ``(X_train_scaled, X_val_scaled, X_test_scaled, scaler)``.
    The fitted scaler is the fourth element of the tuple (frozen API).
    """
    scaler = StandardScaler().fit(X_train)
    return (
        scaler.transform(X_train),
        scaler.transform(X_val),
        scaler.transform(X_test),
        scaler,
    )
