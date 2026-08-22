"""Chronological walk-forward splitting and leakage-free scaling (W3)."""

import numpy as np
from sklearn.preprocessing import StandardScaler


def walk_forward_split(n_samples: int) -> tuple[slice, slice, slice]:
    """Chronological 70/15/15 index slices. No shuffling ever."""
    return (
        slice(0, int(n_samples * 0.70)),
        slice(int(n_samples * 0.70), int(n_samples * 0.85)),
        slice(int(n_samples * 0.85), n_samples),
    )


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
