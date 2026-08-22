"""Shared constants, GPU initialization, and artifact paths for SeismicML."""

from pathlib import Path

START_DATE: str = "2021-01-01"
END_DATE: str = "2024-01-01"
MIN_MAG: float = 2.5
GRID_SIZE_DEG: float = 2.0
TIME_STEP_DAYS: int = 7
TARGET_MAG: float = 4.5
BATCH_SIZE: int = 128
EPOCHS: int = 50
LEARNING_RATE: float = 2e-3
DROPOUT_RATE: float = 0.25

FEATURE_COLS: list[str] = [
    "lat_bin",
    "lon_bin",
    "event_count",
    "mean_depth",
    "mean_gap",
    "mean_sig",
]

ARTIFACTS_DIR: Path = Path("artifacts")


def ensure_artifacts_dir() -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS_DIR


def initialize_gpu() -> None:
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    print(f"GPUs detected: {len(gpus)}; policy: {tf.keras.mixed_precision.global_policy().name}")
