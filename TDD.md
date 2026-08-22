# Technical Design Document: ML Seismic Prediction Pipeline (SeismicML)

> **IMPLEMENTATION STATUS (2026-08-21):** Built and verified per this design. Acceptance results in `ROADMAP.md` §5; deliverables in `artifacts/` (model weights, history, convergence curves, overlay map). One deviation from the reference code below: `fetch_seismic_catalog` paginates around the USGS 20k-event API cap (see `ROADMAP.md` §4.1 amendment).

**Target Hardware:** NVIDIA GeForce RTX 4070 Ti (12GB GDDR6X, Ada Lovelace, Compute Capability 8.9)

**Target Environment:** TensorFlow 2.x / Keras, CUDA 12.x, cuDNN 8.9+, WSL2 (Ubuntu 22.04+) / Linux

**Primary Metrics:** Train / Validation / Holdout Test Accuracy & Loss

**Spatial Deliverable:** Interactive Geospatial Overlay Map (Leaflet / Folium)

---

### 1. Hardware Architecture & Runtime Configuration

```
┌────────────────────────────────────────────────────────────────────────┐
│                   Host Environment: WSL2 / Ubuntu                      │
│  NVIDIA Driver 550+ ──> CUDA 12.x ──> cuDNN 8.9 ──> TensorFlow 2.x    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
 ┌──────────────────────────────────┴───────────────────────────────────┐
 │               RTX 4070 Ti 12GB Hardware Configuration                │
 │  • Dynamic VRAM Growth: Enabled (Prevents initial 12GB lockup)       │
 │  • Compute Policy: mixed_float16 (4th-Gen Tensor Core acceleration)  │
 │  • Kernel Compilation: XLA JIT (jit_compile=True)                    │
 │  • Layer Alignment: Multiples of 64 (Dense-128 / Dense-64 / etc.)    │
 │  • Streaming Pipeline: tf.data.Dataset + AUTOTUNE Prefetching        │
 └──────────────────────────────────────────────────────────────────────┘

```

#### 1.1 Hardware Specifications & Acceleration Strategy

* **Compute Architecture:** 7,680 CUDA cores, 240 4th-Gen Tensor Cores.
* **Mixed Precision Policy (`mixed_float16`):** Intermediary hidden layers execute in FP16 to maximize Tensor Core throughput and halve VRAM bandwidth footprint. The final classification head enforces `dtype="float32"` to prevent gradient underflow during sigmoid probability calculation.
* **XLA Optimization:** Kernel fusion enabled via `jit_compile=True` inside `model.compile()`, eliminating GPU memory round-trip latency.
* **Memory Management:** Dynamic VRAM growth enabled at runtime to prevent host allocation crashes while retaining overhead for concurrent spatial mapping processes.

---

### 2. Problem Formulation & Feature Pipeline

#### 2.1 Problem Definition

The system models seismic forecasting as a spatiotemporal binary classification task. For a spatial grid cell $C_i$ over an observation window $T_{\text{obs}}$ ($30\text{ days}$), the network estimates the probability that an earthquake with magnitude $M \ge 4.5$ occurs in the subsequent target forecast window $T_{\text{target}}$ ($7\text{ days}$):

$$\hat{y} = P(M_{\text{max}} \ge 4.5 \mid X_{\text{precursor}})$$

#### 2.2 Feature Representation

* **Spatial & Sensor Geometry:** Centroid Latitude, Longitude, Mean Hypocenter Depth, Mean Azimuthal Gap (`gap`), Minimum Distance (`dmin`), Seismic Signal (`sig`).
* **Rate Dynamics:** Event count $N_{30\text{d}}$, $N_{7\text{d}}$, seismic acceleration ratio:

$$R = \frac{N_{7\text{d}}}{N_{30\text{d}} / 4.28}$$

* **Cumulative Seismic Energy:** Total scalar moment release:

$$M_0 = \sum 10^{1.5 M_i + 9.1}$$

* **Gutenberg-Richter $b$-value:** Estimated via Maximum Likelihood Estimation (MLE):

$$b = \frac{\log_{10}(e)}{\bar{M} - (M_{\text{min}} - \frac{\Delta M}{2})}$$

---

### 3. Phased Implementation Roadmap

```
Phase 1: Ingestion & Temporal Feature Pipeline (USGS FDSN REST API, Spatial Bins, Walk-Forward Splitting)
Phase 2: Hardware-Optimized Keras Pipeline (Mixed Precision, Tensor Core Layering, Real-time Val/Test Metrics)
Phase 3: Multi-Partition Evaluation & Metric Graphing (Loss & Accuracy Convergence across Train/Val/Test)
Phase 4: Geospatial Inference & Overlay Mapping (Folium Heatmaps + True/False Positive Classification Overlays)

```

---

### 4. Implementation Codebase

#### Phases 1–3: GPU Setup, Data Pipeline, Training, and Metric Logging

```python
import os
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# 0. RTX 4070 Ti HARDWARE INITIALIZATION
# ==============================================================================
def initialize_gpu():
    """Configures TensorFlow for RTX 4070 Ti (12GB) memory & compute cores."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[Hardware Setup] GPU detected: {gpus[0].name} | Memory growth enabled.")
        except RuntimeError as e:
            print(f"[Hardware Setup] Memory growth configuration error: {e}")
            
    # Enable mixed precision for Ada Lovelace 4th-gen Tensor Cores
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    print(f"[Hardware Setup] Active compute policy: {tf.keras.mixed_precision.global_policy().name}")

initialize_gpu()

# ==============================================================================
# 1. DATA INGESTION & FEATURE ENGINEERING
# ==============================================================================
def fetch_seismic_catalog(start_date="2021-01-01", end_date="2024-01-01", min_mag=2.5):
    """Pulls earthquake catalog records from USGS FDSN API."""
    url = (
        "https://earthquake.usgs.gov/fdsnws/event/1/query"
        f"?format=geojson&starttime={start_date}&endtime={end_date}&minmagnitude={min_mag}"
    )
    response = requests.get(url, timeout=45).json()
    records = []
    for f in response["features"]:
        coords = f["geometry"]["coordinates"]
        props = f["properties"]
        if coords[2] is not None and props["mag"] is not None:
            records.append({
                "time": pd.to_datetime(props["time"], unit="ms"),
                "lon": coords[0],
                "lat": coords[1],
                "depth": coords[2],
                "mag": props["mag"],
                "gap": props.get("gap") or 180.0,
                "dmin": props.get("dmin") or 1.0,
                "sig": props.get("sig") or 0
            })
    return pd.DataFrame(records).sort_values("time").reset_index(drop=True)

def build_spatiotemporal_grid(df, grid_size=2.0, time_step_days=7):
    """Aggregates seismic point events into spatiotemporal feature vectors."""
    df["lat_bin"] = (df["lat"] // grid_size) * grid_size
    df["lon_bin"] = (df["lon"] // grid_size) * grid_size
    df["period"] = df["time"].dt.to_period(f"{time_step_days}D")
    
    grouped = df.groupby(["lat_bin", "lon_bin", "period"])
    grid = grouped.agg(
        event_count=("mag", "count"),
        mean_depth=("depth", "mean"),
        max_mag=("mag", "max"),
        mean_gap=("gap", "mean"),
        mean_sig=("sig", "mean")
    ).reset_index()

    # Target: binary forecast of M >= 4.5 earthquake in the subsequent temporal window
    grid["target"] = (grid.groupby(["lat_bin", "lon_bin"])["max_mag"].shift(-1) >= 4.5).astype(int)
    return grid.dropna().sort_values("period").reset_index(drop=True)

raw_df = fetch_seismic_catalog()
grid_df = build_spatiotemporal_grid(raw_df)

feature_cols = ["lat_bin", "lon_bin", "event_count", "mean_depth", "mean_gap", "mean_sig"]
X = grid_df[feature_cols].values
y = grid_df["target"].values

# Chronological walk-forward split (70% Train, 15% Val, 15% Test)
n = len(X)
train_split, val_split = int(n * 0.70), int(n * 0.85)

X_train, y_train = X[:train_split], y[:train_split]
X_val, y_val     = X[train_split:val_split], y[train_split:val_split]
X_test, y_test   = X[val_split:], y[val_split:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

# High-throughput asynchronous tf.data streaming
BATCH_SIZE = 128  # Scaled for 4070 Ti Tensor Core occupancy

def make_tf_dataset(features, labels, is_train=False):
    ds = tf.data.Dataset.from_tensor_slices((features, labels))
    if is_train:
        ds = ds.shuffle(buffer_size=4096)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

train_ds = make_tf_dataset(X_train_scaled, y_train, is_train=True)
val_ds   = make_tf_dataset(X_val_scaled, y_val)
test_ds  = make_tf_dataset(X_test_scaled, y_test)

# ==============================================================================
# 2. MODEL ARCHITECTURE & MULTI-PARTITION METRIC LOGGER
# ==============================================================================
class PartitionMetricTracker(tf.keras.callbacks.Callback):
    """Tracks training, validation, and holdout test metrics at each epoch."""
    def __init__(self, test_dataset):
        super().__init__()
        self.test_dataset = test_dataset
        self.records = {
            "train_loss": [], "train_acc": [],
            "val_loss": [],   "val_acc": [],
            "test_loss": [],  "test_acc": []
        }

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.records["train_loss"].append(logs.get("loss"))
        self.records["train_acc"].append(logs.get("accuracy"))
        self.records["val_loss"].append(logs.get("val_loss"))
        self.records["val_acc"].append(logs.get("val_accuracy"))
        
        # Evaluate holdout test partition every epoch for real-time tracking
        t_loss, t_acc, _ = self.model.evaluate(self.test_dataset, verbose=0)
        self.records["test_loss"].append(t_loss)
        self.records["test_acc"].append(t_acc)

def build_4070ti_model(input_dim):
    """Deep network aligned with Tensor Core dimension multiples (multiples of 64)."""
    inputs = tf.keras.Input(shape=(input_dim,))
    
    # FP16 hidden layers
    x = tf.keras.layers.Dense(128, activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    
    # FP32 output layer for numeric stability under mixed precision
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", dtype="float32")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
        jit_compile=True  # Enables XLA JIT compilation
    )
    return model

model = build_4070ti_model(X_train_scaled.shape[1])
tracker = PartitionMetricTracker(test_dataset=test_ds)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=[tracker],
    verbose=1
)

# ==============================================================================
# 3. METRIC GRAPHING (TRAIN / VAL / TEST ACCURACY & LOSS)
# ==============================================================================
def plot_convergence_curves(metric_tracker):
    epochs = range(1, len(metric_tracker.records["train_loss"]) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Cross-Entropy Loss Curve
    ax1.plot(epochs, metric_tracker.records["train_loss"], label="Train Loss", color="#1f77b4", lw=2)
    ax1.plot(epochs, metric_tracker.records["val_loss"], label="Val Loss", color="#ff7f0e", lw=2, linestyle="--")
    ax1.plot(epochs, metric_tracker.records["test_loss"], label="Test Loss (Holdout)", color="#d62728", lw=1.5, linestyle=":")
    ax1.set_title("Cross-Entropy Loss Across Partitions", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Binary Cross-Entropy")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Accuracy Score Curve
    ax2.plot(epochs, metric_tracker.records["train_acc"], label="Train Accuracy", color="#2ca02c", lw=2)
    ax2.plot(epochs, metric_tracker.records["val_acc"], label="Val Accuracy", color="#9467bd", lw=2, linestyle="--")
    ax2.plot(epochs, metric_tracker.records["test_acc"], label="Test Accuracy (Holdout)", color="#d62728", lw=1.5, linestyle=":")
    ax2.set_title("Classification Accuracy Across Partitions", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()

plot_convergence_curves(tracker)

```

---

#### Phase 4: Geospatial Overlay Mapping Component

```python
import folium
from folium.plugins import HeatMap, FeatureGroupSubGroup

def generate_spatial_overlay_map(grid_data, val_idx, X_test_scaled, y_test, model, output_path="seismic_forecast_map.html"):
    """
    Renders an interactive Leaflet map overlaying:
      1. Training density baseline.
      2. Model forecasted risk probability heatmap.
      3. Classification verification markers (TP, FP, FN).
    """
    # 1. Generate model probabilities on test set
    preds_prob = model.predict(X_test_scaled).flatten()
    preds_binary = (preds_prob >= 0.5).astype(int)
    
    test_slice = grid_data.iloc[val_idx:].copy().reset_index(drop=True)
    test_slice["prob"] = preds_prob
    test_slice["pred"] = preds_binary
    test_slice["actual"] = y_test
    
    # 2. Base map centered on seismic coordinates
    center_lat = test_slice["lat_bin"].mean()
    center_lon = test_slice["lon_bin"].mean()
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=4, tiles="CartoDB dark_matter")
    
    # 3. Layer: Historical / Training Density
    train_slice = grid_data.iloc[:val_idx]
    train_heatmap_data = train_slice[["lat_bin", "lon_bin", "event_count"]].values.tolist()
    train_layer = folium.FeatureGroup(name="Historical Training Density")
    HeatMap(train_heatmap_data, radius=12, blur=10, min_opacity=0.2, gradient={0.2: "blue", 0.8: "cyan"}).add_to(train_layer)
    train_layer.add_to(fmap)
    
    # 4. Layer: Predicted Risk Probability Heatmap
    pred_heatmap_data = test_slice[["lat_bin", "lon_bin", "prob"]].values.tolist()
    pred_heat_layer = folium.FeatureGroup(name="Predicted Risk Heatmap (Test Set)")
    HeatMap(pred_heatmap_data, radius=16, blur=14, min_opacity=0.4, gradient={0.4: "orange", 1.0: "red"}).add_to(pred_heat_layer)
    pred_heat_layer.add_to(fmap)
    
    # 5. Layer: Ground Truth vs Prediction Marker Verification
    marker_layer = folium.FeatureGroup(name="Model Verification Markers")
    
    for _, row in test_slice.iterrows():
        lat, lon = row["lat_bin"], row["lon_bin"]
        prob = row["prob"]
        gt = int(row["actual"])
        pred = int(row["pred"])
        
        # Color coding confusion matrix categories
        if gt == 1 and pred == 1:
            color, label = "#00FF00", "True Positive (Accurate Forecast)"
        elif gt == 0 and pred == 1:
            color, label = "#FFA500", "False Positive (False Alarm)"
        elif gt == 1 and pred == 0:
            color, label = "#FF0000", "False Negative (Missed Event)"
        else:
            continue  # Suppress True Negatives to avoid clutter
            
        folium.CircleMarker(
            location=[lat, lon],
            radius=5 + (prob * 7),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            popup=f"<b>Status:</b> {label}<br><b>Predicted Risk:</b> {prob:.2%}<br><b>Actual M>=4.5:</b> {bool(gt)}"
        ).add_to(marker_layer)
        
    marker_layer.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    
    fmap.save(output_path)
    print(f"[Mapping Complete] Saved interactive map to {output_path}")
    return fmap

# Execute map rendering
generate_spatial_overlay_map(grid_df, val_split, X_test_scaled, y_test, model)

```

---

### 5. Verification, Validation & Hardware Acceptance Criteria

| Validation Target | Metric / Constraint | Target Acceptance Criteria |
| --- | --- | --- |
| **GPU Memory Ceiling** | VRAM Usage via `nvidia-smi` | Total allocated VRAM $\le 6.5\text{ GB}$ (out of $12\text{ GB}$ ceiling) at batch size $128$. |
| **XLA JIT Execution** | Compile Status & Speed | Zero graph compilation fallbacks; training step time $\le 1.8\text{ ms/step}$. |
| **Numeric Stability** | `mixed_float16` underflow | Loss remains finite ($\ne \text{NaN}$); final activation matches `float32`. |
| **Temporal Data Leakage** | Walk-Forward split check | $\max(T_{\text{train}}) < \min(T_{\text{val}}) < \min(T_{\text{test}})$. |
| **Overfitting Gap** | Generalization Delta | $\vert{}\text{Acc}_{\text{train}} - \text{Acc}_{\text{val}}\vert{} \le 0.05$ after 50 epochs with active Dropout ($0.25$). |
| **Spatial Map Calibration** | True Positive Alignment | High-probability prediction clusters intersect with active plate boundary fault lines. |
