#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECONOMETRIA TOPO FAST-FULL-PLUS (WORD-ALIGNED) — "SAVE EVERYTHING" EDITION
========================================================================

Goal
----
Deliver a single-file pipeline that ALWAYS saves a complete artifact package:
  * figures/      -> all expected plots (or placeholders if a module is unavailable)
  * reports/      -> report_tables.xlsx (with all sheets present), CSVs, JSON summary, manifest
  * models/       -> saved models (RF, LogReg, Deep if available), scaler, PCA proxy, configs

Key principle
-------------
"No missing artifacts": if TensorFlow or TDA are unavailable, the code produces:
  - placeholder plots (PNG) explaining the skip reason
  - placeholder tables/sheets in Excel & CSV describing unavailability
  - a manifest.json that lists which artifacts are real vs placeholders

This edition is designed to preserve bounded runtime while improving alignment to the Word narrative:
  - verified self-generation (generator–verifier–selector)
  - curriculum injection (bounded)
  - TDA phases + strided time evolution (bounded; placeholders if TDA missing)
  - time-ordered split (anti-leakage)
  - full export + model persistence

Dependencies (recommended)
--------------------------
Core:
  numpy, pandas, matplotlib, scikit-learn, openpyxl, joblib
Optional:
  tensorflow (for deep model)
  ripser, persim (for TDA)

Run
---
python this_script.py

Outputs
-------
outputs_econometria_topo_fast_plus_word/
  figures/
  reports/
  models/
"""

from __future__ import annotations

import os
import json
import zipfile
import time
import math
import warnings
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd

# Headless Matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report, brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neural_network import MLPRegressor

# Model persistence
try:
    import joblib
    JOBLIB_AVAILABLE = True
except Exception:
    JOBLIB_AVAILABLE = False

# ----------------------------
# Optional TDA dependencies
# ----------------------------
TDA_AVAILABLE = True
WASSERSTEIN_AVAILABLE = False
BOTTLENECK_AVAILABLE = False
try:
    from ripser import ripser
    from persim import plot_diagrams
    try:
        from persim import wasserstein, bottleneck
        WASSERSTEIN_AVAILABLE = True
        BOTTLENECK_AVAILABLE = True
    except Exception:
        WASSERSTEIN_AVAILABLE = False
        BOTTLENECK_AVAILABLE = False
except Exception:
    TDA_AVAILABLE = False

# ----------------------------
# Optional TensorFlow / Keras
# ----------------------------
TF_AVAILABLE = True
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks, regularizers
except Exception:
    TF_AVAILABLE = False


# ============================================================
# Configuration
# ============================================================

@dataclass
class Config:
    # Reproducibility
    SEED: int = 42

    # I/O
    OUTPUT_DIR: str = "outputs_econometria_topo_fast_plus_word"
    RUN_UNIQUE_DIR: bool = True  # avoids PermissionError when previous outputs are open in Excel
    FIG_DIR: str = "figures"
    REP_DIR: str = "reports"
    MOD_DIR: str = "models"

    # Data source (set INPUT_CSV_PATH to use real data)
    INPUT_CSV_PATH: Optional[str] = None
    TARGET_COL: Optional[str] = None
    FEATURE_COLS: Optional[List[str]] = None

    # Synthetic data (default)
    SYN_N_POINTS: int = 24000
    SYN_N_FEATURES: int = 6
    SYN_ANOMALY_RATE: float = 0.035
    SYN_ANOMALY_BLOCKS: int = 14
    SYN_NOISE_STD: float = 0.50

    # Windowing (time-evolution evidence)
    WINDOW_SIZE: int = 256
    WINDOW_HOP: int = 96
    WINDOW_LABEL_RATIO: float = 0.10  # window anomalous if >= this anomaly fraction inside window

    # Temporal split (anti-leakage)
    TEST_SIZE: float = 0.25
    VAL_FROM_TRAIN: float = 0.20

    # Deep model (multi-task)
    DO_DEEP: bool = True
    DEEP_EPOCHS: int = 8
    DEEP_BATCH: int = 768
    DEEP_LATENT_DIM: int = 8
    DEEP_LR: float = 1e-3
    DEEP_PATIENCE: int = 2
    LAMBDA_RECON: float = 0.35
    L2_REG: float = 1e-5

    # Deep model input mode
    #   "STATS_MLP"  -> current window-statistics vector (fast, default)
    #   "RAW_CONV1D" -> raw window -> Conv1D encoder (closer to code-01 style; still bounded)
    DEEP_INPUT_MODE: str = "STATS_MLP"

    # Raw-window (Conv1D) options (used only if DEEP_INPUT_MODE="RAW_CONV1D")
    RAW_WINDOW_SIZE: int = 256
    RAW_WINDOW_HOP: int = 96

    # Geometry/topology-inspired regularization (bounded)
    # GEO_LAMBDA = 0.0 disables the term.
    GEO_LAMBDA: float = 0.05
    GEO_N_PAIRS: int = 64
    GEO_MARGIN: float = 0.0

    # Extra saving for "code-01-like" completeness (still bounded)
    SAVE_CHECKPOINTS: bool = True
    CHECKPOINT_EVERY_EPOCH: bool = False   # if False, saves only best via ModelCheckpoint
    CHECKPOINT_MAX_KEEP: int = 3

    # Monitoring: deep + TDA every few epochs (bounded)
    DO_PER_EPOCH_MONITORING: bool = True
    MONITOR_EVERY_EPOCHS: int = 1
    TDA_EVERY_EPOCHS: int = 2
    MONITOR_MAX_SAMPLES: int = 4096

    # Recon examples (small fixed set)
    N_RECON_EXAMPLES: int = 24

    # Baselines
    DO_RF: bool = True
    RF_N_ESTIMATORS: int = 240
    RF_MAX_DEPTH: Optional[int] = 10
    RF_MIN_SAMPLES_LEAF: int = 2

    DO_LOGREG: bool = True
    LOGREG_C: float = 1.0
    LOGREG_MAX_ITER: int = 250

    # Threshold sweep
    DO_THRESHOLD_SWEEP: bool = True
    THRESH_GRID_N: int = 101

    # Feature importance (RF only)
    DO_RF_PERM_IMPORTANCE: bool = True
    RF_PERM_REPEATS: int = 5

    # ============================================================
    # Verified Self-Generation + Curriculum (WORD-aligned)
    # ============================================================
    DO_SELFGEN: bool = True
    SELFGEN_CANDIDATES: int = 900
    SELFGEN_TOPK: int = 120
    SELFGEN_NOISE_STD: float = 0.35
    SELFGEN_CLIP_Z: float = 3.0

    # Verifier constraints (plausibility)
    VERIFIER_USE_TRAIN_QUANTILES: bool = True
    VERIFIER_Q_LOW: float = 0.001
    VERIFIER_Q_HIGH: float = 0.999
    VERIFIER_MAX_L2_Z: float = 35.0
    VERIFIER_MAX_ABS_Z: float = 3.2

    # Selector scoring
    SELECTOR_ALPHA_UNCERT: float = 0.60
    SELECTOR_BETA_STRESS: float = 0.40
    SELECTOR_GAMMA_TDA: float = 0.20
    SELECTOR_USE_TDA_ON_TOPM: bool = True
    SELECTOR_TDA_TOPM: int = 32

    # Curriculum injection mode: "NONE" | "ONCE" | "PER_EPOCH"
    CURRICULUM_MODE: str = "ONCE"
    CURRICULUM_INJECT_TOPK: int = 80
    CURRICULUM_PER_EPOCH_K: int = 24
    CURRICULUM_MAX_EPOCHS: int = 8

    # Candidate labeling
    CURRICULUM_LABEL_POLICY: str = "MODEL_PROB"  # "MODEL_PROB" | "ANOMALY"

    # ============================================================
    # TDA controls (bounded)
    # ============================================================
    DO_TDA: bool = True
    TDA_MAXDIM: int = 1
    TDA_SUBSAMPLE: int = 280
    TDA_STRIDE_WINDOWS: int = 10
    BETTI_NGRID: int = 96
    DO_TDA_DISTANCES: bool = True

    # TDA phase toggles
    DO_TDA_NORMAL_VS_ANOM: bool = True
    DO_TDA_SELFGEN: bool = True

    # Visualization
    FIG_DPI: int = 140
    TSNE_MAX_N: int = 3500

    # Exports
    SAVE_EXCEL: bool = True
    SAVE_CSV: bool = True
    SAVE_JSON: bool = True

    # Preset aligned to Word narrative
    WORD_COMPAT_PRESET: bool = True


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    try:
        import random
        random.seed(seed)
    except Exception:
        pass
    if TF_AVAILABLE:
        try:
            tf.random.set_seed(seed)
        except Exception:
            pass
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)



def package_reports_zip(output_dir: str, zip_name: str = "reports.zip") -> str:
    """Create a reports.zip that contains the reports/ folder only (deterministic structure)."""
    rep_dir = os.path.join(output_dir, "reports")
    zip_path = os.path.join(output_dir, zip_name)
    if not os.path.isdir(rep_dir):
        return ""
    # Write to temp then replace (reduces partial zips if interrupted)
    tmp_zip = zip_path + ".tmp"
    try:
        if os.path.exists(tmp_zip):
            os.remove(tmp_zip)
    except Exception:
        pass
    with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(rep_dir):
            for fn in files:
                full = os.path.join(root, fn)
                rel = os.path.relpath(full, output_dir)
                z.write(full, rel)
    try:
        if os.path.exists(zip_path):
            os.remove(zip_path)
    except Exception:
        # If the old zip is locked, keep a timestamped name
        zip_path = os.path.join(output_dir, f"reports_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip")
    os.replace(tmp_zip, zip_path)
    return zip_path



def safe_json_convert(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {k: safe_json_convert(v) for k, v in x.items()}
    if isinstance(x, list):
        return [safe_json_convert(v) for v in x]
    return x


def save_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(safe_json_convert(data), f, indent=2, ensure_ascii=False)


def placeholder_plot(out_png: str, title: str, message: str, dpi: int = 140) -> None:
    plt.figure(figsize=(7, 4), dpi=dpi)
    plt.axis("off")
    plt.title(title)
    plt.text(0.01, 0.5, message, fontsize=10, va="center")
    plt.tight_layout()
    ensure_dir(os.path.dirname(out_png))
    plt.savefig(out_png)
    plt.close()


def safe_sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def apply_word_compat_preset(cfg: Config) -> None:
    if not cfg.WORD_COMPAT_PRESET:
        return
    cfg.DEEP_EPOCHS = int(min(cfg.DEEP_EPOCHS, 10))
    cfg.DEEP_PATIENCE = int(min(cfg.DEEP_PATIENCE, 3))

    cfg.TDA_MAXDIM = int(min(cfg.TDA_MAXDIM, 1))
    cfg.TDA_SUBSAMPLE = int(min(cfg.TDA_SUBSAMPLE, 320))
    cfg.TDA_STRIDE_WINDOWS = int(max(cfg.TDA_STRIDE_WINDOWS, 8))
    cfg.BETTI_NGRID = int(min(cfg.BETTI_NGRID, 120))

    cfg.SELFGEN_CANDIDATES = int(min(cfg.SELFGEN_CANDIDATES, 1200))
    cfg.SELFGEN_TOPK = int(min(cfg.SELFGEN_TOPK, 200))
    cfg.SELECTOR_TDA_TOPM = int(min(cfg.SELECTOR_TDA_TOPM, 40))
    cfg.CURRICULUM_INJECT_TOPK = int(min(cfg.CURRICULUM_INJECT_TOPK, cfg.SELFGEN_TOPK))
    cfg.CURRICULUM_MAX_EPOCHS = int(min(cfg.CURRICULUM_MAX_EPOCHS, cfg.DEEP_EPOCHS))



# ============================================================
# Excel/CSV safety: convert tensors / numpy scalars to plain Python
# ============================================================

def _to_py_scalar(v: Any) -> Any:
    """
    Convert values that commonly break Excel export (e.g., tf.Tensor, np scalars)
    into plain Python scalars or JSON/Excel-safe structures.
    """
    # TensorFlow tensor -> numpy / python
    try:
        if TF_AVAILABLE:
            import tensorflow as tf  # type: ignore
            if isinstance(v, tf.Tensor):
                v = v.numpy()
            if isinstance(v, tf.Variable):
                v = v.numpy()
    except Exception:
        pass

    # numpy scalars
    try:
        if isinstance(v, (np.generic,)):
            return v.item()
    except Exception:
        pass

    # numpy arrays
    if isinstance(v, np.ndarray):
        if v.ndim == 0:
            try:
                return v.item()
            except Exception:
                return float(v)
        # keep small arrays readable; otherwise stringify
        try:
            if v.size <= 32:
                return v.tolist()
            return str(v)
        except Exception:
            return str(v)

    # Python numeric/bool/None/str are fine
    if v is None or isinstance(v, (str, int, float, bool)):
        return v

    # bytes -> str
    if isinstance(v, (bytes, bytearray)):
        try:
            return v.decode("utf-8", errors="replace")
        except Exception:
            return str(v)

    # lists/tuples: convert elements
    if isinstance(v, (list, tuple)):
        return [_to_py_scalar(x) for x in v]

    # dict: convert values
    if isinstance(v, dict):
        return {str(k): _to_py_scalar(val) for k, val in v.items()}

    # fallback
    return str(v)


def sanitize_dataframe_for_export(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df contains only Excel/CSV-safe values (no tf.Tensor objects, etc.).
    Keeps numeric columns numeric when possible.
    """
    if df is None:
        return df

    out = df.copy()

    # Apply conversion to object columns (fast path)
    for col in out.columns:
        if out[col].dtype == "object":
            out[col] = out[col].map(_to_py_scalar)

    # Also sanitize index just in case
    try:
        out.index = pd.Index([_to_py_scalar(x) for x in out.index])
    except Exception:
        pass

    return out


def empty_table(columns: List[str], note: str = "") -> pd.DataFrame:
    """Return a 1-row DataFrame with numeric placeholders (no NaNs), plus NOTE.

    This keeps every CSV/Excel sheet non-empty and numeric even when a module is unavailable.
    The NOTE column preserves provenance without breaking downstream Excel formulas/plots.
    """
    cols = list(columns)
    if "NOTE" not in cols:
        cols.append("NOTE")
    row = {}
    for c in cols:
        if c == "NOTE":
            row[c] = str(note) if note is not None else ""
        else:
            row[c] = 0.0
    return pd.DataFrame([row], columns=cols)


# ============================================================
# Data
# ============================================================

def generate_synthetic_time_series(cfg: Config) -> Tuple[pd.DataFrame, np.ndarray]:
    N = cfg.SYN_N_POINTS
    F = cfg.SYN_N_FEATURES
    t = np.arange(N)

    X = np.zeros((N, F), dtype=float)
    for j in range(F):
        base = (
            0.9 * np.sin(2.0 * np.pi * t / (700 + 35 * j)) +
            0.5 * np.cos(2.0 * np.pi * t / (1200 + 50 * j)) +
            0.0012 * (j + 1) * t
        )
        X[:, j] = base

    cov = 0.15 * np.ones((F, F)) + 0.85 * np.eye(F)
    L = np.linalg.cholesky(cov)
    noise = (np.random.randn(N, F) @ L.T) * cfg.SYN_NOISE_STD
    X = X + noise

    y = np.zeros(N, dtype=int)
    rng = np.random.default_rng(cfg.SEED)
    block_len = max(80, int(cfg.SYN_ANOMALY_RATE * N / max(1, cfg.SYN_ANOMALY_BLOCKS) * 4))

    for _ in range(cfg.SYN_ANOMALY_BLOCKS):
        start = int(rng.integers(0, N - block_len))
        end = start + block_len
        y[start:end] = 1

        feats = rng.choice(F, size=max(1, F // 2), replace=False)
        amp = float(rng.uniform(2.5, 5.5))
        mode = int(rng.integers(0, 3))

        if mode == 0:
            X[start:end, feats] += amp * (rng.standard_normal((block_len, len(feats))) * 0.6)
        elif mode == 1:
            X[start:end, feats] += amp
        else:
            tt = np.arange(block_len)
            burst = amp * np.sin(2 * np.pi * tt / rng.integers(10, 40))
            X[start:end, feats] += burst[:, None]

    df = pd.DataFrame(X, columns=[f"x{j+1}" for j in range(F)])
    return df, y


def load_or_generate_data(cfg: Config) -> Tuple[pd.DataFrame, np.ndarray]:
    if cfg.INPUT_CSV_PATH is None:
        return generate_synthetic_time_series(cfg)

    df = pd.read_csv(cfg.INPUT_CSV_PATH)
    if cfg.TARGET_COL is None:
        raise ValueError("TARGET_COL must be set when INPUT_CSV_PATH is provided.")

    y = df[cfg.TARGET_COL].astype(int).to_numpy()

    if cfg.FEATURE_COLS is None:
        feature_cols = [
            c for c in df.columns
            if c != cfg.TARGET_COL and pd.api.types.is_numeric_dtype(df[c])
        ]
    else:
        feature_cols = cfg.FEATURE_COLS

    Xdf = df[feature_cols].copy()
    Xdf = Xdf.replace([np.inf, -np.inf], np.nan)
    Xdf = Xdf.fillna(Xdf.median(numeric_only=True))
    return Xdf, y


# ============================================================
# Windowing
# ============================================================

def make_windows_stats(
    df: pd.DataFrame,
    y_pointwise: np.ndarray,
    cfg: Config
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    X = df.to_numpy(dtype=float)
    N, F = X.shape
    W, H = cfg.WINDOW_SIZE, cfg.WINDOW_HOP
    starts = np.arange(0, N - W + 1, H, dtype=int)

    t = np.arange(W, dtype=float)
    t_centered = t - t.mean()
    denom = np.sum(t_centered ** 2) + 1e-12

    Xw_list = []
    y_ratio = []

    for s in starts:
        seg = X[s:s + W, :]
        mu = seg.mean(axis=0)
        sd = seg.std(axis=0)
        mn = seg.min(axis=0)
        mx = seg.max(axis=0)
        seg_centered = seg - seg.mean(axis=0, keepdims=True)
        slope = (t_centered[:, None] * seg_centered).sum(axis=0) / denom

        Xw_list.append(np.concatenate([mu, sd, mn, mx, slope], axis=0))
        y_ratio.append(float(y_pointwise[s:s + W].mean()))

    Xw = np.vstack(Xw_list)
    y_ratio = np.array(y_ratio, dtype=float)
    yw = (y_ratio >= cfg.WINDOW_LABEL_RATIO).astype(int)

    base_cols = list(df.columns)
    feat_names = []
    for stat in ["mean", "std", "min", "max", "slope"]:
        for c in base_cols:
            feat_names.append(f"{c}_{stat}")

    return Xw, yw, starts, feat_names


def temporal_split_windows(
    X: np.ndarray,
    y: np.ndarray,
    win_starts: np.ndarray,
    test_size: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = X.shape[0]
    order = np.argsort(win_starts)
    Xo = X[order]
    yo = y[order]
    wo = win_starts[order]

    n_test = int(math.ceil(test_size * n))
    n_test = max(1, min(n - 1, n_test))
    n_train = n - n_test

    Xtr = Xo[:n_train]
    ytr = yo[:n_train]
    wtr = wo[:n_train]

    Xte = Xo[n_train:]
    yte = yo[n_train:]
    wte = wo[n_train:]

    return Xtr, Xte, ytr, yte, wtr, wte


# ============================================================
# Models
# ============================================================

def build_rf(cfg: Config) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=cfg.RF_N_ESTIMATORS,
        max_depth=cfg.RF_MAX_DEPTH,
        min_samples_leaf=cfg.RF_MIN_SAMPLES_LEAF,
        n_jobs=-1,
        random_state=cfg.SEED,
        class_weight="balanced_subsample"
    )


def build_logreg(cfg: Config) -> LogisticRegression:
    return LogisticRegression(
        C=cfg.LOGREG_C,
        max_iter=cfg.LOGREG_MAX_ITER,
        solver="lbfgs"
    )


def _geo_pairwise_loss(x_in: tf.Tensor, z: tf.Tensor, n_pairs: int, margin: float = 0.0) -> tf.Tensor:
    """
    Bounded geometry-preservation regularizer.

    Samples a small number of random pairs within the batch and penalizes mismatch
    between normalized input-space and latent-space distances (stress-like penalty).
    """
    x_flat = tf.reshape(x_in, [tf.shape(x_in)[0], -1])
    b = tf.shape(x_flat)[0]
    # at least 1 pair
    n_pairs = tf.maximum(tf.cast(n_pairs, tf.int32), 1)
    n_pairs = tf.minimum(n_pairs, tf.maximum(b, 2) * 2)

    i = tf.random.uniform([n_pairs], minval=0, maxval=b, dtype=tf.int32)
    j = tf.random.uniform([n_pairs], minval=0, maxval=b, dtype=tf.int32)
    j = tf.where(tf.equal(i, j), (j + 1) % b, j)

    xi = tf.gather(x_flat, i)
    xj = tf.gather(x_flat, j)
    zi = tf.gather(z, i)
    zj = tf.gather(z, j)

    dx = tf.norm(xi - xj, axis=1)
    dz = tf.norm(zi - zj, axis=1)

    dx = dx / (tf.reduce_mean(dx) + 1e-6)
    dz = dz / (tf.reduce_mean(dz) + 1e-6)

    diff = tf.nn.relu(tf.abs(dx - dz) - margin)
    return tf.reduce_mean(tf.square(diff))


if TF_AVAILABLE:
    class GeoMultiTaskModel(tf.keras.Model):
        """
        Wrapper around a base multi-output model to add geometry regularization in train_step.
        """
        def __init__(self, base: tf.keras.Model, geo_lambda: float, geo_n_pairs: int, geo_margin: float):
            super().__init__(name="GeoMultiTaskModel")
            self.base = base
            self.geo_lambda = float(geo_lambda)
            self.geo_n_pairs = int(geo_n_pairs)
            self.geo_margin = float(geo_margin)
            self._encoder = None

        def get_config(self):
            """
            Minimal config for reproducibility; note that the wrapped base model
            is saved separately as a standard Keras model (see save_models_always).
            """
            return {
                "geo_lambda": float(self.geo_lambda),
                "geo_n_pairs": int(self.geo_n_pairs),
                "geo_margin": float(self.geo_margin),
            }

        @classmethod
        def from_config(cls, config):
            # This wrapper is primarily for training-time regularization.
            # For inference/reuse, load the separately-saved base model.
            base = tf.keras.Sequential(name="placeholder_base")
            return cls(
                base=base,
                geo_lambda=float(config.get("geo_lambda", 0.0)),
                geo_n_pairs=int(config.get("geo_n_pairs", 64)),
                geo_margin=float(config.get("geo_margin", 0.0)),
            )


        def call(self, inputs, training=False):
            return self.base(inputs, training=training)

        def _get_encoder(self):
            if self._encoder is None:
                inp = self.base.inputs[0]
                z_tensor = self.base.get_layer("z").output
                self._encoder = tf.keras.Model(inp, z_tensor, name="GeoEncoder")
            return self._encoder

        def train_step(self, data):
            x, y = data
            with tf.GradientTape() as tape:
                out = self.base(x, training=True)
                loss = self.compiled_loss(y, out, regularization_losses=self.base.losses)

                if self.geo_lambda > 0.0:
                    enc = self._get_encoder()
                    z_batch = enc(x, training=True)
                    geo = _geo_pairwise_loss(x, z_batch, self.geo_n_pairs, self.geo_margin)
                    loss = loss + self.geo_lambda * geo

            grads = tape.gradient(loss, self.base.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.base.trainable_variables))

            # Update metrics (compiled metrics already include p_anomaly accuracy)
            self.compiled_metrics.update_state(y, out)

            logs = {m.name: m.result() for m in self.metrics}
            if self.geo_lambda > 0.0:
                logs["geo_loss"] = geo
            logs["loss"] = loss
            return logs



else:
    GeoMultiTaskModel = None

def build_deep_multitask(input_dim: int, cfg: Config):
    """
    Builds a deep multi-task model.

    Modes:
      * STATS_MLP  : input_dim vector (fast; default).
      * RAW_CONV1D : input is (RAW_WINDOW_SIZE, n_features).
    """
    if not TF_AVAILABLE:
        return None, None

    mode = str(getattr(cfg, "DEEP_INPUT_MODE", "STATS_MLP")).upper().strip()
    n_features = int(getattr(cfg, "SYN_N_FEATURES", 6))
    raw_T = int(getattr(cfg, "RAW_WINDOW_SIZE", 256))

    if mode == "RAW_CONV1D":
        inp = layers.Input(shape=(raw_T, n_features), name="x_raw")

        x = layers.Conv1D(32, 5, padding="same", activation="relu",
                          kernel_regularizer=regularizers.l2(cfg.L2_REG))(inp)
        x = layers.MaxPool1D(2)(x)
        x = layers.Conv1D(64, 5, padding="same", activation="relu",
                          kernel_regularizer=regularizers.l2(cfg.L2_REG))(x)
        x = layers.MaxPool1D(2)(x)
        x = layers.Conv1D(64, 3, padding="same", activation="relu",
                          kernel_regularizer=regularizers.l2(cfg.L2_REG))(x)
        x = layers.GlobalAveragePooling1D()(x)

        z = layers.Dense(int(cfg.DEEP_LATENT_DIM), activation=None, name="z")(x)

        h = layers.Dense(64, activation="relu")(z)
        p = layers.Dense(1, activation="sigmoid", name="p_anomaly")(h)

        d = layers.Dense(128, activation="relu")(z)
        d = layers.Dense((raw_T // 4) * 64, activation="relu")(d)
        d = layers.Reshape((raw_T // 4, 64))(d)
        d = layers.UpSampling1D(2)(d)
        d = layers.Conv1D(64, 3, padding="same", activation="relu")(d)
        d = layers.UpSampling1D(2)(d)
        d = layers.Conv1D(32, 3, padding="same", activation="relu")(d)
        recon = layers.Conv1D(n_features, 1, padding="same", activation=None, name="x_recon")(d)

        base = models.Model(inp, outputs={"p_anomaly": p, "x_recon": recon}, name="DeepMultiTask_RAW_CONV1D")
        encoder = models.Model(inp, z, name="Encoder_RAW_CONV1D")
    else:
        inp = layers.Input(shape=(input_dim,), name="x")

        x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(cfg.L2_REG))(inp)
        x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(cfg.L2_REG))(x)

        z = layers.Dense(int(cfg.DEEP_LATENT_DIM), activation=None, name="z")(x)

        h = layers.Dense(32, activation="relu")(z)
        p = layers.Dense(1, activation="sigmoid", name="p_anomaly")(h)

        d = layers.Dense(64, activation="relu")(z)
        d = layers.Dense(128, activation="relu")(d)
        recon = layers.Dense(input_dim, activation=None, name="x_recon")(d)

        base = models.Model(inp, outputs={"p_anomaly": p, "x_recon": recon}, name="DeepMultiTask_STATS_MLP")
        encoder = models.Model(inp, z, name="Encoder_STATS_MLP")

    opt = tf.keras.optimizers.Adam(learning_rate=float(cfg.DEEP_LR))

    geo_lambda = float(getattr(cfg, "GEO_LAMBDA", 0.0))
    if geo_lambda > 0.0:
        model = GeoMultiTaskModel(
            base=base,
            geo_lambda=geo_lambda,
            geo_n_pairs=int(getattr(cfg, "GEO_N_PAIRS", 64)),
            geo_margin=float(getattr(cfg, "GEO_MARGIN", 0.0))
        )
    else:
        model = base

    model.compile(
        optimizer=opt,
        loss={"p_anomaly": "binary_crossentropy", "x_recon": "mse"},
        loss_weights={"p_anomaly": 1.0, "x_recon": float(cfg.LAMBDA_RECON)},
        metrics={"p_anomaly": ["accuracy"]}
    )
    return model, encoder


# ============================================================
# Deep Fallback (no TensorFlow / failure-safe) -- fills deep_* tables with REAL numeric results
# ============================================================

def _sk_mlp_forward_hidden(X: np.ndarray, mlp: Any) -> np.ndarray:
    """Compute hidden-layer activations for a 1-hidden-layer sklearn MLP (relu/tanh/logistic/identity)."""
    W1 = mlp.coefs_[0]
    b1 = mlp.intercepts_[0]
    H = X @ W1 + b1
    act = getattr(mlp, "activation", "relu")
    if act == "relu":
        return np.maximum(H, 0.0)
    if act == "tanh":
        return np.tanh(H)
    if act == "logistic":
        return 1.0 / (1.0 + np.exp(-H))
    return H

def train_deep_fallback_stats(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xva: np.ndarray,
    yva: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
    Xall: np.ndarray,
    cfg: Config
) -> Dict[str, Any]:
    """
    Failure-safe deep pipeline that does NOT require TensorFlow.
    It trains:
      * a 1-hidden-layer MLPRegressor as a reconstruction model (autoencoder proxy),
      * a LogisticRegression head on hidden activations as anomaly probability estimator,
    and returns fully populated deep_* tables with real numeric values.
    """
    def _flatten_deep_input(X: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if X is None:
            return None
        X = np.asarray(X)
        if X.ndim == 1:
            return X.reshape(-1, 1)
        if X.ndim >= 3:
            return X.reshape(X.shape[0], -1)
        return X

    Xtr = _flatten_deep_input(Xtr)
    Xva = _flatten_deep_input(Xva)
    Xte = _flatten_deep_input(Xte)
    Xall = _flatten_deep_input(Xall)
    ytr = np.asarray(ytr).astype(int) if ytr is not None else None
    yva = np.asarray(yva).astype(int) if yva is not None else None
    yte = np.asarray(yte).astype(int) if yte is not None else None

    if Xtr is None or len(Xtr) == 0:
        raise RuntimeError("Fallback deep training: empty Xtr.")

    # Ensure a validation set exists
    if Xva is None or len(Xva) == 0:
        n = len(Xtr)
        nva = max(1, int(0.1 * n))
        Xva = Xtr[-nva:].copy()
        yva = ytr[-nva:].copy()
        Xtr = Xtr[:-nva].copy()
        ytr = ytr[:-nva].copy()

    sc = StandardScaler()
    sc.fit(Xtr)
    Xtr_s = sc.transform(Xtr)
    Xva_s = sc.transform(Xva)
    Xte_s = sc.transform(Xte)
    Xall_s = sc.transform(Xall)

    latent_dim = int(getattr(cfg, "DEEP_LATENT_DIM", 8))
    max_epochs = int(getattr(cfg, "DEEP_EPOCHS", 8))
    lr = float(getattr(cfg, "DEEP_LR", 1e-3))

    recon = MLPRegressor(
        hidden_layer_sizes=(latent_dim,),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=lr,
        max_iter=1,
        warm_start=True,
        shuffle=True,
        random_state=int(getattr(cfg, "SEED", 42))
    )

    history_rows = []
    epoch_monitor_rows = []

    eps = 1e-12
    def logloss(y, p):
        y = y.astype(float)
        p = np.clip(p, eps, 1 - eps)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    lam = float(getattr(cfg, "LAMBDA_RECON", 0.25))

    for epoch in range(max_epochs):
        recon.fit(Xtr_s, Xtr_s)

        Xtr_hat = recon.predict(Xtr_s)
        Xva_hat = recon.predict(Xva_s)

        tr_mse = float(np.mean((Xtr_s - Xtr_hat) ** 2))
        va_mse = float(np.mean((Xva_s - Xva_hat) ** 2))

        Ztr = _sk_mlp_forward_hidden(Xtr_s, recon)
        Zva = _sk_mlp_forward_hidden(Xva_s, recon)

        head = LogisticRegression(solver="lbfgs", max_iter=200, class_weight="balanced")

        if len(np.unique(ytr)) < 2:
            r = np.mean((Xtr_s - Xtr_hat) ** 2, axis=1)
            thr = np.quantile(r, 0.95)
            ytr_eff = (r >= thr).astype(int)
        else:
            ytr_eff = ytr.astype(int)

        head.fit(Ztr, ytr_eff)

        ptr = head.predict_proba(Ztr)[:, 1]
        pva = head.predict_proba(Zva)[:, 1]

        if len(np.unique(ytr_eff)) > 1:
            tr_ll = logloss(ytr_eff, ptr)
            tr_auc = float(roc_auc_score(ytr_eff, ptr))
            tr_ap = float(average_precision_score(ytr_eff, ptr))
        else:
            tr_ll, tr_auc, tr_ap = float("nan"), float("nan"), float("nan")

        if len(np.unique(yva)) > 1:
            va_ll = logloss(yva.astype(int), pva)
            va_auc = float(roc_auc_score(yva.astype(int), pva))
            va_ap = float(average_precision_score(yva.astype(int), pva))
        else:
            va_ll, va_auc, va_ap = float("nan"), float("nan"), float("nan")

        tr_total = float(tr_ll + lam * tr_mse) if np.isfinite(tr_ll) else float(lam * tr_mse)
        va_total = float(va_ll + lam * va_mse) if np.isfinite(va_ll) else float(lam * va_mse)

        history_rows.append({
            "epoch": int(epoch + 1),
            "loss": tr_total,
            "p_anomaly_loss": tr_ll,
            "x_recon_loss": tr_mse,
            "val_loss": va_total,
            "val_p_anomaly_loss": va_ll,
            "val_x_recon_loss": va_mse
        })

        epoch_monitor_rows.append({
            "epoch": int(epoch + 1),
            "train_recon_mse": tr_mse,
            "val_recon_mse": va_mse,
            "train_auc_roc_sub": tr_auc,
            "train_auc_pr_sub": tr_ap,
            "val_auc_roc_sub": va_auc,
            "val_auc_pr_sub": va_ap,
            "NOTE": "fallback_sklearn"
        })

    deep_history_df = pd.DataFrame(history_rows)
    deep_epoch_monitor_df = pd.DataFrame(epoch_monitor_rows)

    # Lightweight numeric surrogate for 'TDA over epochs' (keeps deep* artifacts populated)
    # Here it is parameterized using monitored reconstruction/AUC signals so it is deterministic and non-empty.
    n_ep = int(len(deep_epoch_monitor_df))
    if n_ep == 0:
        n_ep = 1
        deep_epoch_monitor_df = pd.DataFrame({'epoch': [1], 'train_recon_mse': [0.0], 'val_recon_mse': [0.0], 'train_auc_roc_sub': [0.0]})
    tda_over_epochs_latent_df = pd.DataFrame({
        'epoch': deep_epoch_monitor_df.get('epoch', pd.Series(np.arange(1, n_ep + 1))).to_numpy(dtype=int),
        'dim': np.zeros(n_ep, dtype=int),
        'n_features': np.zeros(n_ep, dtype=int),
        'total_persistence': np.maximum(0.0, deep_epoch_monitor_df.get('train_recon_mse', pd.Series(np.zeros(n_ep))).to_numpy(dtype=float)),
        'mean_lifetime': np.maximum(0.0, deep_epoch_monitor_df.get('val_recon_mse', pd.Series(np.zeros(n_ep))).to_numpy(dtype=float)),
        'persistence_entropy': np.maximum(0.0, deep_epoch_monitor_df.get('train_auc_roc_sub', pd.Series(np.zeros(n_ep))).to_numpy(dtype=float)),
        'NOTE': ['proxy_fallback_sklearn'] * n_ep
    })

    Zte = _sk_mlp_forward_hidden(Xte_s, recon)
    Zall = _sk_mlp_forward_hidden(Xall_s, recon)

    head_final = LogisticRegression(solver="lbfgs", max_iter=400, class_weight="balanced")
    if len(np.unique(ytr)) < 2:
        Xtr_hat = recon.predict(Xtr_s)
        r = np.mean((Xtr_s - Xtr_hat) ** 2, axis=1)
        thr = np.quantile(r, 0.95)
        ytr_eff = (r >= thr).astype(int)
    else:
        ytr_eff = ytr.astype(int)

    head_final.fit(_sk_mlp_forward_hidden(Xtr_s, recon), ytr_eff)

    pte = head_final.predict_proba(Zte)[:, 1]
    pall = head_final.predict_proba(Zall)[:, 1]

    Xte_hat = recon.predict(Xte_s)
    recon_mse_te = np.mean((Xte_s - Xte_hat) ** 2, axis=1)

    deep_recon_error_df = pd.DataFrame({
        "idx": np.arange(len(recon_mse_te), dtype=int),
        "recon_mse": recon_mse_te.astype(float),
        "p_anomaly": pte.astype(float)
    })

    deep_latent_df = pd.DataFrame(Zall.astype(float))
    deep_latent_df.insert(0, "idx", np.arange(Zall.shape[0], dtype=int))
    deep_latent_df["p_anomaly"] = pall.astype(float)

    n_ex = int(min(24, len(Xte_s)))
    ex_idx = np.linspace(0, len(Xte_s) - 1, n_ex).astype(int) if n_ex > 0 else np.array([], dtype=int)
    deep_recon_examples_df = pd.DataFrame({
        "example_idx": ex_idx.astype(int),
        "recon_mse": recon_mse_te[ex_idx].astype(float) if len(ex_idx) else [],
        "p_anomaly": pte[ex_idx].astype(float) if len(ex_idx) else []
    })

    # Recon MSE for ALL windows (needed so deep_recon_error and deep_latent sheets are real even in fallback)
    Xall_hat = recon.predict(Xall_s)
    recon_mse_all = np.mean((Xall_s - Xall_hat) ** 2, axis=1)

    return {
        "scaler": sc,
        "recon_model": recon,
        "head_model": head_final,
        "latent_all": Zall,
        "recon_mse_all": recon_mse_all,
        "p_test": pte,
        "p_all": pall,
        "deep_training_history": deep_history_df,
        "deep_epoch_monitoring": deep_epoch_monitor_df,
        "tda_over_epochs_latent": tda_over_epochs_latent_df,
        "deep_recon_error": deep_recon_error_df,
        "deep_recon_examples": deep_recon_examples_df,
        "deep_latent": deep_latent_df,
        "NOTE": "fallback_sklearn"
    }



# ============================================================
# Metrics + Plotting (always produce PNGs)
# ============================================================

def plot_roc_pr_always(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    out_dir: str,
    dpi: int,
    prefix: str,
    manifest: Dict[str, Any]
) -> Dict[str, float]:
    ensure_dir(out_dir)
    out = {"auc_roc": float("nan"), "auc_pr": float("nan")}

    if len(np.unique(y_true)) > 1:
        try:
            out["auc_roc"] = float(roc_auc_score(y_true, y_prob))
            out["auc_pr"] = float(average_precision_score(y_true, y_prob))

            fpr, tpr, _ = roc_curve(y_true, y_prob)
            plt.figure(figsize=(6, 5), dpi=dpi)
            plt.plot(fpr, tpr)
            plt.plot([0, 1], [0, 1], linestyle="--")
            plt.title(f"{prefix} ROC (AUC={out['auc_roc']:.4f})")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.tight_layout()
            roc_path = os.path.join(out_dir, f"{prefix.lower()}_roc.png")
            plt.savefig(roc_path)
            plt.close()
            manifest["figures"].append({"path": roc_path, "type": "plot", "status": "real"})

            prec, rec, _ = precision_recall_curve(y_true, y_prob)
            plt.figure(figsize=(6, 5), dpi=dpi)
            plt.plot(rec, prec)
            plt.title(f"{prefix} PR (AP={out['auc_pr']:.4f})")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.tight_layout()
            pr_path = os.path.join(out_dir, f"{prefix.lower()}_pr.png")
            plt.savefig(pr_path)
            plt.close()
            manifest["figures"].append({"path": pr_path, "type": "plot", "status": "real"})
        except Exception as e:
            msg = f"ROC/PR computation failed: {e}"
            roc_path = os.path.join(out_dir, f"{prefix.lower()}_roc.png")
            pr_path = os.path.join(out_dir, f"{prefix.lower()}_pr.png")
            placeholder_plot(roc_path, f"{prefix} ROC (placeholder)", msg, dpi=dpi)
            placeholder_plot(pr_path, f"{prefix} PR (placeholder)", msg, dpi=dpi)
            manifest["figures"].append({"path": roc_path, "type": "plot", "status": "placeholder", "reason": msg})
            manifest["figures"].append({"path": pr_path, "type": "plot", "status": "placeholder", "reason": msg})
    else:
        msg = "Only one class present in y_true for this split; ROC/PR are undefined."
        roc_path = os.path.join(out_dir, f"{prefix.lower()}_roc.png")
        pr_path = os.path.join(out_dir, f"{prefix.lower()}_pr.png")
        placeholder_plot(roc_path, f"{prefix} ROC (placeholder)", msg, dpi=dpi)
        placeholder_plot(pr_path, f"{prefix} PR (placeholder)", msg, dpi=dpi)
        manifest["figures"].append({"path": roc_path, "type": "plot", "status": "placeholder", "reason": msg})
        manifest["figures"].append({"path": pr_path, "type": "plot", "status": "placeholder", "reason": msg})

    return out


def plot_calibration_always(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    out_dir: str,
    dpi: int,
    prefix: str,
    manifest: Dict[str, Any]
) -> Dict[str, float]:
    ensure_dir(out_dir)
    out = {"brier": float("nan")}
    cal_path = os.path.join(out_dir, f"{prefix.lower()}_calibration.png")

    if len(np.unique(y_true)) > 1:
        try:
            out["brier"] = float(brier_score_loss(y_true, y_prob))
            frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")

            plt.figure(figsize=(6, 5), dpi=dpi)
            plt.plot(mean_pred, frac_pos, marker="o")
            plt.plot([0, 1], [0, 1], linestyle="--")
            plt.title(f"{prefix} Calibration (Brier={out['brier']:.4f})")
            plt.xlabel("Mean predicted probability")
            plt.ylabel("Fraction of positives")
            plt.tight_layout()
            plt.savefig(cal_path)
            plt.close()
            manifest["figures"].append({"path": cal_path, "type": "plot", "status": "real"})
        except Exception as e:
            msg = f"Calibration computation failed: {e}"
            placeholder_plot(cal_path, f"{prefix} Calibration (placeholder)", msg, dpi=dpi)
            manifest["figures"].append({"path": cal_path, "type": "plot", "status": "placeholder", "reason": msg})
    else:
        msg = "Only one class present in y_true; calibration is not meaningful."
        placeholder_plot(cal_path, f"{prefix} Calibration (placeholder)", msg, dpi=dpi)
        manifest["figures"].append({"path": cal_path, "type": "plot", "status": "placeholder", "reason": msg})

    return out


def plot_confusion_always(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_dir: str,
    dpi: int,
    prefix: str,
    manifest: Dict[str, Any]
) -> Dict[str, int]:
    ensure_dir(out_dir)
    cm_path = os.path.join(out_dir, f"{prefix.lower()}_confusion.png")
    try:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        plt.figure(figsize=(5, 4), dpi=dpi)
        plt.imshow(cm, interpolation="nearest")
        plt.title(f"{prefix} Confusion Matrix")
        plt.colorbar()
        ticks = [0, 1]
        plt.xticks(ticks, ["Normal", "Anomaly"])
        plt.yticks(ticks, ["Normal", "Anomaly"])
        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(cm_path)
        plt.close()
        manifest["figures"].append({"path": cm_path, "type": "plot", "status": "real"})
        return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    except Exception as e:
        msg = f"Confusion matrix plot failed: {e}"
        placeholder_plot(cm_path, f"{prefix} Confusion Matrix (placeholder)", msg, dpi=dpi)
        manifest["figures"].append({"path": cm_path, "type": "plot", "status": "placeholder", "reason": msg})
        return {"tn": 0, "fp": 0, "fn": 0, "tp": 0}


def plot_score_distributions_always(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    out_dir: str,
    dpi: int,
    prefix: str,
    manifest: Dict[str, Any]
) -> None:
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, f"{prefix.lower()}_score_dist.png")
    try:
        plt.figure(figsize=(7, 4), dpi=dpi)
        if (y_true == 0).any():
            plt.hist(y_prob[y_true == 0], bins=40, alpha=0.7, label="Normal")
        if (y_true == 1).any():
            plt.hist(y_prob[y_true == 1], bins=40, alpha=0.7, label="Anomaly")
        plt.title(f"{prefix} Predicted Probability Distributions")
        plt.xlabel("Predicted P(Anomaly)")
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        manifest["figures"].append({"path": out_path, "type": "plot", "status": "real"})
    except Exception as e:
        msg = f"Score distribution plot failed: {e}"
        placeholder_plot(out_path, f"{prefix} Score Dist (placeholder)", msg, dpi=dpi)
        manifest["figures"].append({"path": out_path, "type": "plot", "status": "placeholder", "reason": msg})


def plot_embedding_2d_always(
    X: np.ndarray,
    y: np.ndarray,
    out_dir: str,
    dpi: int,
    prefix: str,
    tsne_max_n: int,
    manifest: Dict[str, Any]
) -> None:
    ensure_dir(out_dir)

    # PCA
    pca_path = os.path.join(out_dir, f"{prefix.lower()}_pca2d.png")
    try:
        pca = PCA(n_components=2, random_state=0)
        Z = pca.fit_transform(X)
        plt.figure(figsize=(7, 5), dpi=dpi)
        plt.scatter(Z[y == 0, 0], Z[y == 0, 1], s=10, alpha=0.6, label="Normal")
        plt.scatter(Z[y == 1, 0], Z[y == 1, 1], s=10, alpha=0.6, label="Anomaly")
        plt.title(f"{prefix} PCA (2D)")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend()
        plt.tight_layout()
        plt.savefig(pca_path)
        plt.close()
        manifest["figures"].append({"path": pca_path, "type": "plot", "status": "real"})
    except Exception as e:
        msg = f"PCA 2D plot failed: {e}"
        placeholder_plot(pca_path, f"{prefix} PCA (placeholder)", msg, dpi=dpi)
        manifest["figures"].append({"path": pca_path, "type": "plot", "status": "placeholder", "reason": msg})

    # t-SNE (bounded)
    tsne_path = os.path.join(out_dir, f"{prefix.lower()}_tsne2d.png")
    try:
        n = X.shape[0]
        if n > tsne_max_n:
            rng = np.random.default_rng(123)
            idx = rng.choice(n, size=tsne_max_n, replace=False)
            Xs, ys = X[idx], y[idx]
        else:
            Xs, ys = X, y

        tsne = TSNE(n_components=2, random_state=0, init="pca", learning_rate="auto", perplexity=30)
        Zt = tsne.fit_transform(Xs)

        plt.figure(figsize=(7, 5), dpi=dpi)
        plt.scatter(Zt[ys == 0, 0], Zt[ys == 0, 1], s=10, alpha=0.6, label="Normal")
        plt.scatter(Zt[ys == 1, 0], Zt[ys == 1, 1], s=10, alpha=0.6, label="Anomaly")
        plt.title(f"{prefix} t-SNE (2D) [subsample={Xs.shape[0]}]")
        plt.xlabel("Dim1")
        plt.ylabel("Dim2")
        plt.legend()
        plt.tight_layout()
        plt.savefig(tsne_path)
        plt.close()
        manifest["figures"].append({"path": tsne_path, "type": "plot", "status": "real"})
    except Exception as e:
        msg = f"t-SNE plot failed (often heavy): {e}"
        placeholder_plot(tsne_path, f"{prefix} t-SNE (placeholder)", msg, dpi=dpi)
        manifest["figures"].append({"path": tsne_path, "type": "plot", "status": "placeholder", "reason": msg})


def threshold_sweep(y_true: np.ndarray, y_prob: np.ndarray, ngrid: int) -> Tuple[pd.DataFrame, float]:
    eps = 1e-12
    ths = np.linspace(0.0, 1.0, ngrid)
    rows = []
    best_f1 = -1.0
    best_th = 0.5
    for th in ths:
        y_pred = (y_prob >= th).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        f1 = 2 * prec * rec / (prec + rec + eps)
        rows.append({
            "threshold": float(th),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn)
        })
        if f1 > best_f1:
            best_f1 = f1
            best_th = float(th)
    return pd.DataFrame(rows), best_th


def plot_threshold_sweep_always(df_sweep: pd.DataFrame, out_dir: str, dpi: int, prefix: str, manifest: Dict[str, Any]) -> None:
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, f"{prefix.lower()}_threshold_sweep.png")
    try:
        plt.figure(figsize=(7, 4), dpi=dpi)
        plt.plot(df_sweep["threshold"], df_sweep["precision"], label="Precision")
        plt.plot(df_sweep["threshold"], df_sweep["recall"], label="Recall")
        plt.plot(df_sweep["threshold"], df_sweep["f1"], label="F1")
        plt.title(f"{prefix} Threshold Sweep")
        plt.xlabel("Threshold")
        plt.ylabel("Metric")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        manifest["figures"].append({"path": out_path, "type": "plot", "status": "real"})
    except Exception as e:
        msg = f"Threshold sweep plot failed: {e}"
        placeholder_plot(out_path, f"{prefix} Threshold Sweep (placeholder)", msg, dpi=dpi)
        manifest["figures"].append({"path": out_path, "type": "plot", "status": "placeholder", "reason": msg})


# ============================================================
# TDA helpers (bounded, with placeholders)
# ============================================================

def subsample_rows(X: np.ndarray, m: int, seed: int) -> np.ndarray:
    n = X.shape[0]
    if n <= m:
        return X
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=m, replace=False)
    return X[idx, :]


def persistence_entropy(diagram: np.ndarray) -> float:
    if diagram.size == 0:
        return 0.0
    b = diagram[:, 0]
    d = diagram[:, 1]
    finite = np.isfinite(d)
    if not np.any(finite):
        return 0.0
    lifetimes = (d[finite] - b[finite]).astype(float)
    lifetimes = lifetimes[lifetimes > 0]
    if lifetimes.size == 0:
        return 0.0
    p = lifetimes / (lifetimes.sum() + 1e-12)
    return float(-(p * np.log(p + 1e-12)).sum())


def compute_persistence(X_pointcloud: np.ndarray, cfg: Config) -> Dict[str, Any]:
    if not TDA_AVAILABLE:
        return {"tda_available": False, "reason": "ripser/persim not available"}

    dgms = ripser(X_pointcloud, maxdim=cfg.TDA_MAXDIM)["dgms"]
    summaries = []
    for dim, dgm in enumerate(dgms):
        if dgm.size == 0:
            summaries.append({
                "dim": dim, "n_features": 0,
                "total_persistence": 0.0, "mean_lifetime": 0.0, "persistence_entropy": 0.0
            })
            continue
        birth = dgm[:, 0]
        death = dgm[:, 1]
        finite = np.isfinite(death)
        lifetimes = (death[finite] - birth[finite]) if np.any(finite) else np.array([], dtype=float)
        lifetimes = lifetimes[lifetimes > 0]
        total_p = float(lifetimes.sum()) if lifetimes.size else 0.0
        mean_lt = float(lifetimes.mean()) if lifetimes.size else 0.0
        pent = persistence_entropy(dgm)
        summaries.append({
            "dim": dim,
            "n_features": int(dgm.shape[0]),
            "total_persistence": float(total_p),
            "mean_lifetime": float(mean_lt),
            "persistence_entropy": float(pent)
        })
    return {"tda_available": True, "diagrams": dgms, "summaries": summaries}


def betti_curve(diagram: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if diagram.size == 0:
        return np.zeros_like(grid, dtype=float)
    b = diagram[:, 0]
    d = diagram[:, 1]
    finite = np.isfinite(d)
    b = b[finite]
    d = d[finite]
    if b.size == 0:
        return np.zeros_like(grid, dtype=float)
    return np.array([(b <= t).sum() - (d <= t).sum() for t in grid], dtype=float)


def plot_persistence_diagrams_always(dgms: List[np.ndarray], out_png: str, title: str, dpi: int, manifest: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(out_png))
    if not TDA_AVAILABLE:
        msg = "TDA unavailable (ripser/persim missing)."
        placeholder_plot(out_png, f"{title} (placeholder)", msg, dpi=dpi)
        manifest["figures"].append({"path": out_png, "type": "plot", "status": "placeholder", "reason": msg})
        return
    try:
        plt.figure(figsize=(6, 5), dpi=dpi)
        plot_diagrams(dgms, show=False)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_png)
        plt.close()
        manifest["figures"].append({"path": out_png, "type": "plot", "status": "real"})
    except Exception as e:
        msg = f"TDA diagram plot failed: {e}"
        placeholder_plot(out_png, f"{title} (placeholder)", msg, dpi=dpi)
        manifest["figures"].append({"path": out_png, "type": "plot", "status": "placeholder", "reason": msg})


def plot_betti_curves_always(
    dgms: List[np.ndarray],
    out_png: str,
    title: str,
    ngrid: int,
    dpi: int,
    manifest: Dict[str, Any]
) -> Dict[str, Any]:
    ensure_dir(os.path.dirname(out_png))
    if not TDA_AVAILABLE:
        msg = "TDA unavailable (ripser/persim missing)."
        placeholder_plot(out_png, f"{title} (placeholder)", msg, dpi=dpi)
        manifest["figures"].append({"path": out_png, "type": "plot", "status": "placeholder", "reason": msg})
        return {"grid": np.linspace(0.0, 1.0, ngrid), "betti_H0": np.zeros(ngrid), "betti_H1": np.zeros(ngrid)}

    finite_vals = []
    for dgm in dgms[:2]:
        if dgm.size:
            b = dgm[:, 0]
            d = dgm[:, 1]
            finite = np.isfinite(d)
            finite_vals.extend(list(b[finite]))
            finite_vals.extend(list(d[finite]))

    if len(finite_vals) < 2:
        grid = np.linspace(0.0, 1.0, ngrid)
    else:
        lo = float(np.min(finite_vals))
        hi = float(np.max(finite_vals))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            grid = np.linspace(0.0, 1.0, ngrid)
        else:
            grid = np.linspace(lo, hi, ngrid)

    betti_data = {"grid": grid}
    try:
        plt.figure(figsize=(7, 4), dpi=dpi)
        for dim in range(min(2, len(dgms))):
            curve = betti_curve(dgms[dim], grid)
            betti_data[f"betti_H{dim}"] = curve
            plt.plot(grid, curve, label=f"Betti H{dim}")
        if "betti_H0" not in betti_data:
            betti_data["betti_H0"] = np.zeros_like(grid)
        if "betti_H1" not in betti_data:
            betti_data["betti_H1"] = np.zeros_like(grid)
        plt.title(title)
        plt.xlabel("Filtration value")
        plt.ylabel("Betti number")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_png)
        plt.close()
        manifest["figures"].append({"path": out_png, "type": "plot", "status": "real"})
    except Exception as e:
        msg = f"Betti plot failed: {e}"
        placeholder_plot(out_png, f"{title} (placeholder)", msg, dpi=dpi)
        manifest["figures"].append({"path": out_png, "type": "plot", "status": "placeholder", "reason": msg})
        # Still return numeric arrays for tables
        betti_data.setdefault("betti_H0", np.zeros_like(grid))
        betti_data.setdefault("betti_H1", np.zeros_like(grid))
    return betti_data


def tda_phase_report_always(
    Z: np.ndarray,
    y: np.ndarray,
    cfg: Config,
    fig_dir: str,
    prefix: str,
    seed_offset: int,
    manifest: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Always returns two DataFrames (possibly placeholders).
    Always produces the expected plots (real or placeholders).
    """
    sum_cols = ["subset", "dim", "n_features", "total_persistence", "mean_lifetime", "persistence_entropy",
                "H1_wasserstein_normal_vs_anom", "H1_bottleneck_normal_vs_anom", "NOTE"]
    betti_cols = ["grid", "normal_betti_H0", "normal_betti_H1", "anom_betti_H0", "anom_betti_H1", "NOTE"]

    if not (cfg.DO_TDA and TDA_AVAILABLE):
        note = "TDA not computed: DO_TDA is False or ripser/persim missing."
        # placeholder plots
        plot_persistence_diagrams_always([], os.path.join(fig_dir, f"{prefix}_tda_diagrams_normal.png"),
                                         f"{prefix} TDA Diagrams (Normal)", cfg.FIG_DPI, manifest)
        plot_persistence_diagrams_always([], os.path.join(fig_dir, f"{prefix}_tda_diagrams_anomalous.png"),
                                         f"{prefix} TDA Diagrams (Anomalous)", cfg.FIG_DPI, manifest)
        placeholder_plot(os.path.join(fig_dir, f"{prefix}_betti_normal.png"),
                         f"{prefix} Betti Curves (Normal) (placeholder)", note, dpi=cfg.FIG_DPI)
        manifest["figures"].append({"path": os.path.join(fig_dir, f"{prefix}_betti_normal.png"), "type": "plot", "status": "placeholder", "reason": note})
        placeholder_plot(os.path.join(fig_dir, f"{prefix}_betti_anomalous.png"),
                         f"{prefix} Betti Curves (Anomalous) (placeholder)", note, dpi=cfg.FIG_DPI)
        manifest["figures"].append({"path": os.path.join(fig_dir, f"{prefix}_betti_anomalous.png"), "type": "plot", "status": "placeholder", "reason": note})
        return empty_table(sum_cols, note), empty_table(betti_cols, note)

    Z0 = Z[y == 0]
    Z1 = Z[y == 1]
    if Z0.shape[0] < 8 or Z1.shape[0] < 8:
        note = f"TDA skipped: not enough points in subsets (normal={Z0.shape[0]}, anom={Z1.shape[0]})."
        plot_persistence_diagrams_always([], os.path.join(fig_dir, f"{prefix}_tda_diagrams_normal.png"),
                                         f"{prefix} TDA Diagrams (Normal)", cfg.FIG_DPI, manifest)
        plot_persistence_diagrams_always([], os.path.join(fig_dir, f"{prefix}_tda_diagrams_anomalous.png"),
                                         f"{prefix} TDA Diagrams (Anomalous)", cfg.FIG_DPI, manifest)
        placeholder_plot(os.path.join(fig_dir, f"{prefix}_betti_normal.png"),
                         f"{prefix} Betti Curves (Normal) (placeholder)", note, dpi=cfg.FIG_DPI)
        manifest["figures"].append({"path": os.path.join(fig_dir, f"{prefix}_betti_normal.png"), "type": "plot", "status": "placeholder", "reason": note})
        placeholder_plot(os.path.join(fig_dir, f"{prefix}_betti_anomalous.png"),
                         f"{prefix} Betti Curves (Anomalous) (placeholder)", note, dpi=cfg.FIG_DPI)
        manifest["figures"].append({"path": os.path.join(fig_dir, f"{prefix}_betti_anomalous.png"), "type": "plot", "status": "placeholder", "reason": note})
        return empty_table(sum_cols, note), empty_table(betti_cols, note)

    Z0s = subsample_rows(Z0, cfg.TDA_SUBSAMPLE, seed=cfg.SEED + 1000 + seed_offset)
    Z1s = subsample_rows(Z1, cfg.TDA_SUBSAMPLE, seed=cfg.SEED + 2000 + seed_offset)

    out0 = compute_persistence(Z0s, cfg)
    out1 = compute_persistence(Z1s, cfg)

    if not (out0.get("tda_available", False) and out1.get("tda_available", False)):
        note = "TDA failed during persistence computation."
        plot_persistence_diagrams_always([], os.path.join(fig_dir, f"{prefix}_tda_diagrams_normal.png"),
                                         f"{prefix} TDA Diagrams (Normal)", cfg.FIG_DPI, manifest)
        plot_persistence_diagrams_always([], os.path.join(fig_dir, f"{prefix}_tda_diagrams_anomalous.png"),
                                         f"{prefix} TDA Diagrams (Anomalous)", cfg.FIG_DPI, manifest)
        placeholder_plot(os.path.join(fig_dir, f"{prefix}_betti_normal.png"),
                         f"{prefix} Betti Curves (Normal) (placeholder)", note, dpi=cfg.FIG_DPI)
        manifest["figures"].append({"path": os.path.join(fig_dir, f"{prefix}_betti_normal.png"), "type": "plot", "status": "placeholder", "reason": note})
        placeholder_plot(os.path.join(fig_dir, f"{prefix}_betti_anomalous.png"),
                         f"{prefix} Betti Curves (Anomalous) (placeholder)", note, dpi=cfg.FIG_DPI)
        manifest["figures"].append({"path": os.path.join(fig_dir, f"{prefix}_betti_anomalous.png"), "type": "plot", "status": "placeholder", "reason": note})
        return empty_table(sum_cols, note), empty_table(betti_cols, note)

    # Plots (always)
    plot_persistence_diagrams_always(out0["diagrams"], os.path.join(fig_dir, f"{prefix}_tda_diagrams_normal.png"),
                                     f"{prefix} TDA Diagrams (Normal)", cfg.FIG_DPI, manifest)
    plot_persistence_diagrams_always(out1["diagrams"], os.path.join(fig_dir, f"{prefix}_tda_diagrams_anomalous.png"),
                                     f"{prefix} TDA Diagrams (Anomalous)", cfg.FIG_DPI, manifest)

    betti0 = plot_betti_curves_always(out0["diagrams"], os.path.join(fig_dir, f"{prefix}_betti_normal.png"),
                                      f"{prefix} Betti Curves (Normal)", cfg.BETTI_NGRID, cfg.FIG_DPI, manifest)
    betti1 = plot_betti_curves_always(out1["diagrams"], os.path.join(fig_dir, f"{prefix}_betti_anomalous.png"),
                                      f"{prefix} Betti Curves (Anomalous)", cfg.BETTI_NGRID, cfg.FIG_DPI, manifest)

    # Summary table
    rows = []
    for s in out0["summaries"]:
        rows.append({"subset": "normal", **s})
    for s in out1["summaries"]:
        rows.append({"subset": "anomalous", **s})
    sum_df = pd.DataFrame(rows)

    # Distances (optional)
    h1_w = np.nan
    h1_b = np.nan
    if (WASSERSTEIN_AVAILABLE or BOTTLENECK_AVAILABLE) and len(out0["diagrams"]) > 1 and len(out1["diagrams"]) > 1:
        try:
            if WASSERSTEIN_AVAILABLE:
                h1_w = float(wasserstein(out0["diagrams"][1], out1["diagrams"][1]))
            if BOTTLENECK_AVAILABLE:
                h1_b = float(bottleneck(out0["diagrams"][1], out1["diagrams"][1]))
        except Exception:
            pass
    sum_df["H1_wasserstein_normal_vs_anom"] = h1_w
    sum_df["H1_bottleneck_normal_vs_anom"] = h1_b
    sum_df["NOTE"] = ""

    # Betti table
    grid = betti0.get("grid", np.linspace(0.0, 1.0, cfg.BETTI_NGRID))
    bdf = pd.DataFrame({
        "grid": grid.astype(float),
        "normal_betti_H0": np.asarray(betti0.get("betti_H0", np.zeros_like(grid))).astype(float),
        "normal_betti_H1": np.asarray(betti0.get("betti_H1", np.zeros_like(grid))).astype(float),
        "anom_betti_H0": np.asarray(betti1.get("betti_H0", np.zeros_like(grid))).astype(float),
        "anom_betti_H1": np.asarray(betti1.get("betti_H1", np.zeros_like(grid))).astype(float),
        "NOTE": ""
    })
    return sum_df, bdf


def tda_time_evolution_always(
    Z: np.ndarray,
    win_starts: np.ndarray,
    cfg: Config,
    fig_dir: str,
    tag: str,
    manifest: Dict[str, Any]
) -> pd.DataFrame:
    """
    Always returns a DataFrame (possibly placeholder).
    Always produces evolution plots (real or placeholders).
    """
    evo_cols = [
        "window_start", "n_points",
        "H0_n", "H0_total_persistence", "H0_mean_lifetime", "H0_entropy",
        "H1_n", "H1_total_persistence", "H1_mean_lifetime", "H1_entropy",
        "H1_wasserstein_prev", "H1_bottleneck_prev", "NOTE"
    ]

    out_persist = os.path.join(fig_dir, f"tda_time_evolution_{tag}_H1_total_persistence.png")
    out_wass = os.path.join(fig_dir, f"tda_time_evolution_{tag}_H1_wasserstein.png")

    if not (cfg.DO_TDA and TDA_AVAILABLE):
        note = "TDA time evolution not computed: DO_TDA is False or TDA unavailable."
        placeholder_plot(out_persist, f"TDA Time Evolution ({tag}) (placeholder)", note, dpi=cfg.FIG_DPI)
        placeholder_plot(out_wass, f"TDA Wasserstein Evolution ({tag}) (placeholder)", note, dpi=cfg.FIG_DPI)
        manifest["figures"].append({"path": out_persist, "type": "plot", "status": "placeholder", "reason": note})
        manifest["figures"].append({"path": out_wass, "type": "plot", "status": "placeholder", "reason": note})
        return empty_table(evo_cols, note)

    try:
        order = np.argsort(win_starts)
        Zs = Z[order]
        ws = win_starts[order]
        stride = int(cfg.TDA_STRIDE_WINDOWS)

        rows = []
        prev_dgms = None

        for i in range(0, Zs.shape[0], stride):
            block = Zs[i:i + stride]
            if block.shape[0] < 5:
                continue
            Xpc = subsample_rows(block, cfg.TDA_SUBSAMPLE, seed=cfg.SEED + i)
            out = compute_persistence(Xpc, cfg)
            if not out.get("tda_available", False):
                continue

            dgms = out["diagrams"]
            sums = out["summaries"]
            row = {"window_start": int(ws[i]), "n_points": int(Xpc.shape[0])}
            # initialize
            for dim in [0, 1]:
                row[f"H{dim}_n"] = 0
                row[f"H{dim}_total_persistence"] = 0.0
                row[f"H{dim}_mean_lifetime"] = 0.0
                row[f"H{dim}_entropy"] = 0.0
            for s in sums:
                dim = int(s.get("dim", 0))
                row[f"H{dim}_n"] = int(s.get("n_features", 0))
                row[f"H{dim}_total_persistence"] = float(s.get("total_persistence", 0.0))
                row[f"H{dim}_mean_lifetime"] = float(s.get("mean_lifetime", 0.0))
                row[f"H{dim}_entropy"] = float(s.get("persistence_entropy", 0.0))

            row["H1_wasserstein_prev"] = np.nan
            row["H1_bottleneck_prev"] = np.nan

            if cfg.DO_TDA_DISTANCES and prev_dgms is not None and (WASSERSTEIN_AVAILABLE or BOTTLENECK_AVAILABLE):
                try:
                    if WASSERSTEIN_AVAILABLE and len(dgms) > 1 and len(prev_dgms) > 1:
                        row["H1_wasserstein_prev"] = float(wasserstein(prev_dgms[1], dgms[1]))
                    if BOTTLENECK_AVAILABLE and len(dgms) > 1 and len(prev_dgms) > 1:
                        row["H1_bottleneck_prev"] = float(bottleneck(prev_dgms[1], dgms[1]))
                except Exception:
                    pass

            row["NOTE"] = ""
            prev_dgms = dgms
            rows.append(row)

        evo_df = pd.DataFrame(rows)
        if len(evo_df) == 0:
            note = "TDA time evolution produced no blocks (insufficient windows or persistence errors)."
            placeholder_plot(out_persist, f"TDA Time Evolution ({tag}) (placeholder)", note, dpi=cfg.FIG_DPI)
            placeholder_plot(out_wass, f"TDA Wasserstein Evolution ({tag}) (placeholder)", note, dpi=cfg.FIG_DPI)
            manifest["figures"].append({"path": out_persist, "type": "plot", "status": "placeholder", "reason": note})
            manifest["figures"].append({"path": out_wass, "type": "plot", "status": "placeholder", "reason": note})
            return empty_table(evo_cols, note)

        # Plot curves (always)
        plt.figure(figsize=(8, 4), dpi=cfg.FIG_DPI)
        plt.plot(evo_df["window_start"], evo_df["H1_total_persistence"])
        plt.title(f"TDA Time Evolution ({tag}): Total Persistence (H1)")
        plt.xlabel("Time (window start index)")
        plt.ylabel("Total persistence H1")
        plt.tight_layout()
        plt.savefig(out_persist)
        plt.close()
        manifest["figures"].append({"path": out_persist, "type": "plot", "status": "real"})

        if "H1_wasserstein_prev" in evo_df.columns and evo_df["H1_wasserstein_prev"].notna().any():
            plt.figure(figsize=(8, 4), dpi=cfg.FIG_DPI)
            plt.plot(evo_df["window_start"], evo_df["H1_wasserstein_prev"])
            plt.title(f"TDA Time Evolution ({tag}): Wasserstein Distance (H1, consecutive)")
            plt.xlabel("Time (window start index)")
            plt.ylabel("Wasserstein distance")
            plt.tight_layout()
            plt.savefig(out_wass)
            plt.close()
            manifest["figures"].append({"path": out_wass, "type": "plot", "status": "real"})
        else:
            note = "Wasserstein distances not available or all-NaN (missing persim distances)."
            placeholder_plot(out_wass, f"TDA Wasserstein Evolution ({tag}) (placeholder)", note, dpi=cfg.FIG_DPI)
            manifest["figures"].append({"path": out_wass, "type": "plot", "status": "placeholder", "reason": note})

        # Ensure expected columns exist
        for c in evo_cols:
            if c not in evo_df.columns:
                evo_df[c] = np.nan if c != "NOTE" else ""
        evo_df = evo_df[evo_cols]
        return evo_df

    except Exception as e:
        note = f"TDA time evolution failed: {e}"
        placeholder_plot(out_persist, f"TDA Time Evolution ({tag}) (placeholder)", note, dpi=cfg.FIG_DPI)
        placeholder_plot(out_wass, f"TDA Wasserstein Evolution ({tag}) (placeholder)", note, dpi=cfg.FIG_DPI)
        manifest["figures"].append({"path": out_persist, "type": "plot", "status": "placeholder", "reason": note})
        manifest["figures"].append({"path": out_wass, "type": "plot", "status": "placeholder", "reason": note})
        return empty_table(evo_cols, note)


# ============================================================
# Verified Self-generation + Selection + Curriculum (bounded)
# ============================================================

def self_generate_candidates(X_base: np.ndarray, cfg: Config) -> np.ndarray:
    rng = np.random.default_rng(cfg.SEED + 777)
    n = X_base.shape[0]
    m = int(cfg.SELFGEN_CANDIDATES)
    idx = rng.choice(n, size=m, replace=True)
    Xc = X_base[idx].copy()
    noise = rng.standard_normal(Xc.shape) * float(cfg.SELFGEN_NOISE_STD)
    Xc = Xc + noise
    Xc = np.clip(Xc, -float(cfg.SELFGEN_CLIP_Z), float(cfg.SELFGEN_CLIP_Z))
    return Xc


def build_train_quantile_bounds(X_train_z: np.ndarray, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    ql = float(cfg.VERIFIER_Q_LOW)
    qh = float(cfg.VERIFIER_Q_HIGH)
    low = np.quantile(X_train_z, ql, axis=0)
    high = np.quantile(X_train_z, qh, axis=0)
    low = np.where(np.isfinite(low), low, -np.inf)
    high = np.where(np.isfinite(high), high, np.inf)
    return low, high


def verify_candidates(
    Xc: np.ndarray,
    X_train_z: np.ndarray,
    cfg: Config
) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.ones(Xc.shape[0], dtype=bool)
    mask &= np.isfinite(Xc).all(axis=1)

    if cfg.VERIFIER_MAX_ABS_Z is not None:
        mask &= (np.max(np.abs(Xc), axis=1) <= float(cfg.VERIFIER_MAX_ABS_Z))

    l2 = np.linalg.norm(Xc, axis=1)
    mask &= (l2 <= float(cfg.VERIFIER_MAX_L2_Z))

    if cfg.VERIFIER_USE_TRAIN_QUANTILES:
        low, high = build_train_quantile_bounds(X_train_z, cfg)
        mask &= (Xc >= low[None, :]).all(axis=1)
        mask &= (Xc <= high[None, :]).all(axis=1)

    Xv = Xc[mask]
    return Xv, mask


def compute_uncertainty_scores(p: np.ndarray) -> np.ndarray:
    return 1.0 - 2.0 * np.abs(p - 0.5)


def topology_proxy_stress(Z_train: np.ndarray, y_train: np.ndarray, Z_candidates: np.ndarray) -> np.ndarray:
    Zn = Z_train[y_train == 0]
    if Zn.shape[0] < 8:
        c = Z_train.mean(axis=0, keepdims=True)
        d = np.linalg.norm(Z_candidates - c, axis=1)
        return (d - d.mean()) / (d.std() + 1e-12)

    c = Zn.mean(axis=0, keepdims=True)
    dn = np.linalg.norm(Zn - c, axis=1)
    dc = np.linalg.norm(Z_candidates - c, axis=1)
    mu = dn.mean()
    sd = dn.std() + 1e-12
    return (dc - mu) / sd


def limited_tda_delta_scores(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_candidates: np.ndarray,
    cfg: Config
) -> np.ndarray:
    nC = Z_candidates.shape[0]
    scores = np.full(nC, np.nan, dtype=float)
    if not (cfg.DO_TDA and TDA_AVAILABLE):
        return scores

    Zn = Z_train[y_train == 0]
    if Zn.shape[0] < 12:
        return scores

    base = subsample_rows(Zn, m=min(int(cfg.TDA_SUBSAMPLE), 140), seed=cfg.SEED + 9999)
    base_out = compute_persistence(base, cfg)
    if not base_out.get("tda_available", False):
        return scores

    def _get_h1_stats(out: Dict[str, Any]) -> Tuple[float, float]:
        tot, ent = 0.0, 0.0
        for s in out.get("summaries", []):
            if int(s.get("dim", -1)) == 1:
                tot = float(s.get("total_persistence", 0.0))
                ent = float(s.get("persistence_entropy", 0.0))
        return tot, ent

    base_tot, base_ent = _get_h1_stats(base_out)

    for i in range(nC):
        aug = np.vstack([base, Z_candidates[i:i + 1]])
        out = compute_persistence(aug, cfg)
        if not out.get("tda_available", False):
            continue
        tot, ent = _get_h1_stats(out)
        scores[i] = abs(tot - base_tot) + abs(ent - base_ent)

    finite = np.isfinite(scores)
    if np.any(finite):
        s = scores[finite]
        mu = float(np.median(s))
        mad = float(np.median(np.abs(s - mu)) + 1e-12)
        scores[finite] = (scores[finite] - mu) / (1.4826 * mad + 1e-12)
    return scores


def select_candidates(
    Xv: np.ndarray,
    p_cand: np.ndarray,
    Z_train: Optional[np.ndarray],
    y_train: Optional[np.ndarray],
    Z_cand: Optional[np.ndarray],
    cfg: Config
) -> Tuple[np.ndarray, pd.DataFrame]:
    u = compute_uncertainty_scores(p_cand)

    stress = np.zeros_like(u)
    if Z_train is not None and y_train is not None and Z_cand is not None and Z_train.shape[0] > 0:
        stress = topology_proxy_stress(Z_train, y_train, Z_cand)
        mu = float(np.median(stress))
        mad = float(np.median(np.abs(stress - mu)) + 1e-12)
        stress = (stress - mu) / (1.4826 * mad + 1e-12)

    score = float(cfg.SELECTOR_ALPHA_UNCERT) * u + float(cfg.SELECTOR_BETA_STRESS) * stress

    tda_score = np.full_like(score, np.nan)
    if cfg.SELECTOR_USE_TDA_ON_TOPM and cfg.DO_TDA and TDA_AVAILABLE and Z_train is not None and y_train is not None and Z_cand is not None:
        M = int(min(cfg.SELECTOR_TDA_TOPM, Xv.shape[0]))
        topM_idx = np.argsort(-score)[:M]
        tda_local = limited_tda_delta_scores(Z_train, y_train, Z_cand[topM_idx], cfg)
        tda_score[topM_idx] = tda_local
        finite = np.isfinite(tda_score)
        score[finite] = score[finite] + float(cfg.SELECTOR_GAMMA_TDA) * tda_score[finite]

    K = int(min(cfg.SELFGEN_TOPK, Xv.shape[0]))
    sel_idx = np.argsort(-score)[:K]

    audit_df = pd.DataFrame({
        "cand_index": np.arange(Xv.shape[0]),
        "p_anomaly": p_cand.astype(float),
        "uncertainty": u.astype(float),
        "stress_proxy": stress.astype(float),
        "tda_delta_score": tda_score.astype(float),
        "final_score": score.astype(float),
        "selected": False
    })
    audit_df.loc[sel_idx, "selected"] = True
    audit_df = audit_df.sort_values("final_score", ascending=False).reset_index(drop=True)

    return sel_idx, audit_df


# ============================================================
# Export helpers (always create Excel with all sheets)
# ============================================================

def export_tables_always(
    cfg: Config,
    rep_dir: str,
    tables: Dict[str, pd.DataFrame],
    sheet_order: List[str],
    summary: Dict[str, Any],
    manifest: Dict[str, Any]
) -> None:
    ensure_dir(rep_dir)

    # CSVs (always attempt)
    if cfg.SAVE_CSV:
        for name, df in tables.items():
            csv_path = os.path.join(rep_dir, f"{name}.csv")
            try:
                sanitize_dataframe_for_export(df).to_csv(csv_path, index=False)
                manifest["reports"].append({"path": csv_path, "type": "csv", "status": "real"})
            except Exception as e:
                # create a minimal fallback CSV as placeholder
                fallback = pd.DataFrame({"NOTE": [f"CSV export failed for {name}: {e}"]})
                fallback.to_csv(csv_path, index=False)
                manifest["reports"].append({"path": csv_path, "type": "csv", "status": "placeholder", "reason": str(e)})

    # Excel (single file, all sheets in order)
    xlsx_path = os.path.join(rep_dir, "report_tables.xlsx")
    if cfg.SAVE_EXCEL:
        try:
            with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
                for sheet_name in sheet_order:
                    df = tables.get(sheet_name, None)
                    if df is None:
                        df = pd.DataFrame({"NOTE": [f"Sheet '{sheet_name}' missing from tables dict; placeholder created."]})
                    sanitize_dataframe_for_export(df).to_excel(writer, sheet_name=sheet_name[:31], index=False)
            manifest["reports"].append({"path": xlsx_path, "type": "excel", "status": "real"})
        except Exception as e:
            # Fallback: write a minimal Excel-like CSV set only; still keep placeholder note.
            note_path = os.path.join(rep_dir, "report_tables_excel_failed.txt")
            with open(note_path, "w", encoding="utf-8") as f:
                f.write(f"Excel export failed: {e}\n")
            manifest["reports"].append({"path": note_path, "type": "txt", "status": "placeholder", "reason": str(e)})

    # JSON summary
    if cfg.SAVE_JSON:
        summary_path = os.path.join(rep_dir, "summary.json")
        save_json(summary_path, summary)
        manifest["reports"].append({"path": summary_path, "type": "json", "status": "real"})

    # Manifest JSON
    manifest_path = os.path.join(rep_dir, "manifest.json")
    save_json(manifest_path, manifest)
    # also list manifest in itself for convenience
    manifest["reports"].append({"path": manifest_path, "type": "json", "status": "real"})


def save_models_always(
    cfg: Config,
    mod_dir: str,
    rf_model: Optional[RandomForestClassifier],
    lr_model: Optional[LogisticRegression],
    scaler: Optional[StandardScaler],
    pca_proxy: Optional[PCA],
    deep_model: Any,
    encoder: Any,
    manifest: Dict[str, Any],
    fallback_models: Any = None
) -> None:
    ensure_dir(mod_dir)

    # Save config snapshot
    cfg_path = os.path.join(mod_dir, "config_snapshot.json")
    save_json(cfg_path, asdict(cfg))
    manifest["models"].append({"path": cfg_path, "type": "json", "status": "real"})

    # Save sklearn objects with joblib if available; else placeholders
    def _joblib_save(obj, path, name):
        if JOBLIB_AVAILABLE and obj is not None:
            try:
                joblib.dump(obj, path)
                manifest["models"].append({"path": path, "type": "joblib", "status": "real", "name": name})
                return
            except Exception as e:
                note = f"joblib dump failed for {name}: {e}"
        else:
            note = f"joblib unavailable or object None for {name}"
        # placeholder note
        with open(path + ".txt", "w", encoding="utf-8") as f:
            f.write(note + "\n")
        manifest["models"].append({"path": path + ".txt", "type": "txt", "status": "placeholder", "name": name, "reason": note})

    _joblib_save(scaler, os.path.join(mod_dir, "scaler.joblib"), "scaler")
    _joblib_save(pca_proxy, os.path.join(mod_dir, "pca_proxy.joblib"), "pca_proxy")
    _joblib_save(rf_model, os.path.join(mod_dir, "rf_model.joblib"), "rf_model")
    _joblib_save(lr_model, os.path.join(mod_dir, "logreg_model.joblib"), "logreg_model")

    # Save deep model if available; else placeholder
    deep_path = os.path.join(mod_dir, "deep_multitask_model.keras")
    enc_path = os.path.join(mod_dir, "encoder.keras")
    if cfg.DO_DEEP and TF_AVAILABLE and deep_model is not None:
        try:
            # Save base model to keep serialization robust even when using a training wrapper
            model_to_save = deep_model
            try:
                # GeoMultiTaskModel wraps a Functional/Sequential base model in .base
                if hasattr(deep_model, "base"):
                    model_to_save = getattr(deep_model, "base")
            except Exception:
                model_to_save = deep_model
            tf.keras.models.save_model(model_to_save, deep_path, include_optimizer=False)
            manifest["models"].append({"path": deep_path, "type": "keras", "status": "real", "name": "deep_model"})
        except Exception as e:
            note = f"Deep model save failed: {e}"
            with open(deep_path + ".txt", "w", encoding="utf-8") as f:
                f.write(note + "\n")
            manifest["models"].append({"path": deep_path + ".txt", "type": "txt", "status": "placeholder", "name": "deep_model", "reason": note})
        # Save encoder as standalone if possible
        try:
            encoder.save(enc_path)
            manifest["models"].append({"path": enc_path, "type": "keras", "status": "real", "name": "encoder"})
        except Exception as e:
            note = f"Encoder save failed: {e}"
            with open(enc_path + ".txt", "w", encoding="utf-8") as f:
                f.write(note + "\n")
            manifest["models"].append({"path": enc_path + ".txt", "type": "txt", "status": "placeholder", "name": "encoder", "reason": note})
    else:
        if fallback_models is not None:
            try:
                _joblib_save(fallback_models, deep_path, "deep_fallback_models")
                manifest["models"].append({"path": deep_path, "type": "joblib", "status": "ok", "name": "deep_fallback_models"})
                note_enc = "Encoder not saved (fallback deep pipeline)."
                with open(enc_path, "w", encoding="utf-8") as f:
                    f.write(note_enc + "\n")
                manifest["models"].append({"path": enc_path, "type": "text", "status": "note", "note": note_enc})
                manifest["notes"].append("Deep fallback models saved to deep_multitask_model.keras via joblib.")
                return
                # Continue to also write placeholders below (keeps compatibility), but deep_path now contains a real artifact.
            except Exception as e:
                manifest["notes"].append(f"Failed to save deep fallback models: {e}")
        note = "Deep model not saved because TF is unavailable or DO_DEEP=False."
        with open(deep_path + ".txt", "w", encoding="utf-8") as f:
            f.write(note + "\n")
        with open(enc_path + ".txt", "w", encoding="utf-8") as f:
            f.write(note + "\n")
        manifest["models"].append({"path": deep_path + ".txt", "type": "txt", "status": "placeholder", "name": "deep_model", "reason": note})
        manifest["models"].append({"path": enc_path + ".txt", "type": "txt", "status": "placeholder", "name": "encoder", "reason": note})


# ============================================================
# Pipeline
# ============================================================

def run_pipeline(cfg: Config) -> None:
    apply_word_compat_preset(cfg)
    set_seed(cfg.SEED)

    # Prepare output folders
    # Use a unique output directory per run to avoid Windows/Excel file locks (PermissionError)
    if getattr(cfg, 'RUN_UNIQUE_DIR', True):
        cfg.OUTPUT_DIR = f"{cfg.OUTPUT_DIR}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ensure_dir(cfg.OUTPUT_DIR)
    fig_dir = os.path.join(cfg.OUTPUT_DIR, cfg.FIG_DIR)
    rep_dir = os.path.join(cfg.OUTPUT_DIR, cfg.REP_DIR)
    mod_dir = os.path.join(cfg.OUTPUT_DIR, cfg.MOD_DIR)
    ensure_dir(fig_dir)
    ensure_dir(rep_dir)
    ensure_dir(mod_dir)

    # Manifest structure
    manifest: Dict[str, Any] = {
        "created_at_unix": time.time(),
        "environment": {
            "TF_AVAILABLE": bool(TF_AVAILABLE),
            "TDA_AVAILABLE": bool(TDA_AVAILABLE),
            "WASSERSTEIN_AVAILABLE": bool(WASSERSTEIN_AVAILABLE),
            "BOTTLENECK_AVAILABLE": bool(BOTTLENECK_AVAILABLE),
            "JOBLIB_AVAILABLE": bool(JOBLIB_AVAILABLE),
        },
        "figures": [],
        "reports": [],
        "models": [],
        "notes": []
    }

    t0 = time.time()

    # -----------------------------
    # 1) Data
    # -----------------------------
    dfX, y_point = load_or_generate_data(cfg)

    # -----------------------------
    # 2) Window-level dataset
    # -----------------------------
    Xw, yw, win_starts, feat_names = make_windows_stats(dfX, y_point, cfg)

    # Temporal split (anti-leakage)
    Xtr_raw, Xte_raw, ytr, yte, wtr, wte = temporal_split_windows(Xw, yw, win_starts, cfg.TEST_SIZE)

    # Scaling: fit only on train
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr_raw)
    Xte = scaler.transform(Xte_raw)
    Xw_scaled = scaler.transform(Xw)

    windows_index_df = pd.DataFrame({
        "window_index": np.arange(len(yw)),
        "window_start": win_starts,
        "y_window": yw
    }).sort_values("window_start").reset_index(drop=True)

    # PCA proxy always (saved)
    pca_proxy = PCA(n_components=min(8, Xtr.shape[1]), random_state=0)
    Ztr_proxy = pca_proxy.fit_transform(Xtr)
    Zall_proxy = pca_proxy.transform(Xw_scaled)

    # -----------------------------
    # 3) Train models (baselines always attempted)
    # -----------------------------
    probas: Dict[str, np.ndarray] = {}
    preds: Dict[str, np.ndarray] = {}
    reports: Dict[str, pd.DataFrame] = {}
    rf_model = None
    lr_model = None

    if cfg.DO_RF:
        try:
            rf_model = build_rf(cfg)
            rf_model.fit(Xtr, ytr)
            y_prob = rf_model.predict_proba(Xte)[:, 1] if hasattr(rf_model, "predict_proba") else safe_sigmoid(rf_model.decision_function(Xte))
            y_pred = (y_prob >= 0.5).astype(int)
            probas["RF"] = y_prob
            preds["RF"] = y_pred
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rep = classification_report(yte, y_pred, output_dict=True, zero_division=0)
            reports["RF"] = pd.DataFrame(rep).T.reset_index().rename(columns={"index": "class"})
        except Exception as e:
            manifest["notes"].append(f"RF training failed: {e}")

    if cfg.DO_LOGREG:
        try:
            lr_model = build_logreg(cfg)
            lr_model.fit(Xtr, ytr)
            y_prob = lr_model.predict_proba(Xte)[:, 1] if hasattr(lr_model, "predict_proba") else safe_sigmoid(lr_model.decision_function(Xte))
            y_pred = (y_prob >= 0.5).astype(int)
            probas["LOGREG"] = y_prob
            preds["LOGREG"] = y_pred
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rep = classification_report(yte, y_pred, output_dict=True, zero_division=0)
            reports["LOGREG"] = pd.DataFrame(rep).T.reset_index().rename(columns={"index": "class"})
        except Exception as e:
            manifest["notes"].append(f"LogReg training failed: {e}")

    # Choose primary pre model
    primary_pre = "RF" if "RF" in probas else ("LOGREG" if "LOGREG" in probas else None)

    # -----------------------------
    # 4) Verified self-generation + selection
    # -----------------------------
    selfgen_audit_df = None
    selected_for_curriculum = None

    if cfg.DO_SELFGEN and primary_pre is not None:
        try:
            Xc = self_generate_candidates(Xtr, cfg)
            Xv, _mask = verify_candidates(Xc, Xtr, cfg)
            if Xv.shape[0] == 0:
                raise RuntimeError("All candidates rejected by verifier; consider relaxing VERIFIER_* constraints.")

            # Candidate probabilities from primary baseline
            if primary_pre == "RF" and rf_model is not None:
                p_cand = rf_model.predict_proba(Xv)[:, 1]
            elif primary_pre == "LOGREG" and lr_model is not None:
                p_cand = lr_model.predict_proba(Xv)[:, 1]
            else:
                p_cand = np.full(Xv.shape[0], 0.5, dtype=float)

            # Candidate latent proxy
            Zc_proxy = pca_proxy.transform(Xv)

            sel_idx, audit_df = select_candidates(
                Xv=Xv,
                p_cand=p_cand,
                Z_train=Ztr_proxy,
                y_train=ytr,
                Z_cand=Zc_proxy,
                cfg=cfg
            )
            selfgen_audit_df = audit_df.copy()
            selected_for_curriculum = Xv[sel_idx].copy()
        except Exception as e:
            manifest["notes"].append(f"Self-generation failed: {e}")
            # Create placeholder selfgen table
            selfgen_audit_df = empty_table(
                ["cand_index", "p_anomaly", "uncertainty", "stress_proxy", "tda_delta_score", "final_score", "selected", "NOTE"],
                note=f"Self-generation failed: {e}"
            )
            selected_for_curriculum = None
    else:
        note = "Self-generation disabled or no baseline model available."
        selfgen_audit_df = empty_table(
            ["cand_index", "p_anomaly", "uncertainty", "stress_proxy", "tda_delta_score", "final_score", "selected", "NOTE"],
            note=note
        )

    # -----------------------------
    # 5) Deep model (always produce deep outputs or placeholders)
    # -----------------------------
    deep_model = None
    encoder = None
    latent_all = None
    recon_mse_all = None
    deep_history_df = None
    deep_epoch_monitor_df = None
    deep_recon_examples_df = None
    tda_over_epochs_df = None

    def _make_raw_windows(dfX_scaled_pointwise: np.ndarray, y_point: np.ndarray, window_size: int, hop: int, label_ratio: float):
        # Build raw windows: [nW, T, F]
        n = dfX_scaled_pointwise.shape[0]
        F = dfX_scaled_pointwise.shape[1]
        starts = np.arange(0, n - window_size + 1, hop, dtype=int)
        Xraw = np.zeros((len(starts), window_size, F), dtype=np.float32)
        yw = np.zeros(len(starts), dtype=int)
        for k, s in enumerate(starts):
            seg = dfX_scaled_pointwise[s:s + window_size]
            Xraw[k] = seg
            frac = float(np.mean(y_point[s:s + window_size]))
            yw[k] = 1 if frac >= float(label_ratio) else 0
        return Xraw, yw, starts

    if cfg.DO_DEEP:
        try:
            deep_mode = str(getattr(cfg, "DEEP_INPUT_MODE", "STATS_MLP")).upper().strip()
            if not TF_AVAILABLE:
                raise RuntimeError('TF_UNAVAILABLE')

            # ----------------------------------------------------
            # (A) Prepare deep inputs depending on mode
            # ----------------------------------------------------
            if deep_mode == "RAW_CONV1D":
                # Pointwise scaling then windowing
                scaler_raw = StandardScaler()
                X_point_tr = dfX.iloc[:split_idx].to_numpy(dtype=float)
                scaler_raw.fit(X_point_tr)
                X_point_all_scaled = scaler_raw.transform(dfX.to_numpy(dtype=float))

                Xw_raw, yw_raw, win_starts_raw = _make_raw_windows(
                    X_point_all_scaled.astype(np.float32), y_point=y, window_size=int(cfg.RAW_WINDOW_SIZE),
                    hop=int(cfg.RAW_WINDOW_HOP), label_ratio=float(cfg.WINDOW_LABEL_RATIO)
                )

                # Align with existing temporal split (using window starts)
                # Map starts to train/test by comparing start index vs split_idx
                is_train = win_starts_raw < split_idx
                Xtr_deep = Xw_raw[is_train]
                ytr_deep = yw_raw[is_train]
                Xte_deep = Xw_raw[~is_train]
                yte_deep = yw_raw[~is_train]

                # For downstream latent-all & recon-mse-all, use all windows
                Xall_deep = Xw_raw
                yall_deep = yw_raw

                # Save scaler_raw
                try:
                    if JOBLIB_AVAILABLE:
                        joblib.dump(scaler_raw, os.path.join(mod_dir, "scaler_raw.joblib"))
                        manifest["models"].append({"path": os.path.join(mod_dir, "scaler_raw.joblib"), "type": "scaler", "status": "real", "name": "scaler_raw"})
                except Exception as _e:
                    manifest["notes"].append(f"Could not save scaler_raw: {_e}")

                # Build model
                deep_model, encoder = build_deep_multitask(input_dim=Xtr_deep.shape[-1], cfg=cfg)

            else:
                # Default uses window-statistics vectors already built: Xtr, ytr, Xte, yte, Xw_scaled
                Xtr_deep = Xtr.copy()
                ytr_deep = ytr.copy()
                Xte_deep = Xte.copy()
                yte_deep = yte.copy()
                Xall_deep = Xw_scaled
                yall_deep = yw

                deep_model, encoder = build_deep_multitask(Xtr.shape[1], cfg)

            # ----------------------------------------------------
            # (B) Curriculum injection (bounded, preserves prior logic)
            # ----------------------------------------------------
            if cfg.CURRICULUM_MODE.upper() == "ONCE" and selected_for_curriculum is not None and selected_for_curriculum.shape[0] > 0 and deep_mode != "RAW_CONV1D":
                Kinj = int(min(cfg.CURRICULUM_INJECT_TOPK, selected_for_curriculum.shape[0]))
                Xinj = selected_for_curriculum[:Kinj]

                if cfg.CURRICULUM_LABEL_POLICY.upper() == "ANOMALY":
                    yinj = np.ones(Kinj, dtype=int)
                else:
                    if primary_pre == "RF" and rf_model is not None:
                        pinj = rf_model.predict_proba(Xinj)[:, 1]
                    elif primary_pre == "LOGREG" and lr_model is not None:
                        pinj = lr_model.predict_proba(Xinj)[:, 1]
                    else:
                        pinj = np.full(Kinj, 0.5, dtype=float)
                    yinj = (pinj >= 0.5).astype(int)

                Xtr_deep = np.vstack([Xtr_deep, Xinj])
                ytr_deep = np.concatenate([ytr_deep, yinj])

            if cfg.CURRICULUM_MODE.upper() == "PER_EPOCH" and selected_for_curriculum is not None and selected_for_curriculum.shape[0] > 0 and deep_mode != "RAW_CONV1D":
                E = int(min(cfg.CURRICULUM_MAX_EPOCHS, cfg.DEEP_EPOCHS))
                k = int(max(0, cfg.CURRICULUM_PER_EPOCH_K))
                if k > 0:
                    rng = np.random.default_rng(cfg.SEED + 12345)
                    pool = selected_for_curriculum
                    for _ in range(E):
                        idx = rng.choice(pool.shape[0], size=min(k, pool.shape[0]), replace=False)
                        Xinj = pool[idx]
                        if cfg.CURRICULUM_LABEL_POLICY.upper() == "ANOMALY":
                            yinj = np.ones(Xinj.shape[0], dtype=int)
                        else:
                            if primary_pre == "RF" and rf_model is not None:
                                pinj = rf_model.predict_proba(Xinj)[:, 1]
                            elif primary_pre == "LOGREG" and lr_model is not None:
                                pinj = lr_model.predict_proba(Xinj)[:, 1]
                            else:
                                pinj = np.full(Xinj.shape[0], 0.5, dtype=float)
                            yinj = (pinj >= 0.5).astype(int)
                        Xtr_deep = np.vstack([Xtr_deep, Xinj])
                        ytr_deep = np.concatenate([ytr_deep, yinj])

            # ----------------------------------------------------
            # (C) Callbacks: EarlyStopping + checkpoints + per-epoch monitoring (bounded)
            # ----------------------------------------------------
            cb_list = []

            es = callbacks.EarlyStopping(
                monitor="val_p_anomaly_loss",
                mode="min",
                patience=int(cfg.DEEP_PATIENCE),
                restore_best_weights=True,
                verbose=0
            )
            cb_list.append(es)

            # Checkpoint(s)
            if bool(getattr(cfg, "SAVE_CHECKPOINTS", True)):
                ensure_dir(mod_dir)
                ckpt_path = os.path.join(mod_dir, "deep_best.keras")
                mc = callbacks.ModelCheckpoint(
                    ckpt_path,
                    monitor="val_p_anomaly_loss",
                    mode="min",
                    save_best_only=True,
                    save_weights_only=False,
                    verbose=0
                )
                cb_list.append(mc)

            # Per-epoch monitoring (metrics + optional TDA)
            epoch_logs = []
            tda_epoch_rows = []

            class _EpochMonitor(callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    logs = logs or {}
                    if not bool(getattr(cfg, "DO_PER_EPOCH_MONITORING", True)):
                        return
                    if (epoch + 1) % int(max(1, getattr(cfg, "MONITOR_EVERY_EPOCHS", 1))) != 0:
                        return

                    # Bounded subsample for monitoring
                    nmax = int(getattr(cfg, "MONITOR_MAX_SAMPLES", 4096))
                    rng = np.random.default_rng(cfg.SEED + 700 + epoch)
                    ntr = Xtr_deep.shape[0]
                    idx = rng.choice(ntr, size=min(nmax, ntr), replace=False)
                    Xs = Xtr_deep[idx]
                    ys = ytr_deep[idx]

                    out_s = self.model.predict(Xs, batch_size=int(cfg.DEEP_BATCH), verbose=0)
                    p_s = out_s["p_anomaly"].ravel()

                    # Metrics (robust to single-class)
                    auc_roc = np.nan
                    auc_pr = np.nan
                    if len(np.unique(ys)) > 1:
                        try:
                            auc_roc = float(roc_auc_score(ys, p_s))
                            auc_pr = float(average_precision_score(ys, p_s))
                        except Exception:
                            pass

                    row = {"epoch": int(epoch), "auc_roc_sub": auc_roc, "auc_pr_sub": auc_pr}
                    for k, v in logs.items():
                        if isinstance(v, (float, int, np.floating, np.integer)):
                            row[k] = float(v)
                    epoch_logs.append(row)

                    # Optional TDA every few epochs (bounded)
                    if bool(getattr(cfg, "DO_TDA", True)) and bool(getattr(cfg, "DO_PER_EPOCH_MONITORING", True)):
                        if (epoch + 1) % int(max(1, getattr(cfg, "TDA_EVERY_EPOCHS", 2))) == 0:
                            if TDA_AVAILABLE and encoder is not None:
                                Zs = encoder.predict(Xs, batch_size=int(cfg.DEEP_BATCH), verbose=0)
                                # Subsample for TDA
                                Zt = subsample_rows(Zs, int(min(cfg.TDA_SUBSAMPLE, Zs.shape[0])), seed=cfg.SEED + epoch)
                                tda_res = compute_persistence(Zt, cfg)
                                if tda_res.get("tda_available", False):
                                    sums = tda_res.get("summaries", [])
                                    # record H0/H1 total persistence + entropy
                                    for s in sums:
                                        tda_epoch_rows.append({
                                            "epoch": int(epoch),
                                            "dim": int(s.get("dim", 0)),
                                            "n_features": int(s.get("n_features", 0)),
                                            "total_persistence": float(s.get("total_persistence", 0.0)),
                                            "mean_lifetime": float(s.get("mean_lifetime", 0.0)),
                                            "persistence_entropy": float(s.get("persistence_entropy", 0.0))
                                        })
                            else:
                                # placeholder row
                                tda_epoch_rows.append({"epoch": int(epoch), "dim": -1, "n_features": 0, "total_persistence": 0.0,
                                                       "mean_lifetime": 0.0, "persistence_entropy": 0.0, "NOTE": "per_epoch_tda_unavailable"})

            cb_list.append(_EpochMonitor())

            # ----------------------------------------------------
            # (D) Fit
            # ----------------------------------------------------
            # Force NumPy arrays (prevents mixed Tensor/NumPy issues in metrics/export)
            Xtr_deep = np.asarray(Xtr_deep, dtype=np.float32)
            ytr_deep = np.asarray(ytr_deep, dtype=np.float32)
            hist = deep_model.fit(
                Xtr_deep,
                {"p_anomaly": ytr_deep, "x_recon": Xtr_deep},
                validation_split=float(cfg.VAL_FROM_TRAIN),
                epochs=int(cfg.DEEP_EPOCHS),
                batch_size=int(cfg.DEEP_BATCH),
                verbose=0,
                callbacks=cb_list
            )

            deep_history_df = pd.DataFrame(hist.history)
            deep_history_df.insert(0, "epoch", np.arange(len(deep_history_df)))

            deep_epoch_monitor_df = pd.DataFrame(epoch_logs) if epoch_logs else empty_table(["epoch", "auc_roc_sub", "auc_pr_sub", "NOTE"], note="No epoch monitor logs were recorded.")
            tda_over_epochs_df = pd.DataFrame(tda_epoch_rows) if tda_epoch_rows else empty_table(["epoch", "dim", "n_features", "total_persistence", "mean_lifetime", "persistence_entropy", "NOTE"], note="No per-epoch TDA logs were recorded.")

            # ----------------------------------------------------
            # (E) Test predictions
            # ----------------------------------------------------
            out = deep_model.predict(Xte_deep, batch_size=int(cfg.DEEP_BATCH), verbose=0)
            y_prob = out["p_anomaly"].ravel()
            y_pred = (y_prob >= 0.5).astype(int)
            probas["DEEP"] = y_prob
            preds["DEEP"] = y_pred

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rep = classification_report(np.asarray(yte_deep).astype(int), y_pred, output_dict=True, zero_division=0)
            reports["DEEP"] = pd.DataFrame(rep).T.reset_index().rename(columns={"index": "class"})

            # ----------------------------------------------------
            # (F) Latent + recon error on all windows (for plots/TDA)
            # ----------------------------------------------------
            latent_all = encoder.predict(Xall_deep, batch_size=int(cfg.DEEP_BATCH), verbose=0)
            out_all = deep_model.predict(Xall_deep, batch_size=int(cfg.DEEP_BATCH), verbose=0)
            X_recon_all = out_all["x_recon"]

            if deep_mode == "RAW_CONV1D":
                recon_mse_all = np.mean((Xall_deep - X_recon_all) ** 2, axis=(1, 2))
            else:
                recon_mse_all = np.mean((Xall_deep - X_recon_all) ** 2, axis=1)

            # Recon examples plot (bounded)
            try:
                n_ex = int(getattr(cfg, "N_RECON_EXAMPLES", 24))
                rng = np.random.default_rng(cfg.SEED + 999)
                n_all = Xte_deep.shape[0]
                idx_ex = rng.choice(n_all, size=min(n_ex, n_all), replace=False)
                Xex = Xte_deep[idx_ex]
                out_ex = deep_model.predict(Xex, batch_size=int(cfg.DEEP_BATCH), verbose=0)
                Rex = out_ex["x_recon"]
                if deep_mode == "RAW_CONV1D":
                    mse_ex = np.mean((Xex - Rex) ** 2, axis=(1, 2))
                    # save a few panels
                    fig_path = os.path.join(fig_dir, "deep_recon_examples_raw.png")
                    plt.figure(figsize=(10, 7), dpi=cfg.FIG_DPI)
                    kmax = min(6, Xex.shape[0])
                    for k in range(kmax):
                        plt.subplot(kmax, 1, k + 1)
                        plt.plot(Xex[k, :, 0], label="x_true_f1")
                        plt.plot(Rex[k, :, 0], label="x_recon_f1", linestyle="--")
                        plt.title(f"Recon example {k} (mse={mse_ex[k]:.4e})")
                        plt.legend(loc="upper right", fontsize=7)
                    plt.tight_layout()
                    plt.savefig(fig_path)
                    plt.close()
                    manifest["figures"].append({"path": fig_path, "type": "plot", "status": "real"})
                else:
                    mse_ex = np.mean((Xex - Rex) ** 2, axis=1)
                    fig_path = os.path.join(fig_dir, "deep_recon_examples_stats.png")
                    plt.figure(figsize=(8, 5), dpi=cfg.FIG_DPI)
                    plt.hist(mse_ex, bins=20)
                    plt.title("Deep Recon MSE (example subset)")
                    plt.xlabel("MSE")
                    plt.ylabel("Count")
                    plt.tight_layout()
                    plt.savefig(fig_path)
                    plt.close()
                    manifest["figures"].append({"path": fig_path, "type": "plot", "status": "real"})

                deep_recon_examples_df = pd.DataFrame({"example_index": idx_ex.astype(int), "recon_mse": mse_ex.astype(float)})
            except Exception as e:
                msg = f"Recon examples failed: {e}"
                deep_recon_examples_df = empty_table(["example_index", "recon_mse", "NOTE"], note=msg)
                placeholder_plot(os.path.join(fig_dir, "deep_recon_examples.png"), "Deep recon examples (placeholder)", msg, dpi=cfg.FIG_DPI)
                manifest["figures"].append({"path": os.path.join(fig_dir, "deep_recon_examples.png"), "type": "plot", "status": "placeholder", "reason": msg})

        except Exception as e:
            # If TF is unavailable OR deep training fails, run a sklearn fallback so deep_* sheets are REAL.
            try:
                fb = train_deep_fallback_stats(
        Xtr=Xtr_deep,
        ytr=ytr_deep,
        Xva=None,
        yva=None,
        Xte=Xte_deep,
        yte=yte_deep,
        Xall=Xall_deep,
        cfg=cfg
    )
                manifest["notes"].append(f"Deep model fallback used due to: {e}")
                deep_history_df = fb["deep_training_history"]
                deep_epoch_monitor_df = fb["deep_epoch_monitoring"]
                deep_recon_error_df = fb["deep_recon_error"]
                deep_recon_examples_df = fb["deep_recon_examples"]
                deep_latent_df = fb["deep_latent"]
                tda_over_epochs_df = fb.get("tda_over_epochs_latent", empty_table(["epoch", "dim", "n_features", "total_persistence", "mean_lifetime", "persistence_entropy", "NOTE"], note="TDA over epochs not computed in fallback."))
                deep_model = None
                encoder = None
                latent_all = fb["latent_all"]
                recon_mse_all = fb.get("recon_mse_all", None)
                deep_test_prob = fb["p_test"]
                deep_all_prob = fb["p_all"]
                # Populate DEEP entries for metrics/plots/export just like the TF path
                try:
                    probas["DEEP"] = np.asarray(deep_test_prob, dtype=float)
                    preds["DEEP"] = (probas["DEEP"] >= 0.5).astype(int)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        rep = classification_report(np.asarray(yte_deep).astype(int), preds["DEEP"], output_dict=True, zero_division=0)
                    reports["DEEP"] = pd.DataFrame(rep).T.reset_index().rename(columns={"index": "class"})
                except Exception as _e_rep:
                    manifest["notes"].append(f"Deep fallback report build failed: {_e_rep}")
                fallback_models = {'scaler': fb['scaler'], 'recon': fb['recon_model'], 'head': fb['head_model']}
            except Exception as e2:
                manifest["notes"].append(f"Deep model training failed: {e}; fallback also failed: {e2}")
                deep_history_df = empty_table(["epoch", "loss", "p_anomaly_loss", "x_recon_loss", "val_loss", "val_p_anomaly_loss", "val_x_recon_loss", "NOTE"], note=f"Deep model training failed: {e}; fallback failed: {e2}")
                deep_epoch_monitor_df = empty_table(["epoch", "train_recon_mse", "val_recon_mse", "train_auc_roc_sub", "train_auc_pr_sub", "val_auc_roc_sub", "val_auc_pr_sub", "NOTE"], note=f"Deep model training failed: {e}; fallback failed: {e2}")
                deep_recon_error_df = empty_table(["idx", "recon_mse", "p_anomaly", "NOTE"], note=f"Deep model training failed: {e}; fallback failed: {e2}")
                deep_recon_examples_df = empty_table(["example_idx", "recon_mse", "p_anomaly", "NOTE"], note=f"Deep model training failed: {e}; fallback failed: {e2}")
                deep_latent_df = empty_table(["idx", "z0", "z1", "NOTE"], note=f"Deep model training failed: {e}; fallback failed: {e2}")
                tda_over_epochs_df = empty_table(["epoch", "dim", "n_features", "total_persistence", "mean_lifetime", "persistence_entropy", "NOTE"], note=f"Deep model training failed: {e}; fallback failed: {e2}")
                deep_model = None
                encoder = None
                latent_all = None
                deep_test_prob = None
                deep_all_prob = None
                fallback_models = None
    else:
        note = "Deep model skipped because TF is unavailable or DO_DEEP=False."
        manifest["notes"].append(note)
        deep_history_df = empty_table(["epoch", "loss", "p_anomaly_loss", "x_recon_loss", "val_loss", "val_p_anomaly_loss", "val_x_recon_loss", "NOTE"], note=note)
        deep_epoch_monitor_df = empty_table(["epoch", "auc_roc_sub", "auc_pr_sub", "NOTE"], note=note)
        deep_recon_examples_df = empty_table(["example_index", "recon_mse", "NOTE"], note=note)
        tda_over_epochs_df = empty_table(["epoch", "dim", "n_features", "total_persistence", "mean_lifetime", "persistence_entropy", "NOTE"], note=note)

    # -----------------------------
    # 6) Diagnostics plots + metrics (always produce full set per model)
    # -----------------------------
    model_rows = []
    if len(probas) == 0:
        # Create a placeholder model_metrics entry so Excel is never empty
        model_rows.append({"model": "NONE", "auc_roc": np.nan, "auc_pr": np.nan, "brier": np.nan, "tn": 0, "fp": 0, "fn": 0, "tp": 0, "NOTE": "No model produced probabilities."})
        # Placeholder plots expected by downstream
        placeholder_plot(os.path.join(fig_dir, "model_comparison_auc_roc.png"),
                         "Model Comparison: ROC AUC (placeholder)", "No models available to compare.", dpi=cfg.FIG_DPI)
        placeholder_plot(os.path.join(fig_dir, "model_comparison_auc_pr.png"),
                         "Model Comparison: PR AUC (placeholder)", "No models available to compare.", dpi=cfg.FIG_DPI)
        placeholder_plot(os.path.join(fig_dir, "model_comparison_brier.png"),
                         "Model Comparison: Brier (placeholder)", "No models available to compare.", dpi=cfg.FIG_DPI)
        manifest["figures"].append({"path": os.path.join(fig_dir, "model_comparison_auc_roc.png"), "type": "plot", "status": "placeholder", "reason": "No models"})
        manifest["figures"].append({"path": os.path.join(fig_dir, "model_comparison_auc_pr.png"), "type": "plot", "status": "placeholder", "reason": "No models"})
        manifest["figures"].append({"path": os.path.join(fig_dir, "model_comparison_brier.png"), "type": "plot", "status": "placeholder", "reason": "No models"})
    else:
        for name, y_prob in probas.items():
            y_pred = preds[name]
            m1 = plot_roc_pr_always(yte, y_prob, fig_dir, cfg.FIG_DPI, prefix=name, manifest=manifest)
            m2 = plot_calibration_always(yte, y_prob, fig_dir, cfg.FIG_DPI, prefix=name, manifest=manifest)
            cmc = plot_confusion_always(yte, y_pred, fig_dir, cfg.FIG_DPI, prefix=name, manifest=manifest)
            plot_score_distributions_always(yte, y_prob, fig_dir, cfg.FIG_DPI, prefix=name, manifest=manifest)

            row = {
                "model": name,
                "auc_roc": m1.get("auc_roc", np.nan),
                "auc_pr": m1.get("auc_pr", np.nan),
                "brier": m2.get("brier", np.nan),
                "tn": cmc.get("tn", 0),
                "fp": cmc.get("fp", 0),
                "fn": cmc.get("fn", 0),
                "tp": cmc.get("tp", 0),
                "NOTE": ""
            }
            model_rows.append(row)

        model_metrics_df = pd.DataFrame(model_rows).sort_values(["auc_pr", "auc_roc"], ascending=False).reset_index(drop=True)

        # Model comparison plots (always)
        try:
            plt.figure(figsize=(8, 4), dpi=cfg.FIG_DPI)
            plt.bar(model_metrics_df["model"], model_metrics_df["auc_roc"])
            plt.title("Model Comparison: ROC AUC")
            plt.tight_layout()
            p = os.path.join(fig_dir, "model_comparison_auc_roc.png")
            plt.savefig(p)
            plt.close()
            manifest["figures"].append({"path": p, "type": "plot", "status": "real"})
        except Exception as e:
            p = os.path.join(fig_dir, "model_comparison_auc_roc.png")
            placeholder_plot(p, "Model Comparison: ROC AUC (placeholder)", f"Failed: {e}", dpi=cfg.FIG_DPI)
            manifest["figures"].append({"path": p, "type": "plot", "status": "placeholder", "reason": str(e)})

        try:
            plt.figure(figsize=(8, 4), dpi=cfg.FIG_DPI)
            plt.bar(model_metrics_df["model"], model_metrics_df["auc_pr"])
            plt.title("Model Comparison: PR AUC (Average Precision)")
            plt.tight_layout()
            p = os.path.join(fig_dir, "model_comparison_auc_pr.png")
            plt.savefig(p)
            plt.close()
            manifest["figures"].append({"path": p, "type": "plot", "status": "real"})
        except Exception as e:
            p = os.path.join(fig_dir, "model_comparison_auc_pr.png")
            placeholder_plot(p, "Model Comparison: PR AUC (placeholder)", f"Failed: {e}", dpi=cfg.FIG_DPI)
            manifest["figures"].append({"path": p, "type": "plot", "status": "placeholder", "reason": str(e)})

        try:
            plt.figure(figsize=(8, 4), dpi=cfg.FIG_DPI)
            plt.bar(model_metrics_df["model"], model_metrics_df["brier"])
            plt.title("Model Comparison: Brier (lower is better)")
            plt.tight_layout()
            p = os.path.join(fig_dir, "model_comparison_brier.png")
            plt.savefig(p)
            plt.close()
            manifest["figures"].append({"path": p, "type": "plot", "status": "real"})
        except Exception as e:
            p = os.path.join(fig_dir, "model_comparison_brier.png")
            placeholder_plot(p, "Model Comparison: Brier (placeholder)", f"Failed: {e}", dpi=cfg.FIG_DPI)
            manifest["figures"].append({"path": p, "type": "plot", "status": "placeholder", "reason": str(e)})

    # Ensure model_metrics_df exists
    if "model_metrics_df" not in locals():
        model_metrics_df = pd.DataFrame(model_rows)

    # Primary model choice
    primary_name = "DEEP" if "DEEP" in probas else (model_metrics_df.iloc[0]["model"] if len(model_metrics_df) else "NONE")

    # Threshold sweep (always create table + plot placeholders if needed)
    sweep_df = empty_table(["threshold", "precision", "recall", "f1", "tp", "fp", "tn", "fn", "NOTE"], note="Threshold sweep not computed.")
    best_th = 0.5
    sweep_plot_path = os.path.join(fig_dir, f"{primary_name.lower()}_threshold_sweep.png")

    if cfg.DO_THRESHOLD_SWEEP and primary_name in probas:
        try:
            sweep_df, best_th = threshold_sweep(yte, probas[primary_name], cfg.THRESH_GRID_N)
            plot_threshold_sweep_always(sweep_df, fig_dir, cfg.FIG_DPI, primary_name, manifest)
            # Best-F1 confusion plot
            best_pred = (probas[primary_name] >= best_th).astype(int)
            plot_confusion_always(yte, best_pred, fig_dir, cfg.FIG_DPI, prefix=f"{primary_name}_bestF1", manifest=manifest)
        except Exception as e:
            note = f"Threshold sweep failed: {e}"
            sweep_df = empty_table(["threshold", "precision", "recall", "f1", "tp", "fp", "tn", "fn", "NOTE"], note=note)
            placeholder_plot(sweep_plot_path, f"{primary_name} Threshold Sweep (placeholder)", note, dpi=cfg.FIG_DPI)
            manifest["figures"].append({"path": sweep_plot_path, "type": "plot", "status": "placeholder", "reason": note})
    else:
        note = "Threshold sweep disabled or primary model probability missing."
        placeholder_plot(sweep_plot_path, f"{primary_name} Threshold Sweep (placeholder)", note, dpi=cfg.FIG_DPI)
        manifest["figures"].append({"path": sweep_plot_path, "type": "plot", "status": "placeholder", "reason": note})

    # Embeddings: prefer deep latent else PCA proxy; always create PCA/t-SNE plots
    if latent_all is not None:
        plot_embedding_2d_always(latent_all, yw, fig_dir, cfg.FIG_DPI, "DEEP_LATENT_ALL", cfg.TSNE_MAX_N, manifest)
    else:
        plot_embedding_2d_always(Zall_proxy, yw, fig_dir, cfg.FIG_DPI, "PCA_PROXY_ALL", cfg.TSNE_MAX_N, manifest)

    # -----------------------------
    # 7) RF permutation importance (always create table + plot placeholder if needed)
    # -----------------------------
    rf_importance_df = empty_table(["feature", "importance_mean", "importance_std", "NOTE"], note="RF permutation importance not computed.")
    imp_plot_path = os.path.join(fig_dir, "rf_perm_importance_top.png")

    if cfg.DO_RF_PERM_IMPORTANCE and rf_model is not None:
        try:
            imp = permutation_importance(
                rf_model, Xte, yte,
                n_repeats=int(cfg.RF_PERM_REPEATS),
                random_state=cfg.SEED,
                n_jobs=-1
            )
            rf_importance_df = pd.DataFrame({
                "feature": feat_names,
                "importance_mean": imp.importances_mean.astype(float),
                "importance_std": imp.importances_std.astype(float),
                "NOTE": ""
            }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

            topn = min(25, len(rf_importance_df))
            plt.figure(figsize=(9, 5), dpi=cfg.FIG_DPI)
            plt.barh(rf_importance_df["feature"].iloc[:topn][::-1], rf_importance_df["importance_mean"].iloc[:topn][::-1])
            plt.title("RF Permutation Importance (Top)")
            plt.tight_layout()
            plt.savefig(imp_plot_path)
            plt.close()
            manifest["figures"].append({"path": imp_plot_path, "type": "plot", "status": "real"})
        except Exception as e:
            note = f"RF permutation importance failed: {e}"
            rf_importance_df = empty_table(["feature", "importance_mean", "importance_std", "NOTE"], note=note)
            placeholder_plot(imp_plot_path, "RF Permutation Importance (placeholder)", note, dpi=cfg.FIG_DPI)
            manifest["figures"].append({"path": imp_plot_path, "type": "plot", "status": "placeholder", "reason": note})
    else:
        note = "RF permutation importance skipped (RF missing or disabled)."
        placeholder_plot(imp_plot_path, "RF Permutation Importance (placeholder)", note, dpi=cfg.FIG_DPI)
        manifest["figures"].append({"path": imp_plot_path, "type": "plot", "status": "placeholder", "reason": note})

    # -----------------------------
    # 8) TDA diagnostics (always tables + plots)
    # -----------------------------
    # Determine latent for TDA: prefer deep latent else PCA proxy
    Z_for_tda = latent_all if latent_all is not None else Zall_proxy

    tda_sum_df = empty_table(
        ["subset", "dim", "n_features", "total_persistence", "mean_lifetime", "persistence_entropy",
         "H1_wasserstein_normal_vs_anom", "H1_bottleneck_normal_vs_anom", "NOTE"],
        note="TDA phase report not computed."
    )
    tda_betti_df = empty_table(
        ["grid", "normal_betti_H0", "normal_betti_H1", "anom_betti_H0", "anom_betti_H1", "NOTE"],
        note="TDA Betti curves not computed."
    )

    if cfg.DO_TDA_NORMAL_VS_ANOM:
        tda_sum_df, tda_betti_df = tda_phase_report_always(
            Z=Z_for_tda,
            y=yw,
            cfg=cfg,
            fig_dir=fig_dir,
            prefix="LATENT_ALL_NORMAL_VS_ANOM",
            seed_offset=0,
            manifest=manifest
        )
    else:
        # Create placeholder expected files anyway
        note = "DO_TDA_NORMAL_VS_ANOM=False"
        plot_persistence_diagrams_always([], os.path.join(fig_dir, "LATENT_ALL_NORMAL_VS_ANOM_tda_diagrams_normal.png"),
                                         "LATENT_ALL_NORMAL_VS_ANOM TDA Diagrams (Normal)", cfg.FIG_DPI, manifest)
        plot_persistence_diagrams_always([], os.path.join(fig_dir, "LATENT_ALL_NORMAL_VS_ANOM_tda_diagrams_anomalous.png"),
                                         "LATENT_ALL_NORMAL_VS_ANOM TDA Diagrams (Anomalous)", cfg.FIG_DPI, manifest)
        placeholder_plot(os.path.join(fig_dir, "LATENT_ALL_NORMAL_VS_ANOM_betti_normal.png"),
                         "LATENT_ALL_NORMAL_VS_ANOM Betti (Normal) (placeholder)", note, dpi=cfg.FIG_DPI)
        placeholder_plot(os.path.join(fig_dir, "LATENT_ALL_NORMAL_VS_ANOM_betti_anomalous.png"),
                         "LATENT_ALL_NORMAL_VS_ANOM Betti (Anomalous) (placeholder)", note, dpi=cfg.FIG_DPI)
        manifest["figures"].append({"path": os.path.join(fig_dir, "LATENT_ALL_NORMAL_VS_ANOM_betti_normal.png"), "type": "plot", "status": "placeholder", "reason": note})
        manifest["figures"].append({"path": os.path.join(fig_dir, "LATENT_ALL_NORMAL_VS_ANOM_betti_anomalous.png"), "type": "plot", "status": "placeholder", "reason": note})

    # Selfgen TDA phase (topK) as proxy; always provide table + placeholder if not possible
    tda_selfgen_sum_df = empty_table(
        ["subset", "dim", "n_features", "total_persistence", "mean_lifetime", "persistence_entropy",
         "H1_wasserstein_normal_vs_anom", "H1_bottleneck_normal_vs_anom", "NOTE"],
        note="TDA selfgen report not computed."
    )
    tda_selfgen_betti_df = empty_table(
        ["grid", "normal_betti_H0", "normal_betti_H1", "anom_betti_H0", "anom_betti_H1", "NOTE"],
        note="TDA selfgen Betti not computed."
    )

    if cfg.DO_TDA_SELFGEN and selected_for_curriculum is not None and selected_for_curriculum.shape[0] > 0:
        # Partition selected by baseline prob to create 2 groups
        try:
            if primary_pre == "RF" and rf_model is not None:
                p_sel = rf_model.predict_proba(selected_for_curriculum)[:, 1]
            elif primary_pre == "LOGREG" and lr_model is not None:
                p_sel = lr_model.predict_proba(selected_for_curriculum)[:, 1]
            else:
                p_sel = np.full(selected_for_curriculum.shape[0], 0.5, dtype=float)
            y_sel = (p_sel >= 0.5).astype(int)

            # embed selected in same latent space used by TDA
            if encoder is not None and TF_AVAILABLE:
                Z_sel = encoder.predict(selected_for_curriculum, batch_size=int(cfg.DEEP_BATCH), verbose=0)
            else:
                Z_sel = pca_proxy.transform(selected_for_curriculum)

            tda_selfgen_sum_df, tda_selfgen_betti_df = tda_phase_report_always(
                Z=Z_sel,
                y=y_sel,
                cfg=cfg,
                fig_dir=fig_dir,
                prefix="SELFGEN_SELECTED_PROXY",
                seed_offset=1,
                manifest=manifest
            )
        except Exception as e:
            note = f"Selfgen TDA failed: {e}"
            tda_selfgen_sum_df = empty_table(tda_selfgen_sum_df.columns.tolist(), note=note)
            tda_selfgen_betti_df = empty_table(tda_selfgen_betti_df.columns.tolist(), note=note)
            # placeholder files already handled by tda_phase_report_always if called; ensure them if not
            placeholder_plot(os.path.join(fig_dir, "SELFGEN_SELECTED_PROXY_tda_diagrams_normal.png"),
                             "SELFGEN_SELECTED_PROXY TDA Diagrams (Normal) (placeholder)", note, dpi=cfg.FIG_DPI)
            placeholder_plot(os.path.join(fig_dir, "SELFGEN_SELECTED_PROXY_tda_diagrams_anomalous.png"),
                             "SELFGEN_SELECTED_PROXY TDA Diagrams (Anomalous) (placeholder)", note, dpi=cfg.FIG_DPI)
            placeholder_plot(os.path.join(fig_dir, "SELFGEN_SELECTED_PROXY_betti_normal.png"),
                             "SELFGEN_SELECTED_PROXY Betti (Normal) (placeholder)", note, dpi=cfg.FIG_DPI)
            placeholder_plot(os.path.join(fig_dir, "SELFGEN_SELECTED_PROXY_betti_anomalous.png"),
                             "SELFGEN_SELECTED_PROXY Betti (Anomalous) (placeholder)", note, dpi=cfg.FIG_DPI)
            manifest["figures"].append({"path": os.path.join(fig_dir, "SELFGEN_SELECTED_PROXY_tda_diagrams_normal.png"), "type": "plot", "status": "placeholder", "reason": note})
            manifest["figures"].append({"path": os.path.join(fig_dir, "SELFGEN_SELECTED_PROXY_tda_diagrams_anomalous.png"), "type": "plot", "status": "placeholder", "reason": note})
            manifest["figures"].append({"path": os.path.join(fig_dir, "SELFGEN_SELECTED_PROXY_betti_normal.png"), "type": "plot", "status": "placeholder", "reason": note})
            manifest["figures"].append({"path": os.path.join(fig_dir, "SELFGEN_SELECTED_PROXY_betti_anomalous.png"), "type": "plot", "status": "placeholder", "reason": note})
    else:
        note = "Selfgen TDA skipped (no selected candidates)."
        placeholder_plot(os.path.join(fig_dir, "SELFGEN_SELECTED_PROXY_tda_diagrams_normal.png"),
                         "SELFGEN_SELECTED_PROXY TDA Diagrams (Normal) (placeholder)", note, dpi=cfg.FIG_DPI)
        placeholder_plot(os.path.join(fig_dir, "SELFGEN_SELECTED_PROXY_tda_diagrams_anomalous.png"),
                         "SELFGEN_SELECTED_PROXY TDA Diagrams (Anomalous) (placeholder)", note, dpi=cfg.FIG_DPI)
        placeholder_plot(os.path.join(fig_dir, "SELFGEN_SELECTED_PROXY_betti_normal.png"),
                         "SELFGEN_SELECTED_PROXY Betti (Normal) (placeholder)", note, dpi=cfg.FIG_DPI)
        placeholder_plot(os.path.join(fig_dir, "SELFGEN_SELECTED_PROXY_betti_anomalous.png"),
                         "SELFGEN_SELECTED_PROXY Betti (Anomalous) (placeholder)", note, dpi=cfg.FIG_DPI)
        manifest["figures"].append({"path": os.path.join(fig_dir, "SELFGEN_SELECTED_PROXY_tda_diagrams_normal.png"), "type": "plot", "status": "placeholder", "reason": note})
        manifest["figures"].append({"path": os.path.join(fig_dir, "SELFGEN_SELECTED_PROXY_tda_diagrams_anomalous.png"), "type": "plot", "status": "placeholder", "reason": note})
        manifest["figures"].append({"path": os.path.join(fig_dir, "SELFGEN_SELECTED_PROXY_betti_normal.png"), "type": "plot", "status": "placeholder", "reason": note})
        manifest["figures"].append({"path": os.path.join(fig_dir, "SELFGEN_SELECTED_PROXY_betti_anomalous.png"), "type": "plot", "status": "placeholder", "reason": note})

    # Time evolution TDA (always)
    tda_evo_df = tda_time_evolution_always(Z_for_tda, win_starts, cfg, fig_dir, "latent_all", manifest)

    # -----------------------------
    # 9) Build required tables (ensure ALL exist)
    # -----------------------------
    tables: Dict[str, pd.DataFrame] = {}

    # Model metrics
    tables["model_metrics"] = model_metrics_df

    # Reports per model (ensure placeholders for missing ones)
    for m in ["RF", "LOGREG", "DEEP"]:
        if m in reports:
            tables[f"report_{m.lower()}"] = reports[m]
        else:
            tables[f"report_{m.lower()}"] = empty_table(["class", "precision", "recall", "f1-score", "support", "NOTE"],
                                                        note=f"Classification report not available for {m}.")

    # Test predictions table for primary (always)
    if primary_name in probas:
        primary_prob = probas[primary_name]
        primary_pred = (primary_prob >= 0.5).astype(int)
        test_pred_df = pd.DataFrame({
            "test_row": np.arange(len(yte)),
            "window_start": wte,
            "y_true": yte,
            "y_pred": primary_pred,
            "y_prob": primary_prob.astype(float),
        }).sort_values("window_start").reset_index(drop=True)
        tables["test_predictions_primary"] = test_pred_df
    else:
        tables["test_predictions_primary"] = empty_table(
            ["test_row", "window_start", "y_true", "y_pred", "y_prob", "NOTE"],
            note="Primary model has no probabilities; cannot build predictions table."
        )

    # Windows index
    tables["windows_index"] = windows_index_df

    # Threshold sweep and best threshold (always)
    tables[f"threshold_sweep_{primary_name.lower()}"] = sweep_df
    tables["best_threshold"] = pd.DataFrame([{
        "primary_model": primary_name,
        "best_f1_threshold": float(best_th),
        "NOTE": "" if primary_name in probas else "Primary model probs missing; best threshold is default 0.5."
    }])

    # Deep latent + recon error (always placeholders if deep missing)
    if latent_all is not None and recon_mse_all is not None:
        deep_lat_df = pd.DataFrame(latent_all, columns=[f"z{i+1}" for i in range(latent_all.shape[1])])
        deep_lat_df.insert(0, "window_index", np.arange(len(yw)))
        deep_lat_df.insert(1, "window_start", win_starts)
        deep_lat_df.insert(2, "y_window", yw)
        deep_lat_df["NOTE"] = ""
        tables["deep_latent"] = deep_lat_df

        recon_df = pd.DataFrame({
            "window_index": np.arange(len(yw)),
            "window_start": win_starts,
            "y_window": yw,
            "recon_mse": recon_mse_all.astype(float),
            "NOTE": ""
        })
        tables["deep_recon_error"] = recon_df
    else:
        tables["deep_latent"] = empty_table(["window_index", "window_start", "y_window", "z1", "NOTE"],
                                            note="Deep latent not available (deep model not trained).")
        tables["deep_recon_error"] = empty_table(["window_index", "window_start", "y_window", "recon_mse", "NOTE"],
                                                 note="Reconstruction error not available (deep model not trained).")

    # Deep history (always)
    if deep_history_df is None:
        deep_history_df = empty_table(["epoch", "loss", "p_anomaly_loss", "x_recon_loss", "val_loss", "val_p_anomaly_loss", "val_x_recon_loss", "NOTE"],
                                      note="Deep training history unavailable.")
    tables["deep_training_history"] = deep_history_df

    tables["deep_epoch_monitoring"] = deep_epoch_monitor_df
    tables["deep_recon_examples"] = deep_recon_examples_df
    tables["tda_over_epochs_latent"] = tda_over_epochs_df

    

    # RF importance (always)
    tables["rf_perm_importance"] = rf_importance_df

    # Self-generation audit (always)
    tables["selfgen_verified_scored"] = selfgen_audit_df

    # TDA tables (always)
    tables["tda_latent_normal_vs_anom_summary"] = tda_sum_df
    tables["tda_latent_normal_vs_anom_betti"] = tda_betti_df
    tables["tda_selfgen_selected_summary"] = tda_selfgen_sum_df
    tables["tda_selfgen_selected_betti"] = tda_selfgen_betti_df
    tables["tda_time_evolution_latent_all"] = tda_evo_df

    # Add a "artifact index" table (always)
    # (We will fill after saving, but include placeholder here)
    tables["artifact_index"] = pd.DataFrame({"NOTE": ["Artifact index will be filled from manifest.json after run."]})

    # -----------------------------
    # 10) Save models (always)
    # -----------------------------
    save_models_always(cfg, mod_dir, rf_model, lr_model, scaler, pca_proxy, deep_model, encoder, manifest, fallback_models)

    # -----------------------------
    # 11) Summary JSON + run_info (always)
    # -----------------------------
    runtime_sec = float(time.time() - t0)
    summary = {
        "primary_model": str(primary_name),
        "n_windows_total": int(len(yw)),
        "window_anomaly_rate": float(np.mean(yw)),
        "n_train_windows": int(Xtr.shape[0]),
        "n_test_windows": int(Xte.shape[0]),
        "temporal_split": True,
        "runtime_sec": runtime_sec,
        "TF_AVAILABLE": bool(TF_AVAILABLE),
        "TDA_AVAILABLE": bool(TDA_AVAILABLE),
        "WASSERSTEIN_AVAILABLE": bool(WASSERSTEIN_AVAILABLE),
        "BOTTLENECK_AVAILABLE": bool(BOTTLENECK_AVAILABLE),
        "JOBLIB_AVAILABLE": bool(JOBLIB_AVAILABLE),
        "config": asdict(cfg),
        "notes": manifest.get("notes", [])
    }

    run_info_path = os.path.join(rep_dir, "run_info.txt")
    with open(run_info_path, "w", encoding="utf-8") as f:
        f.write("ECONOMETRIA TOPO FAST-FULL-PLUS (WORD-ALIGNED) — SAVE EVERYTHING EDITION\n")
        f.write(f"Primary model: {primary_name}\n")
        f.write(f"Temporal split: True\n")
        f.write(f"Windows total: {len(yw)}\n")
        f.write(f"Train windows: {Xtr.shape[0]}\n")
        f.write(f"Test windows:  {Xte.shape[0]}\n")
        f.write(f"Anomaly rate (window): {np.mean(yw):.6f}\n")
        f.write(f"Runtime (sec): {runtime_sec:.2f}\n")
        f.write(f"TF_AVAILABLE: {TF_AVAILABLE}\n")
        f.write(f"TDA_AVAILABLE: {TDA_AVAILABLE}\n")
        f.write(f"WASSERSTEIN_AVAILABLE: {WASSERSTEIN_AVAILABLE}\n")
        f.write(f"BOTTLENECK_AVAILABLE: {BOTTLENECK_AVAILABLE}\n")
        f.write(f"JOBLIB_AVAILABLE: {JOBLIB_AVAILABLE}\n")
        if manifest["notes"]:
            f.write("\nNOTES:\n")
            for n in manifest["notes"]:
                f.write(f"- {n}\n")
    manifest["reports"].append({"path": run_info_path, "type": "txt", "status": "real"})

    # -----------------------------
    # 12) Export tables (Excel/CSV/JSON/manifest)
    # -----------------------------
    # A deterministic sheet order so "nothing is missing" and users find everything.
    sheet_order = [
        "model_metrics",
        "report_rf", "report_logreg", "report_deep",
        "test_predictions_primary",
        "best_threshold", f"threshold_sweep_{primary_name.lower()}",
        "rf_perm_importance",
        "selfgen_verified_scored",
        "deep_training_history",
        "deep_epoch_monitoring",
        "deep_recon_examples",
        "tda_over_epochs_latent",
        "deep_latent",
        "deep_recon_error",
        "windows_index",
        "tda_latent_normal_vs_anom_summary",
        "tda_latent_normal_vs_anom_betti",
        "tda_selfgen_selected_summary",
        "tda_selfgen_selected_betti",
        "tda_time_evolution_latent_all",
        "artifact_index"
    ]

    export_tables_always(cfg, rep_dir, tables, sheet_order, summary, manifest)

    # -----------------------------
    # 13) Package reports.zip (so the user can share / archive the exact Excel/CSV artifacts)
    # -----------------------------
    try:
        zip_path = package_reports_zip(cfg.OUTPUT_DIR, zip_name='reports.zip')
        if zip_path:
            manifest['reports'].append({'path': zip_path, 'type': 'zip', 'status': 'real'})
    except Exception as e:
        manifest['notes'].append(f'Packaging reports.zip failed: {e}')

    # After manifest is written, create artifact_index from manifest and overwrite Excel/CSV sheet only.
    # (This keeps everything "complete" and auditable.)
    try:
        artifact_rows = []
        for section in ["figures", "reports", "models"]:
            for item in manifest.get(section, []):
                artifact_rows.append({
                    "section": section,
                    "path": item.get("path", ""),
                    "type": item.get("type", ""),
                    "status": item.get("status", ""),
                    "name": item.get("name", ""),
                    "reason": item.get("reason", "")
                })
        artifact_index_df = pd.DataFrame(artifact_rows)
        artifact_csv = os.path.join(rep_dir, "artifact_index.csv")
        artifact_index_df.to_csv(artifact_csv, index=False)

        # Also store in tables for Excel rewrite
        # Rewrite Excel with updated artifact_index (still bounded; the dataset is small)
        xlsx_path = os.path.join(rep_dir, "report_tables.xlsx")
        if cfg.SAVE_EXCEL:
            with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
                for sheet_name in sheet_order:
                    if sheet_name == "artifact_index":
                        artifact_index_df.to_excel(writer, sheet_name="artifact_index", index=False)
                    else:
                        df = tables.get(sheet_name, pd.DataFrame({"NOTE": [f"Missing {sheet_name} (unexpected)."]}))
                        sanitize_dataframe_for_export(df).to_excel(writer, sheet_name=sheet_name[:31], index=False)
        manifest["reports"].append({"path": artifact_csv, "type": "csv", "status": "real"})
    except Exception as e:
        manifest["notes"].append(f"Artifact index post-write failed: {e}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    cfg = Config()

    # If using a real CSV:
    # cfg.INPUT_CSV_PATH = "path/to/data.csv"
    # cfg.TARGET_COL = "label"
    # cfg.FEATURE_COLS = ["x1", "x2", ...]

    run_pipeline(cfg)