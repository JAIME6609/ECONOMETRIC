
# -*- coding: utf-8 -*-
"""
TOPOLOGICAL MONTE CARLO FOR ASSET DYNAMICS
PART 5 — Results and analysis (5.1, 5.2, 5.3)
=============================================

This script is an end-to-end, reproducible implementation intended to support
the paper's Sections 2–4 and to generate the *minimal* tables and figures needed
to populate Section 5:

  5.1 Single-asset experiments:
      - regime separation
      - turbulence signatures
      - structural stability of simulated dynamics

  5.2 Multi-asset experiments:
      - dependence topology
      - correlation breakdowns
      - diversification limits in high-dimensional clouds

  5.3 Risk and stress-testing experiments:
      - topology-informed early-warning signals for VaR/ES fragility
      - scenario sensitivity

Design alignment with the uploaded article (Sections 2–4)
---------------------------------------------------------
* Section 2: Monte Carlo as a "generative laboratory" with regime-sensitive dynamics,
  and scenario spaces treated geometrically/topologically via point clouds and filtrations.
* Section 3: Simulation-to-geometry pipeline, persistent homology (Vietoris–Rips),
  and evaluation protocol combining stability checks (bootstrap, jitter) with finance metrics.
* Section 4: Reproducible architecture: configuration, run_id, and structured logging.

Output contract (user requirement)
----------------------------------
Creates one root results folder with three subfolders (5.1, 5.2, 5.3). Inside each:
  * tables/  : one Excel workbook with multiple sheets (the minimal set of tables)
  * figures/ : a minimal set of PNG plots supporting the subsection narrative

No "extra" tables/figures are generated beyond those that directly support 5.1–5.3.
A root-level config.json and run.log are produced for reproducibility (not tables/figures).

Dependencies
------------
Core:
  numpy, pandas, matplotlib, scipy, scikit-learn, openpyxl

Topological:
  ripser, persim

Install:
  pip install numpy pandas matplotlib scipy scikit-learn openpyxl ripser persim

Usage
-----
Simulation-only (default):
  python part5_topological_monte_carlo.py

Optional: adjust configuration via CLI flags:
  python part5_topological_monte_carlo.py --out_base results_part5 --seed 7

The code is self-contained and can run without external datasets. If desired, it can
be extended to ingest real asset data; however, Section 5 here is framed explicitly
around simulated dynamics and scenario-space analysis.

"""

import argparse
import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import t as student_t
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_fscore_support, confusion_matrix

# --- Topological dependencies (persistent homology + Wasserstein distance) ---
try:
    from ripser import ripser
    from persim import wasserstein
except Exception as e:
    raise RuntimeError(
        "Topological dependencies are missing. Install with: pip install ripser persim\n"
        f"Original error: {repr(e)}"
    )


# =============================================================================
# Reproducibility & filesystem utilities (Section 4 alignment)
# =============================================================================
def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_excel_with_sheets(path_xlsx: str, sheets: Dict[str, pd.DataFrame]) -> None:
    """
    Save multiple DataFrames in one Excel workbook, reducing file clutter while
    preserving traceability and completeness.
    """
    ensure_dir(os.path.dirname(path_xlsx))
    with pd.ExcelWriter(path_xlsx, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)


def save_figure(path_png: str, fig: plt.Figure) -> None:
    ensure_dir(os.path.dirname(path_png))
    fig.savefig(path_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def setup_logging(log_path: str) -> logging.Logger:
    """
    Structured logging to enable forensic replay and reproducibility audits,
    consistent with Section 4's emphasis on logging and run identifiers.
    """
    ensure_dir(os.path.dirname(log_path))
    logger = logging.getLogger("TopologicalMonteCarlo")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


# =============================================================================
# Configuration (explicit, auditable run contract)
# =============================================================================
@dataclass
class GlobalConfig:
    out_base: str = "results_part5"
    seed: int = 7
    run_id: str = ""  # populated at runtime


@dataclass
class SingleAssetConfig:
    # Scenario count and horizon
    n_paths_per_regime: int = 250
    T: int = 252  # trading days ~ 1 year

    # Regime A (low turbulence)
    mu_A: float = 0.0002
    sigma_A: float = 0.010
    jump_prob_A: float = 0.005
    jump_size_A: float = 0.020

    # Regime B (high turbulence)
    mu_B: float = -0.0001
    sigma_B: float = 0.030
    jump_prob_B: float = 0.020
    jump_size_B: float = 0.050

    # Heavy-tail degrees of freedom
    df_t: int = 6

    # Topology parameters for scenario-space PH
    ph_maxdim: int = 1
    ph_max_points: int = 350  # subsample if too many points for PH

    # Stability checks (Section 3.2 evaluation protocol)
    stability_bootstrap: int = 30
    stability_jitter_std: float = 0.03  # jitter in standardized feature space


@dataclass
class MultiAssetConfig:
    # Scenario count and horizon
    n_paths_per_regime: int = 180
    T: int = 252
    d: int = 12

    # Normal regime dependence
    mu_N: float = 0.00015
    sigma_N: float = 0.012
    rho_N: float = 0.20
    jump_prob_N: float = 0.005
    jump_size_N: float = 0.015

    # Crisis regime dependence (correlation breakdown / compression)
    mu_C: float = -0.00005
    sigma_C: float = 0.020
    rho_C: float = 0.75
    jump_prob_C: float = 0.020
    jump_size_C: float = 0.040

    df_t: int = 7

    # PH on high-dimensional dependence cloud (correlation vectors)
    ph_maxdim: int = 1
    ph_max_points: int = 260

    stability_bootstrap: int = 25
    stability_jitter_std: float = 0.03


@dataclass
class RiskStressConfig:
    # Rolling analysis uses one long multi-asset simulation to align topology signals
    # and rolling VaR/ES, supporting early-warning evaluation.
    T_long: int = 3200
    d: int = 12

    # Markov switching for correlation/volatility regimes (simplified discrete-time)
    p_NN: float = 0.993
    p_CC: float = 0.975
    rho_N: float = 0.20
    rho_C: float = 0.75
    sigma_N: float = 0.012
    sigma_C: float = 0.022
    mu: float = 0.00010
    df_t: int = 7

    # Rolling windows for topology + risk
    corr_window: int = 260
    corr_step: int = 10

    risk_lookback: int = 250
    alpha: float = 0.01  # VaR/ES level
    early_warning_horizon: int = 10

    # Stress scenarios (scenario sensitivity)
    stress_n_sims: int = 20000
    stress_vol_mult: float = 1.6
    stress_corr_shock: float = 0.25
    stress_jump_prob: float = 0.03
    stress_jump_size: float = 0.06


# =============================================================================
# Section 2: Simulation engines (single asset + multi asset)
# =============================================================================
def _scaled_student_t(rng: np.random.Generator, df: int, size: Tuple[int, ...]) -> np.ndarray:
    """
    Student-t innovations scaled to have approximately unit variance:
      Var(t_df) = df/(df-2) for df>2, so scale by sqrt((df-2)/df).
    """
    z = student_t.rvs(df=df, size=size, random_state=rng.integers(0, 2**31 - 1))
    return z * np.sqrt((df - 2) / df)


def simulate_single_asset_paths(cfg: SingleAssetConfig, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate two labeled scenario families (Regime A vs Regime B) for single-asset returns.
    This directly supports "regime separation" and "turbulence signatures" at the scenario level.

    Returns:
      R: returns array (N, T)
      y: true regime labels (N,) with 0=A (low turbulence), 1=B (high turbulence)
    """
    rng = np.random.default_rng(seed)
    N = 2 * cfg.n_paths_per_regime
    T = cfg.T

    y = np.zeros(N, dtype=int)
    y[cfg.n_paths_per_regime:] = 1

    R = np.zeros((N, T), dtype=float)

    # Regime A
    zA = _scaled_student_t(rng, cfg.df_t, size=(cfg.n_paths_per_regime, T))
    jumpsA = (rng.uniform(size=(cfg.n_paths_per_regime, T)) < cfg.jump_prob_A).astype(float)
    jump_sizesA = cfg.jump_size_A * rng.standard_normal(size=(cfg.n_paths_per_regime, T))
    R[:cfg.n_paths_per_regime, :] = cfg.mu_A + cfg.sigma_A * zA + jumpsA * jump_sizesA

    # Regime B
    zB = _scaled_student_t(rng, cfg.df_t, size=(cfg.n_paths_per_regime, T))
    jumpsB = (rng.uniform(size=(cfg.n_paths_per_regime, T)) < cfg.jump_prob_B).astype(float)
    jump_sizesB = cfg.jump_size_B * rng.standard_normal(size=(cfg.n_paths_per_regime, T))
    R[cfg.n_paths_per_regime:, :] = cfg.mu_B + cfg.sigma_B * zB + jumpsB * jump_sizesB

    return R, y


def _corr_matrix(d: int, rho: float) -> np.ndarray:
    C = np.full((d, d), rho, dtype=float)
    np.fill_diagonal(C, 1.0)
    return C


def simulate_multi_asset_paths(cfg: MultiAssetConfig, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate two labeled scenario families (Normal vs Crisis) for d-asset returns,
    supporting dependence topology and correlation breakdown experiments.

    Returns:
      R: returns array (N, T, d)
      y: labels (N,) 0=Normal, 1=Crisis
    """
    rng = np.random.default_rng(seed)
    N = 2 * cfg.n_paths_per_regime
    T = cfg.T
    d = cfg.d

    y = np.zeros(N, dtype=int)
    y[cfg.n_paths_per_regime:] = 1

    R = np.zeros((N, T, d), dtype=float)

    # Common heavy-tailed innovations
    Z = _scaled_student_t(rng, cfg.df_t, size=(N, T, d))

    # Normal regime: correlation rho_N
    Cn = _corr_matrix(d, cfg.rho_N)
    Ln = np.linalg.cholesky(Cn + 1e-10 * np.eye(d))

    # Crisis regime: correlation rho_C
    Cc = _corr_matrix(d, cfg.rho_C)
    Lc = np.linalg.cholesky(Cc + 1e-10 * np.eye(d))

    # Jumps
    jumps = rng.uniform(size=(N, T, d))

    for i in range(N):
        if y[i] == 0:
            base = cfg.mu_N + cfg.sigma_N * (Z[i] @ Ln.T)
            J = (jumps[i] < cfg.jump_prob_N).astype(float) * (cfg.jump_size_N * rng.standard_normal(size=(T, d)))
        else:
            base = cfg.mu_C + cfg.sigma_C * (Z[i] @ Lc.T)
            J = (jumps[i] < cfg.jump_prob_C).astype(float) * (cfg.jump_size_C * rng.standard_normal(size=(T, d)))
        R[i] = base + J

    return R, y


# =============================================================================
# Feature embeddings (Section 3.1)
# =============================================================================
def _max_drawdown_from_returns(r: np.ndarray) -> float:
    """
    Max drawdown computed from cumulative log-return approximation:
      P_t = exp(sum r_t), drawdown = 1 - P_t / max_{s<=t} P_s.
    """
    x = np.asarray(r, dtype=float)
    P = np.exp(np.cumsum(x))
    peak = np.maximum.accumulate(P)
    dd = 1.0 - (P / (peak + 1e-12))
    return float(np.max(dd))


def _var_es_from_returns(r: np.ndarray, alpha: float = 0.01) -> Tuple[float, float]:
    """
    Historical VaR/ES of losses where loss = -return.
    VaR is the (1-alpha) quantile of loss, ES is the mean beyond VaR.
    """
    loss = -np.asarray(r, dtype=float)
    var = float(np.quantile(loss, 1 - alpha))
    tail = loss[loss >= var]
    es = float(np.mean(tail)) if tail.size > 0 else var
    return var, es


def single_asset_path_features(R: np.ndarray, alpha: float = 0.01) -> pd.DataFrame:
    """
    Feature embedding for each single-asset path, creating a scenario-space point cloud
    as described in the article's simulation-to-geometry pipeline.
    """
    rows = []
    for i in range(R.shape[0]):
        r = R[i]
        mu = float(np.mean(r))
        sd = float(np.std(r))
        skew = float(pd.Series(r).skew())
        kurt = float(pd.Series(r).kurtosis())
        mdd = _max_drawdown_from_returns(r)
        var, es = _var_es_from_returns(r, alpha=alpha)
        turb = float(np.mean(np.abs(r) > 2.0 * (sd + 1e-12)))  # turbulence share
        rows.append({
            "path_id": i,
            "mean_ret": mu,
            "std_ret": sd,
            "skew_ret": skew,
            "kurt_ret": kurt,
            "max_drawdown": mdd,
            "VaR_loss": var,
            "ES_loss": es,
            "turbulence_share": turb
        })
    return pd.DataFrame(rows)


def multi_asset_path_features(R: np.ndarray, alpha: float = 0.01) -> pd.DataFrame:
    """
    Feature embedding per multi-asset scenario path, emphasizing dependence descriptors
    plus risk-relevant portfolio quantities.
    """
    N, T, d = R.shape
    w = np.ones(d) / d
    rows = []
    for i in range(N):
        Ri = R[i]
        C = np.corrcoef(Ri, rowvar=False)
        avg_corr = float(C[np.triu_indices_from(C, k=1)].mean())
        eig = np.linalg.eigvalsh(C)
        max_eig = float(np.max(eig))
        participation_ratio = float((np.sum(eig) ** 2) / (np.sum(eig ** 2) + 1e-12))

        port = Ri @ w
        port_vol = float(np.std(port))
        var, es = _var_es_from_returns(port, alpha=alpha)

        rows.append({
            "path_id": i,
            "avg_corr": avg_corr,
            "max_eig": max_eig,
            "participation_ratio": participation_ratio,
            "portfolio_vol": port_vol,
            "portfolio_VaR_loss": var,
            "portfolio_ES_loss": es
        })
    return pd.DataFrame(rows)


def correlation_vector_embedding(R: np.ndarray) -> np.ndarray:
    """
    High-dimensional "cloud" embedding for multi-asset dependence:
    each scenario path -> vectorized upper-triangular entries of its correlation matrix.
    """
    N, _, d = R.shape
    iu = np.triu_indices(d, k=1)
    X = np.zeros((N, len(iu[0])), dtype=float)
    for i in range(N):
        C = np.corrcoef(R[i], rowvar=False)
        X[i] = C[iu]
    return X


# =============================================================================
# Topological inference (Vietoris–Rips PH) + summaries + stability checks (Section 3.2)
# =============================================================================
def standardize_matrix(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-score standardization with numerical safeguards.
    Returns standardized X plus mean/std for auditability.
    """
    X = np.asarray(X, dtype=float)
    mu = np.mean(X, axis=0)
    sd = np.std(X, axis=0) + 1e-12
    Xz = (X - mu) / sd
    return Xz, mu, sd


def persistence_summaries_H1(dgms: List[np.ndarray]) -> Dict[str, float]:
    """
    Minimal, interpretable H1 summaries:
      - n_H1
      - total_persistence_H1
      - mean_lifetime_H1
      - entropy_H1
    """
    if len(dgms) < 2 or dgms[1] is None or dgms[1].size == 0:
        return {"n_H1": 0, "total_persistence_H1": 0.0, "mean_lifetime_H1": 0.0, "entropy_H1": 0.0}

    D = dgms[1]
    b = D[:, 0]
    d = D[:, 1]
    finite = np.isfinite(d)
    life = (d - b)[finite]

    if life.size == 0:
        return {"n_H1": 0, "total_persistence_H1": 0.0, "mean_lifetime_H1": 0.0, "entropy_H1": 0.0}

    total = float(np.sum(life))
    p = life / (np.sum(life) + 1e-12)
    entropy = float(-np.sum(p * np.log(p + 1e-12)))
    return {
        "n_H1": int(life.size),
        "total_persistence_H1": total,
        "mean_lifetime_H1": float(np.mean(life)),
        "entropy_H1": entropy
    }


def compute_ph_on_pointcloud(X: np.ndarray, maxdim: int = 1, max_points: int = 300, seed: int = 0) -> List[np.ndarray]:
    """
    Persistent homology for a point cloud. If the cloud is large, a random subsample
    is used for computational tractability while preserving the qualitative structure.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    if X.shape[0] > max_points:
        idx = rng.choice(X.shape[0], size=max_points, replace=False)
        X = X[idx]
    res = ripser(X, maxdim=maxdim)
    return res["dgms"]


def bootstrap_stability_score(
    X: np.ndarray,
    base_dgm_H1: np.ndarray,
    n_boot: int,
    jitter_std: float,
    maxdim: int,
    max_points: int,
    seed: int
) -> float:
    """
    Stability check consistent with Section 3.2:
      repeat PH under bootstrap subsampling + jitter/noise perturbations,
      then compute median diagram deviation (Wasserstein) to the base diagram.

    Returns:
      median Wasserstein distance of H1 diagrams across resamples.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    dists = []
    for b in range(n_boot):
        idx = rng.choice(X.shape[0], size=X.shape[0], replace=True)
        Xb = X[idx].copy()
        Xb += jitter_std * rng.standard_normal(size=Xb.shape)

        dgms = compute_ph_on_pointcloud(Xb, maxdim=maxdim, max_points=max_points, seed=int(rng.integers(0, 2**31 - 1)))
        dgm1 = dgms[1] if len(dgms) > 1 else np.empty((0, 2))

        dists.append(float(wasserstein(base_dgm_H1, dgm1, matching=False)))

    return float(np.median(dists)) if len(dists) > 0 else float("nan")


def plot_persistence_diagram_H1(dgm: np.ndarray, title: str) -> plt.Figure:
    """
    Minimal persistence diagram visualization for H1:
      points (birth, death) with diagonal reference.
    """
    fig = plt.figure(figsize=(5.4, 5.0))
    ax = fig.add_subplot(111)

    if dgm is None or dgm.size == 0:
        ax.text(0.5, 0.5, "Empty H1 diagram", ha="center", va="center")
        ax.set_title(title)
        ax.set_xlabel("birth")
        ax.set_ylabel("death")
        ax.grid(True, alpha=0.25)
        return fig

    finite = np.isfinite(dgm[:, 1])
    pts = dgm[finite]
    if pts.size == 0:
        ax.text(0.5, 0.5, "No finite H1 points", ha="center", va="center")
        ax.set_title(title)
        ax.set_xlabel("birth")
        ax.set_ylabel("death")
        ax.grid(True, alpha=0.25)
        return fig

    ax.scatter(pts[:, 0], pts[:, 1], s=18)
    mx = float(np.nanmax(pts))
    ax.plot([0, mx], [0, mx], linestyle="--", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("birth")
    ax.set_ylabel("death")
    ax.grid(True, alpha=0.25)
    return fig


# =============================================================================
# Risk tools (VaR/ES rolling + basic backtesting + early warning evaluation)
# =============================================================================
def rolling_var_es(returns: np.ndarray, lookback: int, alpha: float) -> pd.DataFrame:
    """
    Rolling historical VaR/ES on a univariate return series.
    """
    r = np.asarray(returns, dtype=float)
    rows = []
    for t in range(lookback, len(r)):
        hist = r[t - lookback:t]
        var, es = _var_es_from_returns(hist, alpha=alpha)
        loss_t = -r[t]
        breach = int(loss_t >= var)
        rows.append({"t": t, "ret": float(r[t]), "loss": float(loss_t), "VaR": var, "ES": es, "breach": breach})
    return pd.DataFrame(rows)


def kupiec_pof_test(n: int, x: int, alpha: float) -> Tuple[float, float]:
    """
    Kupiec POF test for VaR exceptions (likelihood ratio + p-value).
    """
    from scipy.stats import chi2
    if n <= 0:
        return float("nan"), float("nan")
    eps = 1e-12
    pi = np.clip(x / n, eps, 1 - eps)
    a = np.clip(alpha, eps, 1 - eps)
    LR = -2.0 * ((n - x) * np.log((1 - a) / (1 - pi)) + x * np.log(a / pi))
    p = float(1.0 - chi2.cdf(LR, df=1))
    return float(LR), p


def christoffersen_independence_test(breaches: np.ndarray) -> Tuple[float, float]:
    """
    Christoffersen independence test for VaR exceedances (LR + p-value).
    """
    from scipy.stats import chi2
    b = np.asarray(breaches, dtype=int)
    if len(b) < 5:
        return float("nan"), float("nan")

    n00 = np.sum((b[:-1] == 0) & (b[1:] == 0))
    n01 = np.sum((b[:-1] == 0) & (b[1:] == 1))
    n10 = np.sum((b[:-1] == 1) & (b[1:] == 0))
    n11 = np.sum((b[:-1] == 1) & (b[1:] == 1))

    eps = 1e-12
    p01 = (n01) / (n00 + n01 + eps)
    p11 = (n11) / (n10 + n11 + eps)
    p1 = (n01 + n11) / (n00 + n01 + n10 + n11 + eps)

    def ll_ind(p):
        p = np.clip(p, eps, 1 - eps)
        return (n01 + n11) * np.log(p) + (n00 + n10) * np.log(1 - p)

    def ll_dep(p01_, p11_):
        p01_ = np.clip(p01_, eps, 1 - eps)
        p11_ = np.clip(p11_, eps, 1 - eps)
        return (
            n01 * np.log(p01_) + n00 * np.log(1 - p01_) +
            n11 * np.log(p11_) + n10 * np.log(1 - p11_)
        )

    LR = -2.0 * (ll_ind(p1) - ll_dep(p01, p11))
    p = float(1.0 - chi2.cdf(LR, df=1))
    return float(LR), p


def breach_soon_target(breaches: np.ndarray, horizon: int) -> np.ndarray:
    """
    y_t = 1 if any breach occurs within the next 'horizon' steps.
    """
    b = np.asarray(breaches, dtype=int)
    y = np.zeros_like(b)
    for i in range(len(b)):
        y[i] = int(np.any(b[i+1:i+1+horizon]) if i + 1 < len(b) else 0)
    return y


def eval_early_warning(y_true: np.ndarray, score: np.ndarray) -> pd.DataFrame:
    """
    Evaluate early-warning score with AUC + median-threshold operational metrics.
    """
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    ok = np.isfinite(s)
    y = y[ok]
    s = s[ok]

    if len(np.unique(y)) < 2 or len(y) < 30:
        return pd.DataFrame([{
            "AUC": float("nan"),
            "threshold": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "n_eval": int(len(y)),
            "n_positive": int(np.sum(y))
        }])

    auc = float(roc_auc_score(y, s))
    thr = float(np.median(s))
    yhat = (s >= thr).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y, yhat, average="binary", zero_division=0)

    return pd.DataFrame([{
        "AUC": auc,
        "threshold": thr,
        "precision": float(pr),
        "recall": float(rc),
        "f1": float(f1),
        "n_eval": int(len(y)),
        "n_positive": int(np.sum(y))
    }])


# =============================================================================
# Rolling topology on correlation geometry (for 5.3 early warning)
# =============================================================================
def corr_to_distance(C: np.ndarray) -> np.ndarray:
    """
    Standard correlation-to-distance map used in dependence geometry:
      d_ij = sqrt(2 * (1 - corr_ij))
    """
    C = np.clip(C, -1.0, 1.0)
    D = np.sqrt(np.maximum(0.0, 2.0 * (1.0 - C)))
    np.fill_diagonal(D, 0.0)
    return D


def rolling_corr_topology_metrics(
    R: np.ndarray,
    window: int,
    step: int,
    maxdim: int = 1
) -> pd.DataFrame:
    """
    Compute topological summaries and a Wasserstein-based structural change index
    across rolling correlation-distance windows.

    Output columns are purposely compact: they are meant to serve as the tables
    and quantitative evidence in Section 5.3.
    """
    n, d = R.shape
    rows = []
    prev_dgm1 = None
    wid = 0

    for start in range(0, n - window + 1, step):
        end = start + window
        Ri = R[start:end].copy()

        # mild winsorization for numerical stability
        for j in range(d):
            x = Ri[:, j]
            lo = np.quantile(x, 0.001)
            hi = np.quantile(x, 0.999)
            Ri[:, j] = np.clip(x, lo, hi)

        C = np.corrcoef(Ri, rowvar=False)
        D = corr_to_distance(C)

        dgms = ripser(D, distance_matrix=True, maxdim=maxdim)["dgms"]
        summ = persistence_summaries_H1(dgms)
        dgm1 = dgms[1] if len(dgms) > 1 else np.empty((0, 2))

        if prev_dgm1 is None:
            wdist = float("nan")
        else:
            wdist = float(wasserstein(prev_dgm1, dgm1, matching=False))
        prev_dgm1 = dgm1

        avg_corr = float(C[np.triu_indices_from(C, k=1)].mean())

        rows.append({
            "window_id": wid,
            "start": start,
            "end": end,
            "avg_corr": avg_corr,
            **summ,
            "wasserstein_H1_to_prev": wdist
        })
        wid += 1

    return pd.DataFrame(rows)


# =============================================================================
# Stress scenarios (scenario sensitivity) — 5.3
# =============================================================================
def stress_scenarios_monte_carlo(
    mu: np.ndarray,
    Sigma: np.ndarray,
    n_sims: int,
    seed: int,
    vol_mult: float,
    corr_shock: float,
    jump_prob: float,
    jump_size: float
) -> Dict[str, np.ndarray]:
    """
    Create stressed one-step return simulations under baseline, vol shock, corr shock, jump shock.

    This supports "scenario sensitivity" in 5.3 while remaining minimal and interpretable.
    """
    rng = np.random.default_rng(seed)
    d = Sigma.shape[0]

    base = rng.multivariate_normal(mean=mu, cov=Sigma, size=n_sims)

    # Volatility shock
    Sigma_vol = (vol_mult ** 2) * Sigma
    vol = rng.multivariate_normal(mean=mu, cov=Sigma_vol, size=n_sims)

    # Correlation shock (pull correlation toward 1 by convex combination)
    std = np.sqrt(np.clip(np.diag(Sigma), 1e-12, None))
    Corr = Sigma / np.outer(std, std)
    Corr = np.clip(Corr, -1.0, 1.0)
    Corr_target = np.ones_like(Corr)
    Corr_shocked = (1 - corr_shock) * Corr + corr_shock * Corr_target
    np.fill_diagonal(Corr_shocked, 1.0)
    Sigma_corr = Corr_shocked * np.outer(std, std)
    corr = rng.multivariate_normal(mean=mu, cov=Sigma_corr, size=n_sims)

    # Jump shock applied to baseline
    jump = base.copy()
    events = rng.uniform(size=(n_sims, d)) < jump_prob
    jump = jump - events.astype(float) * jump_size

    return {"baseline": base, "vol_shock": vol, "corr_shock": corr, "jump_shock": jump}


# =============================================================================
# Section 5.1 — Single-asset experiments
# =============================================================================
def run_5_1(
    root: str,
    cfg: SingleAssetConfig,
    risk_alpha: float,
    seed: int,
    logger: logging.Logger
) -> None:
    """
    5.1 outputs (minimal set):
      Tables (one workbook):
        - path_features
        - regime_summary
        - gmm_confusion
        - topology_summary
        - stability_bootstrap

      Figures:
        - pca_regime_separation.png
        - volatility_boxplot.png
        - persistence_diagrams_H1.png (A vs B)

    These outputs directly support:
      * regime separation (PCA + GMM confusion)
      * turbulence signatures (volatility distribution by regime)
      * structural stability (PH summaries + stability score + diagrams)
    """
    sec = ensure_dir(os.path.join(root, "5_1_single_asset"))
    tdir = ensure_dir(os.path.join(sec, "tables"))
    fdir = ensure_dir(os.path.join(sec, "figures"))

    logger.info("5.1 | Simulating single-asset scenario families...")
    R, y = simulate_single_asset_paths(cfg, seed=seed)

    logger.info("5.1 | Building feature embeddings (scenario point cloud)...")
    df_feat = single_asset_path_features(R, alpha=risk_alpha)
    df_feat["true_regime"] = y

    # Standardize features for geometry/topology
    feature_cols = ["mean_ret", "std_ret", "skew_ret", "kurt_ret", "max_drawdown", "VaR_loss", "ES_loss", "turbulence_share"]
    X = df_feat[feature_cols].values
    Xz, muX, sdX = standardize_matrix(X)

    # Regime separation: PCA visualization
    pca = PCA(n_components=2, random_state=0)
    Z2 = pca.fit_transform(Xz)
    df_feat["PC1"] = Z2[:, 0]
    df_feat["PC2"] = Z2[:, 1]

    # Unsupervised regime separation: GMM on standardized features
    gmm = GaussianMixture(n_components=2, random_state=0)
    yhat = gmm.fit_predict(Xz)
    # Align labels so that label 1 corresponds to higher volatility regime
    if np.mean(df_feat.loc[yhat == 1, "std_ret"]) < np.mean(df_feat.loc[yhat == 0, "std_ret"]):
        yhat = 1 - yhat
    df_feat["gmm_regime"] = yhat

    cm = confusion_matrix(y, yhat)
    gmm_conf = pd.DataFrame(cm, index=["true_A", "true_B"], columns=["pred_A", "pred_B"])

    regime_summary = df_feat.groupby("true_regime")[feature_cols].agg(["mean", "std", "median"]).reset_index()
    # Flatten multiindex columns for Excel readability
    regime_summary.columns = ["_".join([c for c in col if c]) if isinstance(col, tuple) else str(col) for col in regime_summary.columns]

    # Topology on scenario-space point cloud (combined and by regime)
    logger.info("5.1 | Computing persistent homology on scenario-space (A, B, combined)...")
    dg_all = compute_ph_on_pointcloud(Xz, maxdim=cfg.ph_maxdim, max_points=cfg.ph_max_points, seed=seed + 10)
    dg_A = compute_ph_on_pointcloud(Xz[y == 0], maxdim=cfg.ph_maxdim, max_points=cfg.ph_max_points, seed=seed + 11)
    dg_B = compute_ph_on_pointcloud(Xz[y == 1], maxdim=cfg.ph_maxdim, max_points=cfg.ph_max_points, seed=seed + 12)

    dgmA = dg_A[1] if len(dg_A) > 1 else np.empty((0, 2))
    dgmB = dg_B[1] if len(dg_B) > 1 else np.empty((0, 2))

    topo_rows = []
    topo_rows.append({"cloud": "A_low_turbulence", **persistence_summaries_H1(dg_A)})
    topo_rows.append({"cloud": "B_high_turbulence", **persistence_summaries_H1(dg_B)})
    topo_rows.append({"cloud": "combined", **persistence_summaries_H1(dg_all)})

    # Diagram distance between regimes (quantitative regime discrimination)
    topo_rows.append({"cloud": "Wasserstein_H1(A,B)", "n_H1": np.nan,
                      "total_persistence_H1": np.nan, "mean_lifetime_H1": np.nan,
                      "entropy_H1": np.nan})
    wAB = float(wasserstein(dgmA, dgmB, matching=False))
    topo_summary = pd.DataFrame(topo_rows)
    topo_summary.loc[topo_summary["cloud"] == "Wasserstein_H1(A,B)", "total_persistence_H1"] = wAB

    # Stability checks via bootstrap + jitter (Section 3.2 evaluation protocol)
    logger.info("5.1 | Stability checks via bootstrap + jitter (median diagram deviation)...")
    stability_A = bootstrap_stability_score(
        Xz[y == 0], base_dgm_H1=dgmA, n_boot=cfg.stability_bootstrap,
        jitter_std=cfg.stability_jitter_std, maxdim=cfg.ph_maxdim,
        max_points=cfg.ph_max_points, seed=seed + 101
    )
    stability_B = bootstrap_stability_score(
        Xz[y == 1], base_dgm_H1=dgmB, n_boot=cfg.stability_bootstrap,
        jitter_std=cfg.stability_jitter_std, maxdim=cfg.ph_maxdim,
        max_points=cfg.ph_max_points, seed=seed + 102
    )
    stability_tbl = pd.DataFrame([
        {"cloud": "A_low_turbulence", "median_wasserstein_to_base_H1": stability_A,
         "bootstrap_reps": cfg.stability_bootstrap, "jitter_std": cfg.stability_jitter_std},
        {"cloud": "B_high_turbulence", "median_wasserstein_to_base_H1": stability_B,
         "bootstrap_reps": cfg.stability_bootstrap, "jitter_std": cfg.stability_jitter_std}
    ])

    # -------------------- Figures (minimal) --------------------
    logger.info("5.1 | Saving figures...")
    # Figure 1: PCA regime separation
    fig1 = plt.figure(figsize=(7.2, 5.8))
    ax = fig1.add_subplot(111)
    for lab, name in [(0, "Regime A (low turbulence)"), (1, "Regime B (high turbulence)")]:
        sub = df_feat[df_feat["true_regime"] == lab]
        ax.scatter(sub["PC1"], sub["PC2"], s=18, label=name, alpha=0.85)
    ax.set_title("5.1 — Regime separation in scenario-space (PCA of feature embeddings)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.25)
    ax.legend()
    save_figure(os.path.join(fdir, "pca_regime_separation.png"), fig1)

    # Figure 2: Turbulence signature via volatility distribution
    fig2 = plt.figure(figsize=(6.8, 4.8))
    ax = fig2.add_subplot(111)
    Avol = df_feat[df_feat["true_regime"] == 0]["std_ret"].values
    Bvol = df_feat[df_feat["true_regime"] == 1]["std_ret"].values
    ax.boxplot([Avol, Bvol], labels=["A (low)", "B (high)"])
    ax.set_title("5.1 — Turbulence signatures (distribution of path volatility)")
    ax.set_ylabel("Std(returns)")
    ax.grid(True, axis="y", alpha=0.25)
    save_figure(os.path.join(fdir, "volatility_boxplot.png"), fig2)

    # Figure 3: Persistence diagrams H1 for A vs B (topological evidence)
    fig3 = plt.figure(figsize=(11.0, 5.0))
    ax1 = fig3.add_subplot(1, 2, 1)
    ax2 = fig3.add_subplot(1, 2, 2)

    def _scatter_diag(axh, dgm, title):
        if dgm is None or dgm.size == 0:
            axh.text(0.5, 0.5, "Empty H1 diagram", ha="center", va="center")
            axh.set_title(title)
            axh.set_xlabel("birth")
            axh.set_ylabel("death")
            axh.grid(True, alpha=0.25)
            return
        finite = np.isfinite(dgm[:, 1])
        pts = dgm[finite]
        if pts.size == 0:
            axh.text(0.5, 0.5, "No finite H1 points", ha="center", va="center")
        else:
            axh.scatter(pts[:, 0], pts[:, 1], s=18)
            mx = float(np.nanmax(pts))
            axh.plot([0, mx], [0, mx], linestyle="--", linewidth=1.0)
        axh.set_title(title)
        axh.set_xlabel("birth")
        axh.set_ylabel("death")
        axh.grid(True, alpha=0.25)

    _scatter_diag(ax1, dgmA, "Regime A (low turbulence) — H1")
    _scatter_diag(ax2, dgmB, "Regime B (high turbulence) — H1")
    fig3.suptitle("5.1 — Persistence diagrams (H1) of scenario-space embeddings")
    save_figure(os.path.join(fdir, "persistence_diagrams_H1.png"), fig3)

    # -------------------- Tables (one workbook) --------------------
    logger.info("5.1 | Saving tables workbook...")
    xlsx = os.path.join(tdir, "5_1_single_asset_tables.xlsx")
    save_excel_with_sheets(xlsx, {
        "path_features": df_feat,
        "regime_summary": regime_summary,
        "gmm_confusion": gmm_conf.reset_index().rename(columns={"index": "true\\pred"}),
        "topology_summary": topo_summary,
        "stability_bootstrap": stability_tbl
    })

    logger.info("5.1 | Completed.")


# =============================================================================
# Section 5.2 — Multi-asset experiments
# =============================================================================
def run_5_2(
    root: str,
    cfg: MultiAssetConfig,
    risk_alpha: float,
    seed: int,
    logger: logging.Logger
) -> None:
    """
    5.2 outputs (minimal set):
      Tables (one workbook):
        - scenario_features
        - regime_summary
        - corr_cloud_pca
        - topology_summary
        - stability_bootstrap

      Figures:
        - pca_dependence_cloud.png
        - diversification_limits_scatter.png
        - corr_heatmaps_normal_vs_crisis.png
        - persistence_diagrams_H1_dependence_cloud.png

    These directly support:
      * dependence topology (PH on correlation-vector cloud)
      * correlation breakdowns (heatmaps + avg_corr distribution)
      * diversification limits (portfolio risk vs dependence intensity)
    """
    sec = ensure_dir(os.path.join(root, "5_2_multi_asset"))
    tdir = ensure_dir(os.path.join(sec, "tables"))
    fdir = ensure_dir(os.path.join(sec, "figures"))

    logger.info("5.2 | Simulating multi-asset scenario families...")
    R, y = simulate_multi_asset_paths(cfg, seed=seed + 200)

    logger.info("5.2 | Building per-scenario features and dependence cloud embedding...")
    df_feat = multi_asset_path_features(R, alpha=risk_alpha)
    df_feat["true_regime"] = y  # 0=Normal, 1=Crisis

    Xcorr = correlation_vector_embedding(R)
    Xz, _, _ = standardize_matrix(Xcorr)

    # PCA for visualization of the high-dimensional cloud
    pca = PCA(n_components=2, random_state=0)
    Z2 = pca.fit_transform(Xz)
    df_pca = pd.DataFrame({"path_id": np.arange(Xz.shape[0]), "PC1": Z2[:, 0], "PC2": Z2[:, 1], "true_regime": y})

    # Dependence topology via PH on the correlation-vector cloud
    logger.info("5.2 | Computing PH on dependence cloud (Normal, Crisis, combined)...")
    dg_all = compute_ph_on_pointcloud(Xz, maxdim=cfg.ph_maxdim, max_points=cfg.ph_max_points, seed=seed + 210)
    dg_N = compute_ph_on_pointcloud(Xz[y == 0], maxdim=cfg.ph_maxdim, max_points=cfg.ph_max_points, seed=seed + 211)
    dg_C = compute_ph_on_pointcloud(Xz[y == 1], maxdim=cfg.ph_maxdim, max_points=cfg.ph_max_points, seed=seed + 212)

    dgmN = dg_N[1] if len(dg_N) > 1 else np.empty((0, 2))
    dgmC = dg_C[1] if len(dg_C) > 1 else np.empty((0, 2))
    wNC = float(wasserstein(dgmN, dgmC, matching=False))

    topo_summary = pd.DataFrame([
        {"cloud": "Normal", **persistence_summaries_H1(dg_N)},
        {"cloud": "Crisis", **persistence_summaries_H1(dg_C)},
        {"cloud": "Combined", **persistence_summaries_H1(dg_all)},
        {"cloud": "Wasserstein_H1(Normal,Crisis)", "n_H1": np.nan,
         "total_persistence_H1": wNC, "mean_lifetime_H1": np.nan, "entropy_H1": np.nan}
    ])

    # Stability checks
    logger.info("5.2 | Stability checks via bootstrap + jitter...")
    stab_N = bootstrap_stability_score(
        Xz[y == 0], base_dgm_H1=dgmN, n_boot=cfg.stability_bootstrap,
        jitter_std=cfg.stability_jitter_std, maxdim=cfg.ph_maxdim,
        max_points=cfg.ph_max_points, seed=seed + 301
    )
    stab_C = bootstrap_stability_score(
        Xz[y == 1], base_dgm_H1=dgmC, n_boot=cfg.stability_bootstrap,
        jitter_std=cfg.stability_jitter_std, maxdim=cfg.ph_maxdim,
        max_points=cfg.ph_max_points, seed=seed + 302
    )
    stability_tbl = pd.DataFrame([
        {"cloud": "Normal", "median_wasserstein_to_base_H1": stab_N,
         "bootstrap_reps": cfg.stability_bootstrap, "jitter_std": cfg.stability_jitter_std},
        {"cloud": "Crisis", "median_wasserstein_to_base_H1": stab_C,
         "bootstrap_reps": cfg.stability_bootstrap, "jitter_std": cfg.stability_jitter_std}
    ])

    # Regime summary for dependence and diversification limits
    summary_cols = ["avg_corr", "max_eig", "participation_ratio", "portfolio_vol", "portfolio_VaR_loss", "portfolio_ES_loss"]
    regime_summary = df_feat.groupby("true_regime")[summary_cols].agg(["mean", "std", "median"]).reset_index()
    regime_summary.columns = ["_".join([c for c in col if c]) if isinstance(col, tuple) else str(col) for col in regime_summary.columns]

    # Representative correlation heatmaps (choose one scenario in each regime)
    idxN = int(df_feat[df_feat["true_regime"] == 0].iloc[0]["path_id"])
    idxC = int(df_feat[df_feat["true_regime"] == 1].iloc[0]["path_id"])
    CN = np.corrcoef(R[idxN], rowvar=False)
    CC = np.corrcoef(R[idxC], rowvar=False)

    # -------------------- Figures (minimal) --------------------
    logger.info("5.2 | Saving figures...")
    # Figure 1: PCA of dependence cloud
    fig1 = plt.figure(figsize=(7.2, 5.8))
    ax = fig1.add_subplot(111)
    for lab, name in [(0, "Normal"), (1, "Crisis")]:
        sub = df_pca[df_pca["true_regime"] == lab]
        ax.scatter(sub["PC1"], sub["PC2"], s=18, label=name, alpha=0.85)
    ax.set_title("5.2 — Dependence cloud (correlation-vector embedding) with regimes (PCA)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.25)
    ax.legend()
    save_figure(os.path.join(fdir, "pca_dependence_cloud.png"), fig1)

    # Figure 2: Diversification limits (portfolio risk vs average correlation)
    fig2 = plt.figure(figsize=(7.0, 5.0))
    ax = fig2.add_subplot(111)
    for lab, name in [(0, "Normal"), (1, "Crisis")]:
        sub = df_feat[df_feat["true_regime"] == lab]
        ax.scatter(sub["avg_corr"], sub["portfolio_vol"], s=20, label=name, alpha=0.85)
    ax.set_title("5.2 — Diversification limits: portfolio volatility vs dependence intensity")
    ax.set_xlabel("Average correlation")
    ax.set_ylabel("Portfolio volatility (equal-weight)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    save_figure(os.path.join(fdir, "diversification_limits_scatter.png"), fig2)

    # Figure 3: Correlation heatmaps (Normal vs Crisis)
    fig3 = plt.figure(figsize=(11.0, 5.0))
    ax1 = fig3.add_subplot(1, 2, 1)
    ax2 = fig3.add_subplot(1, 2, 2)

    im1 = ax1.imshow(CN, aspect="auto")
    ax1.set_title("Normal — correlation")
    ax1.set_xlabel("asset")
    ax1.set_ylabel("asset")
    fig3.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    im2 = ax2.imshow(CC, aspect="auto")
    ax2.set_title("Crisis — correlation")
    ax2.set_xlabel("asset")
    ax2.set_ylabel("asset")
    fig3.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    fig3.suptitle("5.2 — Correlation breakdown visualization (representative scenarios)")
    save_figure(os.path.join(fdir, "corr_heatmaps_normal_vs_crisis.png"), fig3)

    # Figure 4: Persistence diagrams for dependence cloud (Normal vs Crisis)
    fig4 = plt.figure(figsize=(11.0, 5.0))
    ax1 = fig4.add_subplot(1, 2, 1)
    ax2 = fig4.add_subplot(1, 2, 2)

    def _scatter_diag(axh, dgm, title):
        if dgm is None or dgm.size == 0:
            axh.text(0.5, 0.5, "Empty H1 diagram", ha="center", va="center")
            axh.set_title(title)
            axh.set_xlabel("birth")
            axh.set_ylabel("death")
            axh.grid(True, alpha=0.25)
            return
        finite = np.isfinite(dgm[:, 1])
        pts = dgm[finite]
        if pts.size == 0:
            axh.text(0.5, 0.5, "No finite H1 points", ha="center", va="center")
        else:
            axh.scatter(pts[:, 0], pts[:, 1], s=18)
            mx = float(np.nanmax(pts))
            axh.plot([0, mx], [0, mx], linestyle="--", linewidth=1.0)
        axh.set_title(title)
        axh.set_xlabel("birth")
        axh.set_ylabel("death")
        axh.grid(True, alpha=0.25)

    _scatter_diag(ax1, dgmN, "Normal — H1")
    _scatter_diag(ax2, dgmC, "Crisis — H1")
    fig4.suptitle("5.2 — Persistence diagrams (H1) of the dependence cloud")
    save_figure(os.path.join(fdir, "persistence_diagrams_H1_dependence_cloud.png"), fig4)

    # -------------------- Tables (one workbook) --------------------
    logger.info("5.2 | Saving tables workbook...")
    xlsx = os.path.join(tdir, "5_2_multi_asset_tables.xlsx")
    save_excel_with_sheets(xlsx, {
        "scenario_features": df_feat,
        "regime_summary": regime_summary,
        "corr_cloud_pca": df_pca,
        "topology_summary": topo_summary,
        "stability_bootstrap": stability_tbl
    })

    logger.info("5.2 | Completed.")


# =============================================================================
# Section 5.3 — Risk & stress-testing experiments
# =============================================================================
def simulate_long_markov_switching_multivariate(cfg: RiskStressConfig, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a single long multivariate return stream with regime switching in correlation/volatility.
    This supports time-resolved topology monitoring and early-warning alignment with VaR/ES.

    Returns:
      R: (T_long, d)
      regimes: (T_long,) 0=Normal, 1=Crisis
    """
    rng = np.random.default_rng(seed)

    T = cfg.T_long
    d = cfg.d

    P = np.array([[cfg.p_NN, 1 - cfg.p_NN],
                  [1 - cfg.p_CC, cfg.p_CC]], dtype=float)

    regimes = np.zeros(T, dtype=int)
    regimes[0] = 0 if rng.uniform() < 0.85 else 1
    for t in range(1, T):
        prev = regimes[t - 1]
        regimes[t] = 0 if rng.uniform() < P[prev, 0] else 1

    # heavy-tailed innovations
    Z = _scaled_student_t(rng, cfg.df_t, size=(T, d))

    Cn = _corr_matrix(d, cfg.rho_N)
    Ln = np.linalg.cholesky(Cn + 1e-10 * np.eye(d))
    Cc = _corr_matrix(d, cfg.rho_C)
    Lc = np.linalg.cholesky(Cc + 1e-10 * np.eye(d))

    R = np.zeros((T, d), dtype=float)
    for t in range(T):
        if regimes[t] == 0:
            R[t] = cfg.mu + cfg.sigma_N * (Z[t] @ Ln.T)
        else:
            R[t] = cfg.mu + cfg.sigma_C * (Z[t] @ Lc.T)

    return R, regimes


def run_5_3(
    root: str,
    cfg: RiskStressConfig,
    seed: int,
    logger: logging.Logger
) -> None:
    """
    5.3 outputs (minimal set):
      Tables (one workbook):
        - rolling_risk_metrics
        - backtests
        - stress_sensitivity
        - early_warning_performance

      Figures:
        - rolling_VaR_ES_and_topology_signal.png
        - early_warning_ROC.png
        - stress_sensitivity_VaR_ES.png

    These directly support:
      * topology-informed early-warning signals (rolling topology + ROC)
      * VaR/ES fragility (rolling breaches + backtests)
      * scenario sensitivity (stress scenario VaR/ES comparison)
    """
    sec = ensure_dir(os.path.join(root, "5_3_risk_stress"))
    tdir = ensure_dir(os.path.join(sec, "tables"))
    fdir = ensure_dir(os.path.join(sec, "figures"))

    logger.info("5.3 | Simulating long multivariate regime-switching stream...")
    R, reg = simulate_long_markov_switching_multivariate(cfg, seed=seed + 500)

    # Portfolio returns (equal-weight) for risk computations
    w = np.ones(cfg.d) / cfg.d
    port = R @ w

    logger.info("5.3 | Rolling VaR/ES + breaches...")
    risk_df = rolling_var_es(port, lookback=cfg.risk_lookback, alpha=cfg.alpha)

    logger.info("5.3 | Rolling correlation-topology metrics (Wasserstein structural change)...")
    topo_df = rolling_corr_topology_metrics(
        R,
        window=cfg.corr_window,
        step=cfg.corr_step,
        maxdim=1
    )

    # Align topology metrics to the risk timeline by window end time interpolation
    t_risk = risk_df["t"].values.astype(float)
    t_topo = topo_df["end"].values.astype(float)
    sig = topo_df["wasserstein_H1_to_prev"].values.astype(float)

    ok = np.isfinite(sig)
    if np.sum(ok) >= 2:
        topo_sig_interp = np.interp(t_risk, t_topo[ok], sig[ok], left=np.nan, right=np.nan)
    else:
        topo_sig_interp = np.full_like(t_risk, np.nan, dtype=float)

    risk_df["topology_signal"] = topo_sig_interp

    # Early warning target: breach soon
    y = breach_soon_target(risk_df["breach"].values, horizon=cfg.early_warning_horizon)
    ew_tbl = eval_early_warning(y_true=y, score=risk_df["topology_signal"].values)
    ew_tbl["horizon"] = cfg.early_warning_horizon
    ew_tbl["alpha"] = cfg.alpha

    # Backtests (minimal, standard, auditable)
    n = int(len(risk_df))
    x = int(risk_df["breach"].sum())
    LR_pof, p_pof = kupiec_pof_test(n=n, x=x, alpha=cfg.alpha)
    LR_ind, p_ind = christoffersen_independence_test(risk_df["breach"].values)
    backtests = pd.DataFrame([{
        "alpha": cfg.alpha,
        "n_obs": n,
        "n_breaches": x,
        "breach_rate": float(x / n) if n > 0 else float("nan"),
        "kupiec_POF_LR": LR_pof,
        "kupiec_POF_p_value": p_pof,
        "christoffersen_IND_LR": LR_ind,
        "christoffersen_IND_p_value": p_ind
    }])

    # Scenario sensitivity: one-step stress experiments + VaR/ES comparison
    logger.info("5.3 | Stress scenario sensitivity (baseline vs vol/corr/jump shocks)...")
    mu = np.mean(R, axis=0)
    Sigma = np.cov(R, rowvar=False)

    sims = stress_scenarios_monte_carlo(
        mu=mu,
        Sigma=Sigma,
        n_sims=cfg.stress_n_sims,
        seed=seed + 777,
        vol_mult=cfg.stress_vol_mult,
        corr_shock=cfg.stress_corr_shock,
        jump_prob=cfg.stress_jump_prob,
        jump_size=cfg.stress_jump_size
    )

    stress_rows = []
    # Additionally compute a topology sensitivity proxy: Wasserstein distance between
    # dependence clouds (correlation vectors) of baseline vs stressed one-step samples.
    # This is done minimally by constructing correlation clouds from samples:
    # each sample is a point in R^d, but dependence is better captured by correlation.
    # Here, "topology_distance" is defined as Wasserstein distance between H1 diagrams
    # of the point clouds after standardization.
    def topo_distance_between_clouds(A: np.ndarray, B: np.ndarray) -> float:
        Az, _, _ = standardize_matrix(A)
        Bz, _, _ = standardize_matrix(B)
        dA = compute_ph_on_pointcloud(Az, maxdim=1, max_points=350, seed=seed + 900)
        dB = compute_ph_on_pointcloud(Bz, maxdim=1, max_points=350, seed=seed + 901)
        dgmA = dA[1] if len(dA) > 1 else np.empty((0, 2))
        dgmB = dB[1] if len(dB) > 1 else np.empty((0, 2))
        return float(wasserstein(dgmA, dgmB, matching=False))

    # Build baseline cloud for topology distance
    base_cloud = sims["baseline"]

    for scen, arr in sims.items():
        prt = arr @ w
        var, es = _var_es_from_returns(prt, alpha=cfg.alpha)
        td = 0.0 if scen == "baseline" else topo_distance_between_clouds(base_cloud, arr)
        stress_rows.append({"scenario": scen, "VaR": var, "ES": es, "topology_distance_to_baseline_H1": td})
    stress_tbl = pd.DataFrame(stress_rows)

    # -------------------- Figures (minimal) --------------------
    logger.info("5.3 | Saving figures...")

    # Figure 1: Rolling VaR/ES with topology signal overlay
    fig1 = plt.figure(figsize=(12.0, 5.0))
    ax1 = fig1.add_subplot(111)
    ax2 = ax1.twinx()

    ax1.plot(risk_df["t"], risk_df["loss"], linewidth=0.8, label="loss = -return")
    ax1.plot(risk_df["t"], risk_df["VaR"], linewidth=1.0, label="VaR")
    ax1.plot(risk_df["t"], risk_df["ES"], linewidth=1.0, linestyle="--", label="ES")

    breaches = risk_df[risk_df["breach"] == 1]
    if len(breaches) > 0:
        ax1.scatter(breaches["t"], breaches["loss"], s=18, marker="x", label="VaR breach")

    ax2.plot(risk_df["t"], risk_df["topology_signal"], linewidth=1.0, linestyle=":", label="topology signal (Wasserstein H1)")

    ax1.set_title("5.3 — Rolling VaR/ES fragility with topology-informed structural change signal")
    ax1.set_xlabel("t")
    ax1.set_ylabel("loss / VaR / ES")
    ax2.set_ylabel("topology signal")
    ax1.grid(True, alpha=0.25)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    save_figure(os.path.join(fdir, "rolling_VaR_ES_and_topology_signal.png"), fig1)

    # Figure 2: Early-warning ROC (only if feasible)
    fig2_path = os.path.join(fdir, "early_warning_ROC.png")
    score = risk_df["topology_signal"].values
    ok2 = np.isfinite(score)
    y2 = y[ok2]
    s2 = score[ok2]
    if len(np.unique(y2)) >= 2 and len(y2) >= 30:
        fpr, tpr, _ = roc_curve(y2, s2)
        auc = roc_auc_score(y2, s2)
        fig2 = plt.figure(figsize=(6.0, 5.0))
        ax = fig2.add_subplot(111)
        ax.plot(fpr, tpr, linewidth=1.5, label=f"AUC={auc:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0)
        ax.set_title("5.3 — Early-warning ROC: topology signal predicts 'breach soon'")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="lower right")
        save_figure(fig2_path, fig2)

    # Figure 3: Stress sensitivity VaR/ES
    fig3 = plt.figure(figsize=(8.7, 4.6))
    ax = fig3.add_subplot(111)
    x = np.arange(len(stress_tbl))
    ax.bar(x - 0.15, stress_tbl["VaR"].values, width=0.3, label="VaR")
    ax.bar(x + 0.15, stress_tbl["ES"].values, width=0.3, label="ES")
    ax.set_xticks(x)
    ax.set_xticklabels(stress_tbl["scenario"].values, rotation=20, ha="right")
    ax.set_title("5.3 — Scenario sensitivity of VaR/ES (stress-testing)")
    ax.set_ylabel("risk measure")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    save_figure(os.path.join(fdir, "stress_sensitivity_VaR_ES.png"), fig3)

    # -------------------- Tables (one workbook) --------------------
    logger.info("5.3 | Saving tables workbook...")
    xlsx = os.path.join(tdir, "5_3_risk_stress_tables.xlsx")
    save_excel_with_sheets(xlsx, {
        "rolling_risk_metrics": risk_df,
        "backtests": backtests,
        "stress_sensitivity": stress_tbl,
        "early_warning_performance": ew_tbl
    })

    logger.info("5.3 | Completed.")


# =============================================================================
# Orchestration: create folder structure + run all sections
# =============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Topological Monte Carlo — Part 5 results generator (5.1–5.3)")
    p.add_argument("--out_base", type=str, default="results_part5", help="Base name for results folder")
    p.add_argument("--seed", type=int, default=7, help="Random seed for reproducibility")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Build run_id and root output folder
    gcfg = GlobalConfig(out_base=args.out_base, seed=args.seed, run_id=f"{now_stamp()}_seed{args.seed}")
    root = ensure_dir(f"{gcfg.out_base}_{gcfg.run_id}")

    # Logging and configuration persistence (Section 4)
    logger = setup_logging(os.path.join(root, "run.log"))
    logger.info("Run started.")
    logger.info(f"Root output folder: {os.path.abspath(root)}")

    # Save configuration snapshot for auditability
    scfg = SingleAssetConfig()
    mcfg = MultiAssetConfig()
    rcfg = RiskStressConfig()

    config_dump = {
        "GlobalConfig": asdict(gcfg),
        "SingleAssetConfig": asdict(scfg),
        "MultiAssetConfig": asdict(mcfg),
        "RiskStressConfig": asdict(rcfg)
    }
    with open(os.path.join(root, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_dump, f, indent=2)

    # Run sections
    logger.info("Launching Section 5.1 ...")
    run_5_1(root=root, cfg=scfg, risk_alpha=rcfg.alpha, seed=gcfg.seed, logger=logger)

    logger.info("Launching Section 5.2 ...")
    run_5_2(root=root, cfg=mcfg, risk_alpha=rcfg.alpha, seed=gcfg.seed, logger=logger)

    logger.info("Launching Section 5.3 ...")
    run_5_3(root=root, cfg=rcfg, seed=gcfg.seed, logger=logger)

    logger.info("Run completed successfully.")
    logger.info("All tables and figures are stored under 5_1_single_asset/, 5_2_multi_asset/, 5_3_risk_stress/.")
    print("\n====================================")
    print("PART 5 RESULTS GENERATED")
    print("====================================")
    print(f"Output folder: {os.path.abspath(root)}")
    print("====================================\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

