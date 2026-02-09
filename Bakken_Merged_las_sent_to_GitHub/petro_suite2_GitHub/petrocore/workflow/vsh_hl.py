# petrocore/workflows/vsh_hl.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd


# -----------------------------
# Robust stats helpers
# -----------------------------
def _hodges_lehmann_location(x: np.ndarray) -> float:
    """
    Hodges–Lehmann estimator of location:
      median of all pairwise averages (x_i + x_j)/2 for i<=j.
    O(n^2) memory/time, but window sizes for shale picking are usually modest.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    # Pairwise averages
    avgs = (x[:, None] + x[None, :]) / 2.0
    return float(np.nanmedian(avgs))


def _hl_sliding(x: np.ndarray, win: int) -> np.ndarray:
    """
    Sliding HL location per-sample using centered window of size win.
    Slow but transparent and robust.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if win < 3:
        raise ValueError("win must be >= 3")
    if win % 2 == 0:
        win += 1  # force odd for centered window

    half = win // 2
    out = np.full(n, np.nan, dtype=float)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = _hodges_lehmann_location(x[lo:hi])
    return out


def _clip01(a: np.ndarray) -> np.ndarray:
    return np.clip(a, 0.0, 1.0)


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class VshHLConfig:
    # which curve to use (default GR; could pass CGR too)
    gr_curve: str

    # method to define clean & shale endpoints
    # "global_quantiles": use quantiles of GR within analysis interval
    # "hl_windows": use HL-smoothed GR then endpoints from quantiles of smoothed
    endpoint_mode: str = "hl_windows"

    # quantiles for clean/shale endpoints
    q_clean: float = 0.05
    q_shale: float = 0.95

    # HL window length in samples (not feet)
    hl_window: int = 61

    # output column name
    out_col: str = "VSH_HL"

    # also store endpoints/diagnostics
    out_gr_clean: str = "GR_CLEAN"
    out_gr_shale: str = "GR_SHALE"
    out_gr_hl: str = "GR_HL"  # HL-smoothed GR


# -----------------------------
# Main workflow
# -----------------------------
def compute_vsh_hl(
    *,
    analysis_df: pd.DataFrame,
    cfg: VshHLConfig,
    logger: Optional[Callable[[str], None]] = None,
) -> dict:
    """
    Compute a robust GR-based Vsh using HL smoothing for shale indicator stability.

    Returns:
      {"analysis_df": df_out, "report": str, "gr_clean": float, "gr_shale": float}
    """
    if analysis_df is None or analysis_df.empty:
        raise ValueError("analysis_df is empty.")
    if cfg.gr_curve not in analysis_df.columns:
        raise ValueError(f"GR curve not found in analysis_df: {cfg.gr_curve}")

    df = analysis_df.copy()
    gr = pd.to_numeric(df[cfg.gr_curve], errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(gr)
    if m.sum() < 10:
        raise ValueError(f"Not enough finite GR samples in '{cfg.gr_curve}' to compute Vsh.")

    # --- HL-smoothed GR (optional but default) ---
    gr_hl = None
    if cfg.endpoint_mode.lower() in ("hl_windows", "hl", "hodges-lehmann"):
        gr_hl = _hl_sliding(gr, win=cfg.hl_window)
        df[cfg.out_gr_hl] = gr_hl
        base = gr_hl
    else:
        base = gr

    base_f = base[np.isfinite(base)]
    if base_f.size < 10:
        raise ValueError("Not enough finite samples after smoothing to compute endpoints.")

    # --- endpoints ---
    q_clean = float(cfg.q_clean)
    q_shale = float(cfg.q_shale)
    if not (0.0 <= q_clean < q_shale <= 1.0):
        raise ValueError("Require 0 <= q_clean < q_shale <= 1")

    gr_clean = float(np.nanquantile(base_f, q_clean))
    gr_shale = float(np.nanquantile(base_f, q_shale))

    # protect divide-by-zero
    denom = (gr_shale - gr_clean)
    if not np.isfinite(denom) or abs(denom) < 1e-9:
        raise ValueError("Invalid GR endpoints (gr_shale ~ gr_clean). Check interval or quantiles.")

    # --- Vsh (linear GR index) ---
    igr = (gr - gr_clean) / denom
    vsh = _clip01(igr)

    df[cfg.out_col] = vsh
    df[cfg.out_gr_clean] = gr_clean
    df[cfg.out_gr_shale] = gr_shale

    report = (
        "=== VSH_HL (GR-based) ===\n"
        f"GR curve        : {cfg.gr_curve}\n"
        f"Endpoint mode   : {cfg.endpoint_mode}\n"
        f"HL window (pts) : {cfg.hl_window if gr_hl is not None else 'N/A'}\n"
        f"q_clean/q_shale : {cfg.q_clean:.3f} / {cfg.q_shale:.3f}\n"
        f"GR_clean        : {gr_clean:.3f}\n"
        f"GR_shale        : {gr_shale:.3f}\n"
        f"Output column   : {cfg.out_col}\n"
    )

    if logger:
        logger(report)

    return {
        "analysis_df": df,
        "report": report,
        "gr_clean": gr_clean,
        "gr_shale": gr_shale,
    }
