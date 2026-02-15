# petrocore/workflows/chartbook_porosity.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None


# -----------------------------
# Core helpers
# -----------------------------
def _normalize(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return (x - lo) / (hi - lo)


# -----------------------------
# Fast KNN (your core)
# -----------------------------
def chartbook_phit_rhomaa_knn(
    tnph: np.ndarray,
    rhob: np.ndarray,
    chart_df: pd.DataFrame,
    *,
    k: int = 3,
    tnph_range: tuple[float, float] = (-0.05, 0.60),
    rhob_range: tuple[float, float] = (1.90, 3.00),
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute PHIT and RHOMAA using inverse-distance weighted KNN
    in normalized (TNPH, RHOB) space.

    chart_df must have columns:
      - 'Neutron'  (TNPH/CNL axis)
      - 'RHOB'
      - 'Porosity'
      - 'Rho_Matrix'
    """
    if cKDTree is None:
        raise ImportError(
            "scipy is required for cKDTree. Install scipy or we can provide a pure-numpy fallback."
        )
    if k < 1:
        raise ValueError("k must be >= 1")

    # --- normalize log data ---
    tn = _normalize(tnph, tnph_range[0], tnph_range[1])
    rb = _normalize(rhob, rhob_range[0], rhob_range[1])

    # --- normalize chartbook data ---
    c_tn = _normalize(chart_df["Neutron"].to_numpy(dtype=float), tnph_range[0], tnph_range[1])
    c_rb = _normalize(chart_df["RHOB"].to_numpy(dtype=float), rhob_range[0], rhob_range[1])

    X_chart = np.c_[c_tn, c_rb]
    X_logs = np.c_[tn, rb]

    # mask invalid log points
    m = np.isfinite(X_logs).all(axis=1)

    # build tree on chart points
    tree = cKDTree(X_chart)

    # query KNN for valid log points
    dists, idx = tree.query(X_logs[m], k=k)

    # IMPORTANT: scipy returns 1D arrays when k==1; force 2D
    if k == 1:
        dists = dists[:, None]
        idx = idx[:, None]

    # inverse-distance weights
    w = 1.0 / np.maximum(dists, eps)
    w_sum = w.sum(axis=1, keepdims=True)

    por_chart = chart_df["Porosity"].to_numpy(dtype=float)[idx]
    rma_chart = chart_df["Rho_Matrix"].to_numpy(dtype=float)[idx]

    phit = (w * por_chart).sum(axis=1) / w_sum[:, 0]
    rhomaa = (w * rma_chart).sum(axis=1) / w_sum[:, 0]

    # reinsert into full-length arrays
    phit_full = np.full(len(tnph), np.nan, dtype=float)
    rma_full = np.full(len(tnph), np.nan, dtype=float)
    phit_full[m] = phit
    rma_full[m] = rhomaa

    return phit_full, rma_full


# -----------------------------
# Workflow wrapper (button-ready)
# -----------------------------
@dataclass(frozen=True)
class ChartbookSpec:
    xlsx_path: str
    neutron_col: str = "Neutron"
    rhob_col: str = "RHOB"
    por_col: str = "Porosity"
    rho_matrix_col: str = "Rho_Matrix"


def compute_chartbook_porosity(
    *,
    analysis_df: pd.DataFrame,
    tnph_curve: str,
    rhob_curve: str,
    chartbook: ChartbookSpec,
    k: int = 3,
    tnph_range: tuple[float, float] = (-0.05, 0.60),
    rhob_range: tuple[float, float] = (1.90, 3.00),
    out_phit: str = "PHIT",
    out_rhomaa: str = "RHOMAA",
    logger: Optional[Callable[[str], None]] = None,
) -> dict:
    """
    Reads chartbook XLSX, computes PHIT/RHOMAA, writes columns into analysis_df.

    Returns:
      {"analysis_df": df_out, "report": str}
    """
    if analysis_df is None or analysis_df.empty:
        raise ValueError("analysis_df is empty.")
    if tnph_curve not in analysis_df.columns:
        raise ValueError(f"Neutron curve not found in analysis_df: {tnph_curve}")
    if rhob_curve not in analysis_df.columns:
        raise ValueError(f"Density curve not found in analysis_df: {rhob_curve}")

    # load chartbook
    chart_df = pd.read_excel(chartbook.xlsx_path, index_col=False).copy()

    # normalize expected chartbook column names to what chartbook_phit_rhomaa_knn expects
    rename_map = {
        chartbook.neutron_col: "Neutron",
        chartbook.rhob_col: "RHOB",
        chartbook.por_col: "Porosity",
        chartbook.rho_matrix_col: "Rho_Matrix",
    }
    missing = [src for src in rename_map.keys() if src not in chart_df.columns]
    if missing:
        raise ValueError(f"Chartbook file missing columns: {missing}")

    chart_df = chart_df.rename(columns=rename_map)

    tnph = pd.to_numeric(analysis_df[tnph_curve], errors="coerce").to_numpy(dtype=float)
    rhob = pd.to_numeric(analysis_df[rhob_curve], errors="coerce").to_numpy(dtype=float)

    phit, rhomaa = chartbook_phit_rhomaa_knn(
        tnph,
        rhob,
        chart_df,
        k=k,
        tnph_range=tnph_range,
        rhob_range=rhob_range,
    )

    df_out = analysis_df.copy()
    df_out[out_phit] = phit
    df_out[out_rhomaa] = rhomaa

    n_valid = int(np.isfinite(phit).sum())

    report = (
        "=== CHARTBOOK POROSITY (KNN) ===\n"
        f"Chartbook file : {chartbook.xlsx_path}\n"
        f"TNPH curve     : {tnph_curve}\n"
        f"RHOB curve     : {rhob_curve}\n"
        f"k neighbors    : {k}\n"
        f"Valid samples  : {n_valid:,} / {len(df_out):,}\n"
        f"Outputs        : {out_phit}, {out_rhomaa}\n"
    )

    if logger:
        logger(report)

    return {"analysis_df": df_out, "report": report}
