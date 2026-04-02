# petrocore/workflow/phit_chartbook.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None


CHARTBOOK_FILES = {
    "TNPH ρf=1.00 (SLB)": "TNPH_1pt0.xlsx",
    "CNL ρf=1.0 (SLB)": "CNL_1pt0.xlsx",
    "CNL ρf=1.1 (SLB)": "CNL_1pt1.xlsx",
    "TNPH ρf=1.19 (SLB)": "TNPH_1pt19.xlsx",
}

DEFAULT_CHARTBOOK_KEY = "TNPH ρf=1.19 (SLB)"


_chart_cache: dict[str, pd.DataFrame] = {}


@dataclass(frozen=True)
class ChartbookSpec:
    xlsx_path: str
    neutron_col: str = "Neutron"
    rhob_col: str = "RHOB"
    por_col: str = "Porosity"
    rho_matrix_col: str = "Rho_Matrix"


def _chart_path(filename: str) -> Path:
    return Path(__file__).resolve().parent / "chartbooks" / filename


def _normalize(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return (x - lo) / (hi - lo)




def load_chartbook_df(chartbook_key: str = DEFAULT_CHARTBOOK_KEY) -> pd.DataFrame:
    if not chartbook_key:
        chartbook_key = DEFAULT_CHARTBOOK_KEY

    if chartbook_key in _chart_cache:
        return _chart_cache[chartbook_key].copy()

    rel_name = CHARTBOOK_FILES.get(chartbook_key)
    if rel_name is None:
        raise ValueError(
            f"Unknown chartbook key: {chartbook_key!r}. "
            f"Valid keys are: {list(CHARTBOOK_FILES.keys())}"
        )

    file_path = _chart_path(rel_name)
    if not file_path.exists():
        raise FileNotFoundError(f"Chartbook file not found: {file_path}")

    df = pd.read_excel(file_path, index_col=False).copy()
    _chart_cache[chartbook_key] = df

    return df.copy()




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
    if cKDTree is None:
        raise ImportError("scipy is required for cKDTree.")

    if k < 1:
        raise ValueError("k must be >= 1")

    required_cols = ["Neutron", "RHOB", "Porosity", "Rho_Matrix"]
    missing = [c for c in required_cols if c not in chart_df.columns]
    if missing:
        raise ValueError(f"Chartbook file missing columns: {missing}")

    # sanitize log data first
    tnph = np.asarray(tnph, dtype=float).copy()
    rhob = np.asarray(rhob, dtype=float).copy()

    tnph[(tnph < tnph_range[0]) | (tnph > tnph_range[1])] = np.nan
    rhob[(rhob < rhob_range[0]) | (rhob > rhob_range[1])] = np.nan

    # normalize logs
    tn = _normalize(tnph, tnph_range[0], tnph_range[1])
    rb = _normalize(rhob, rhob_range[0], rhob_range[1])

    # sanitize and normalize chartbook
    c_tn_raw = pd.to_numeric(chart_df["Neutron"], errors="coerce").to_numpy(dtype=float)
    c_rb_raw = pd.to_numeric(chart_df["RHOB"], errors="coerce").to_numpy(dtype=float)
    c_por_raw = pd.to_numeric(chart_df["Porosity"], errors="coerce").to_numpy(dtype=float)
    c_rma_raw = pd.to_numeric(chart_df["Rho_Matrix"], errors="coerce").to_numpy(dtype=float)

    valid_chart = (
        np.isfinite(c_tn_raw)
        & np.isfinite(c_rb_raw)
        & np.isfinite(c_por_raw)
        & np.isfinite(c_rma_raw)
    )

    if not valid_chart.any():
        raise ValueError("Chartbook contains no valid numeric rows.")

    c_tn = _normalize(c_tn_raw[valid_chart], tnph_range[0], tnph_range[1])
    c_rb = _normalize(c_rb_raw[valid_chart], rhob_range[0], rhob_range[1])
    c_por = c_por_raw[valid_chart]
    c_rma = c_rma_raw[valid_chart]

    X_chart = np.c_[c_tn, c_rb]
    X_logs = np.c_[tn, rb]

    # valid logs only
    m = np.isfinite(X_logs).all(axis=1)

    phit_full = np.full(len(tnph), np.nan, dtype=float)
    rma_full = np.full(len(tnph), np.nan, dtype=float)

    if not m.any():
        return phit_full, rma_full

    tree = cKDTree(X_chart)
    dists, idx = tree.query(X_logs[m], k=k)

    if k == 1:
        dists = dists[:, None]
        idx = idx[:, None]

    w = 1.0 / np.maximum(dists, eps)
    w_sum = w.sum(axis=1, keepdims=True)

    por_chart = c_por[idx]
    rma_chart = c_rma[idx]

    phit = (w * por_chart).sum(axis=1) / w_sum[:, 0]
    rhomaa = (w * rma_chart).sum(axis=1) / w_sum[:, 0]

    phit_full[m] = phit
    rma_full[m] = rhomaa

    return phit_full, rma_full


def compute_phit_chartbook(
    df: pd.DataFrame,
    rhob_curve: str,
    nphi_curve: str,
    chartbook_key: str = DEFAULT_CHARTBOOK_KEY,
    k: int = 3,
    tnph_range: tuple[float, float] = (-0.05, 0.60),
    rhob_range: tuple[float, float] = (1.90, 3.00),
) -> pd.DataFrame:
    """
    Returns updated dataframe with PHIT and RHOMAA_CHART.
    """
    if df is None or df.empty:
        raise ValueError("analysis_df is empty.")
    if nphi_curve not in df.columns:
        raise ValueError(f"Neutron curve not found: {nphi_curve}")
    if rhob_curve not in df.columns:
        raise ValueError(f"Density curve not found: {rhob_curve}")
    if not chartbook_key:
        raise ValueError("No chartbook key was provided.")

    print(f"[PHIT] Using chartbook: {chartbook_key}")

    chart_df = load_chartbook_df(chartbook_key=chartbook_key)

    tnph = pd.to_numeric(df[nphi_curve], errors="coerce").to_numpy(dtype=float)
    rhob = pd.to_numeric(df[rhob_curve], errors="coerce").to_numpy(dtype=float)

    phit, rhomaa = chartbook_phit_rhomaa_knn(
        tnph=tnph,
        rhob=rhob,
        chart_df=chart_df,
        k=k,
        tnph_range=tnph_range,
        rhob_range=rhob_range,
    )

    out = df.copy()
    out["PHIT"] = phit
    out["RHOMAA_CHART"] = rhomaa

    valid_count = int(np.isfinite(phit).sum())
    print(f"[PHIT] Valid PHIT values: {valid_count} of {len(phit)}")

    if valid_count > 0:
        print(f"[PHIT] PHIT min/max: {np.nanmin(phit):.4f}, {np.nanmax(phit):.4f}")
    else:
        print("[PHIT] No valid PHIT values were calculated.")

    return out