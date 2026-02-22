# petrocore/workflows/vsh.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd


# -----------------------------
# Basic Vsh methods (same math as your code)
# -----------------------------
def vsh_gr(gr: np.ndarray, gr_clean: float, gr_shale: float, clip: bool = True) -> np.ndarray:
    denom = (gr_shale - gr_clean)
    if np.isclose(denom, 0.0):
        raise ValueError("gr_shale - gr_clean is zero. Fix endpoints.")
    v = (np.asarray(gr, dtype=float) - gr_clean) / denom
    v = np.where(np.isfinite(v), v, np.nan)
    if clip:
        v = np.clip(v, 0.0, 1.0)
    return v


def vsh_dt_den(
    dt: np.ndarray,
    den: np.ndarray,
    *,
    dt_matrix: float, den_matrix: float,
    dt_fl: float, den_fl: float,
    dt_shale: float, den_shale: float,
    clip: bool = True,
) -> np.ndarray:
    dt = np.asarray(dt, dtype=float)
    den = np.asarray(den, dtype=float)

    term1 = (den_fl - den_matrix) * (dt - dt_matrix) - (den - den_matrix) * (dt_fl - dt_matrix)
    term2 = (den_fl - den_matrix) * (dt_shale - dt_matrix) - (den_shale - den_matrix) * (dt_fl - dt_matrix)

    if np.isclose(term2, 0.0):
        raise ValueError("DT-DEN shale triangle denominator is ~0. Fix endpoints.")
    v = term1 / term2
    v = np.where(np.isfinite(v), v, np.nan)
    if clip:
        v = np.clip(v, 0.0, 1.0)
    return v


def vsh_cmr(
    nphi: np.ndarray,
    mphi: np.ndarray,
    *,
    nphi_sh: float,
    mphi_sh: float,
    clip: bool = True,
) -> np.ndarray:
    nphi = np.asarray(nphi, dtype=float)
    mphi = np.asarray(mphi, dtype=float)

    # your original formula
    denom = (nphi_sh - mphi_sh)
    if np.isclose(denom, 0.0):
        raise ValueError("nphi_sh - mphi_sh is ~0. Fix endpoints.")
    phi = (mphi * nphi_sh - nphi * mphi_sh) / denom
    v = (nphi - phi) / nphi_sh
    v = np.where(np.isfinite(v), v, np.nan)
    if clip:
        v = np.clip(v, 0.0, 1.0)
    return v


def vsh_nd(
    nphi: np.ndarray,
    den: np.ndarray,
    *,
    nphi_matrix: float, den_matrix: float,
    nphi_fl: float, den_fl: float,
    nphi_shale: float, den_shale: float,
    clip: bool = True,
) -> np.ndarray:
    nphi = np.asarray(nphi, dtype=float)
    den = np.asarray(den, dtype=float)

    term1 = (den_fl - den_matrix) * (nphi - nphi_matrix) - (den - den_matrix) * (nphi_fl - nphi_matrix)
    term2 = (den_fl - den_matrix) * (nphi_shale - nphi_matrix) - (den_shale - den_matrix) * (nphi_fl - nphi_matrix)

    if np.isclose(term2, 0.0):
        raise ValueError("ND shale triangle denominator is ~0. Fix endpoints.")
    v = term1 / term2
    v = np.where(np.isfinite(v), v, np.nan)
    if clip:
        v = np.clip(v, 0.0, 1.0)
    return v


# -----------------------------
# Hodges–Lehmann median (robust)
# (If you already have your own HL function, we can plug it in here.)
# -----------------------------
def hodges_lehmann_1d(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    # HL estimator = median of pairwise averages
    # O(n^2) but n is number of methods (small), so fine.
    pairs = []
    for i in range(x.size):
        for j in range(i, x.size):
            pairs.append(0.5 * (x[i] + x[j]))
    return float(np.nanmedian(pairs))


def hodges_lehmann_rowwise(*cols: np.ndarray) -> np.ndarray:
    M = np.vstack([np.asarray(c, dtype=float) for c in cols]).T  # shape (n, k)
    out = np.full(M.shape[0], np.nan, dtype=float)
    for i in range(M.shape[0]):
        out[i] = hodges_lehmann_1d(M[i, :])
    return out


# -----------------------------
# Config + master compute
# -----------------------------
@dataclass(frozen=True)
class VshConfig:
    dept_col: str = "DEPT"

    # inputs (family winners)
    gr_col: Optional[str] = None   # GR or CGR winner
    rhob_col: Optional[str] = None
    tnph_col: Optional[str] = None
    dtco_col: Optional[str] = None
    cmrp_col: Optional[str] = None  # NMR effective (CMRP/PHIE_NMR)

    # endpoints (your sliders)
    gr_clean: float = 20.0
    gr_shale: float = 120.0

    dt_matrix: float = 55.0
    dt_shale: float = 110.0

    neut_shale: float = 0.32
    mphi_shale: float = 0.00  # your mphi_shale

    # constants for shale triangles (you already use these)
    den_matrix: float = 2.65
    den_shale: float = 2.65
    den_fl: float = 1.10

    dt_fl: float = 140.0

    neut_matrix: float = -0.04
    neut_fl: float = 1.00

    # outputs
    out_vsh_gr: str = "VSH_GR"
    out_vsh_dtden: str = "VSH_DTDEN"
    out_vsh_nd: str = "VSH_ND"
    out_vsh_cmr: str = "VSH_CMR"
    out_vsh_hl: str = "VSH_HL"


def compute_vsh_family(
    analysis_df: pd.DataFrame,
    cfg: VshConfig,
    *,
    logger: Optional[Callable[[str], None]] = None,
) -> dict:
    df = analysis_df.copy()

    # ---- pull arrays (robust to missing) ----
    def arr(col: Optional[str]) -> Optional[np.ndarray]:
        if not col or col not in df.columns:
            return None
        return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)

    gr = arr(cfg.gr_col)
    rhob = arr(cfg.rhob_col)
    tnph = arr(cfg.tnph_col)
    dtco = arr(cfg.dtco_col)
    cmrp = arr(cfg.cmrp_col)

    # ---- compute methods ----
    if gr is not None:
        df[cfg.out_vsh_gr] = vsh_gr(gr, cfg.gr_clean, cfg.gr_shale, clip=True)

    if (dtco is not None) and (rhob is not None) and (cfg.dtco_col is not None):
        df[cfg.out_vsh_dtden] = vsh_dt_den(
            dtco, rhob,
            dt_matrix=cfg.dt_matrix, den_matrix=cfg.den_matrix,
            dt_fl=cfg.dt_fl, den_fl=cfg.den_fl,
            dt_shale=cfg.dt_shale, den_shale=cfg.den_shale,
            clip=True,
        )

    if (tnph is not None) and (rhob is not None):
        df[cfg.out_vsh_nd] = vsh_nd(
            tnph, rhob,
            nphi_matrix=cfg.neut_matrix, den_matrix=cfg.den_matrix,
            nphi_fl=cfg.neut_fl, den_fl=cfg.den_fl,
            nphi_shale=cfg.neut_shale, den_shale=cfg.den_shale,
            clip=True,
        )

    if (tnph is not None) and (cmrp is not None):
        df[cfg.out_vsh_cmr] = vsh_cmr(
            tnph, cmrp,
            nphi_sh=cfg.neut_shale,
            mphi_sh=cfg.mphi_shale,
            clip=True,
        )

    # ---- HL combine (robust, ignores missing) ----
    # Use any subset that exists.
    candidates = []
    for c in [cfg.out_vsh_gr, cfg.out_vsh_nd, cfg.out_vsh_dtden, cfg.out_vsh_cmr]:
        if c in df.columns:
            candidates.append(pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float))

    if len(candidates) == 0:
        df[cfg.out_vsh_hl] = np.nan
        if logger:
            logger("VSH: no methods available to combine into VSH_HL.")
    else:
        df[cfg.out_vsh_hl] = np.clip(hodges_lehmann_rowwise(*candidates), 0.0, 1.0)

    if logger:
        logger("VSH: computed " + ", ".join([c for c in [cfg.out_vsh_gr, cfg.out_vsh_dtden, cfg.out_vsh_nd, cfg.out_vsh_cmr, cfg.out_vsh_hl] if c in df.columns]))

    return {"analysis_df": df}
