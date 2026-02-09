# petrocore/workflows/waxman_smits.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd


# -------------------------------------------------
# Config
# -------------------------------------------------
@dataclass(frozen=True)
class WaxmanSmitsConfig:
    rt_col: str = "RT"          # you will pass your family-winner (AT90/ILD/RT)
    phit_col: str = "PHIT"
    qv_col: str = "Qv"

    out_sw: str = "SW_WS"
    out_bvwt: str = "BVWT_WS"
    out_bvwe: str = "BVWe_WS"

    cbw_col: str = "CBWapp"     # for BVWe = PHIT*Sw - CBWapp
    qv_cap: float = 5.0         # keep consistent with your Qv cap


# -------------------------------------------------
# Core physics pieces (transparent)
# -------------------------------------------------
def _b_waxman_smits(temperature_f: float) -> float:
    """
    B parameter (brine conductivity factor) for Waxman–Smits.
    Many implementations use a temperature-dependent B.
    If you already use a different B model in petrocore, swap this.
    Here we provide a simple, common approximation.

    IMPORTANT: If you have a preferred equation, tell me and I'll drop it in.
    """
    # A mild, usable default (not high-stakes accurate).
    # You can replace with your existing petrocore B(T) function.
    # Typical B is ~4–6 (1/ohm·m)/meq/cc depending on units.
    # We'll keep it as a parameter in the solver anyway.
    return 4.8


def _safe_clip(a: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(a, lo, hi)


def waxman_smits_sw_iterative(
    rt: np.ndarray,
    phit: np.ndarray,
    qv: np.ndarray,
    rw: float,
    *,
    m: float,
    n: float,
    B: float,
    max_iter: int = 60,
    tol: float = 1e-6,
    sw0: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Iterative Waxman–Smits water saturation solve for Sw.

    Uses the common form:
      1/Rt = (1/F) * (Sw^n) * ( (1/Rw) + B*Qv/Sw )

    where F = a / phit^m (Archie formation factor), often a=1.
    Rearranged and solved by fixed-point iteration.

    Notes:
      - This is deliberately written clearly (not "clever") for transparency.
      - Handles NaNs and invalid inputs safely.
    """
    rt = np.asarray(rt, dtype=float)
    phit = np.asarray(phit, dtype=float)
    qv = np.asarray(qv, dtype=float)

    npts = len(rt)
    sw = np.full(npts, np.nan, dtype=float)

    # masks
    ok = np.isfinite(rt) & (rt > 0) & np.isfinite(phit) & (phit > 0) & np.isfinite(qv) & (qv >= 0)
    if not np.any(ok):
        return sw

    # formation factor (assume a=1)
    F = 1.0 / np.power(phit[ok], m)

    # initial guess
    if sw0 is None:
        swk = np.full(ok.sum(), 0.6, dtype=float)
    else:
        sw0 = np.asarray(sw0, dtype=float)
        swk = sw0[ok].copy()
        swk[~np.isfinite(swk)] = 0.6
        swk = _safe_clip(swk, 1e-4, 1.0)

    inv_rt = 1.0 / rt[ok]
    inv_rw = 1.0 / rw
    qv_ok = qv[ok]

    # fixed-point iteration
    for _ in range(max_iter):
        # conductivity term: inv_rw + B*Qv/Sw
        term = inv_rw + (B * qv_ok / np.maximum(swk, 1e-6))

        # target: inv_rt = (1/F) * Sw^n * term
        # => Sw_new = [ inv_rt * F / term ]^(1/n)
        rhs = (inv_rt * F) / np.maximum(term, 1e-12)
        sw_new = np.power(np.maximum(rhs, 1e-12), 1.0 / n)

        sw_new = _safe_clip(sw_new, 1e-4, 1.0)

        # check convergence
        if np.nanmax(np.abs(sw_new - swk)) < tol:
            swk = sw_new
            break
        swk = sw_new

    sw[ok] = swk
    return sw


# -------------------------------------------------
# Workflow wrapper
# -------------------------------------------------
def compute_waxman_smits(
    *,
    analysis_df: pd.DataFrame,
    cfg: WaxmanSmitsConfig,
    rw: float,
    m: float,
    n: float,
    B: Optional[float] = None,
    temperature_f: Optional[float] = None,
    logger: Optional[Callable[[str], None]] = None,
) -> dict:
    """
    Adds:
      SW_WS, BVWT_WS, BVWe_WS
    to analysis_df.

    BVWT_WS = PHIT * SW_WS
    BVWe_WS = PHIT * SW_WS - CBWapp
    """
    if analysis_df is None or analysis_df.empty:
        raise ValueError("analysis_df is empty.")

    for c in (cfg.rt_col, cfg.phit_col, cfg.qv_col):
        if c not in analysis_df.columns:
            raise ValueError(f"Missing required column in analysis_df: '{c}'")

    if cfg.cbw_col not in analysis_df.columns:
        raise ValueError(
            f"Missing CBW column '{cfg.cbw_col}'. Run CBW step first or change cfg.cbw_col."
        )

    if not np.isfinite(rw) or rw <= 0:
        raise ValueError("rw must be > 0.")
    if not np.isfinite(m) or m <= 0:
        raise ValueError("m must be > 0.")
    if not np.isfinite(n) or n <= 0:
        raise ValueError("n must be > 0.")

    if B is None:
        if temperature_f is None:
            # reasonable fallback constant if you don't provide temperature
            B = 4.8
        else:
            B = _b_waxman_smits(float(temperature_f))

    df = analysis_df.copy()

    rt = pd.to_numeric(df[cfg.rt_col], errors="coerce").to_numpy(dtype=float)
    phit = pd.to_numeric(df[cfg.phit_col], errors="coerce").to_numpy(dtype=float)
    qv = pd.to_numeric(df[cfg.qv_col], errors="coerce").to_numpy(dtype=float)
    qv = np.clip(qv, 0.0, cfg.qv_cap)

    sw = waxman_smits_sw_iterative(
        rt=rt,
        phit=phit,
        qv=qv,
        rw=rw,
        m=m,
        n=n,
        B=float(B),
        max_iter=60,
        tol=1e-6,
        sw0=None,
    )

    bvwt = np.clip(phit * sw, 0.0, 1.0)

    cbwapp = pd.to_numeric(df[cfg.cbw_col], errors="coerce").to_numpy(dtype=float)
    bvwe = np.clip(bvwt - cbwapp, 0.0, 1.0)

    df[cfg.out_sw] = sw
    df[cfg.out_bvwt] = bvwt
    df[cfg.out_bvwe] = bvwe

    n_valid = int(np.isfinite(sw).sum())

    report = (
        "=== WAXMAN–SMITS Sw ===\n"
        f"Rt column     : {cfg.rt_col}\n"
        f"PHIT column   : {cfg.phit_col}\n"
        f"Qv column     : {cfg.qv_col}\n"
        f"CBW column    : {cfg.cbw_col}\n"
        f"rw, m, n      : {rw:g}, {m:g}, {n:g}\n"
        f"B             : {float(B):g}\n"
        f"Outputs       : {cfg.out_sw}, {cfg.out_bvwt}, {cfg.out_bvwe}\n"
        f"Valid Sw      : {n_valid:,} / {len(df):,}\n"
    )

    if logger:
        logger(report)

    return {"analysis_df": df, "report": report}
