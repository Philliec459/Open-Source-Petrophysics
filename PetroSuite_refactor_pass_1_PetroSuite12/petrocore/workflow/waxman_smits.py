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


def _safe_clip(a: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(a, lo, hi)

def _as_depth_array(x, npts: int, name: str) -> np.ndarray:
    """
    Normalize x to a 1D float array of length npts.

    Accepts:
      - scalar -> broadcast to length npts
      - array-like length npts -> returned as float 1D

    Raises:
      ValueError if array-like is wrong length.
    """
    if np.isscalar(x) or (isinstance(x, np.ndarray) and x.ndim == 0):
        return np.full(npts, float(x), dtype=float)

    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size != npts:
        raise ValueError(f"{name} must be scalar or length {npts}; got length {arr.size}")
    return arr


def waxman_smits_sw_iterative(
    rt: np.ndarray,
    phit: np.ndarray,
    qv: np.ndarray,
    rw: float,
    *,
    m,
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

    # normalize m to depth array
    m_arr = _as_depth_array(m, npts, "m")

    # masks
    ok = (
        np.isfinite(rt) & (rt > 0) &
        np.isfinite(phit) & (phit > 0) &
        np.isfinite(qv) & (qv >= 0) &
        np.isfinite(m_arr) & (m_arr > 0)
    )
    if not np.any(ok):
        return sw

    # formation factor (assume a=1): F = 1 / phit^m
    F = 1.0 / np.power(phit[ok], m_arr[ok])




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


