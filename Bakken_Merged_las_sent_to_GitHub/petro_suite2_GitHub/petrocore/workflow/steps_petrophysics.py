"""
petrocore/workflow/steps_petrophysics.py

Thin wrappers around your real petrophysics workflow functions.

Each step must:
  - accept (df, params)
  - return df with added/updated columns
  - raise clear errors if required inputs are missing

This keeps UI code clean and lets you debug steps in a notebook.
"""

from __future__ import annotations

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


# -----------------------------
# Small helpers
# -----------------------------
def _require_cols(df: pd.DataFrame, cols: list[str], step_name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{step_name}: missing required columns: {missing}")

def _get(p: Dict[str, Any], key: str, default=None):
    return p.get(key, default)

def _as_float_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce").astype(float)


# ==========================================================
# STEP 1 — Hodges–Lehmann Vsh
# ==========================================================
def step_vsh_hl(df: pd.DataFrame, p: Dict[str, Any]) -> pd.DataFrame:
    """
    Adds: VSH_HL (or column name in params)
    Expects: GR curve available
    Params:
      - gr_curve (default "GR")
      - vsh_col (default "VSH_HL")
      - (any other shale/clean GR params you need)
    """
    gr_curve = _get(p, "gr_curve", "GR")
    vsh_col  = _get(p, "vsh_col", "VSH_HL")

    _require_cols(df, [gr_curve], "step_vsh_hl")
    gr = _as_float_series(df, gr_curve)

    # ------------------------------------------------------
    # TODO: Replace this placeholder with YOUR HL Vsh code.
    # Example placeholder: simple linear Vsh from GR endpoints
    # ------------------------------------------------------
    gr_clean = float(_get(p, "gr_clean", np.nan))
    gr_shale = float(_get(p, "gr_shale", np.nan))
    if np.isfinite(gr_clean) and np.isfinite(gr_shale) and (gr_shale != gr_clean):
        vsh = (gr - gr_clean) / (gr_shale - gr_clean)
        vsh = vsh.clip(lower=0.0, upper=1.0)
    else:
        # If you don't have endpoints configured yet, just return NaNs
        vsh = pd.Series(np.nan, index=df.index)

    df[vsh_col] = vsh
    return df


# ==========================================================
# STEP 2 — Clay bound water model + PHIE
# ==========================================================
def step_cbw(df: pd.DataFrame, p: Dict[str, Any]) -> pd.DataFrame:
    """
    Adds: CBW, PHIE (names configurable)
    Expects: PHIT and VSH curve (or whatever your model uses)

    Params:
      - phit_curve (default "PHIT")
      - vsh_col (default "VSH_HL")
      - cbw_col (default "CBW")
      - phie_col (default "PHIE")
      - (your CBW model params)
    """
    phit_curve = _get(p, "phit_curve", "PHIT")
    vsh_col    = _get(p, "vsh_col", "VSH_HL")
    cbw_col    = _get(p, "cbw_col", "CBW")
    phie_col   = _get(p, "phie_col", "PHIE")

    _require_cols(df, [phit_curve, vsh_col], "step_cbw")

    phit = _as_float_series(df, phit_curve)
    vsh  = _as_float_series(df, vsh_col)

    # ------------------------------------------------------
    # TODO: Replace with YOUR CBW model.
    # Placeholder: CBW = cbw_max * Vsh
    # ------------------------------------------------------
    cbw_max = float(_get(p, "cbw_max", 0.08))  # example
    cbw = (cbw_max * vsh).clip(lower=0.0)

    phie = (phit - cbw).clip(lower=0.0)

    df[cbw_col] = cbw
    df[phie_col] = phie
    return df


# ==========================================================
# STEP 3 — Waxman–Smits Sw + BVW
# ==========================================================
def step_waxman_smits(df: pd.DataFrame, p: Dict[str, Any]) -> pd.DataFrame:
    """
    Adds: SW_WS, BVW_WS (names configurable)
    Expects: Rt + PHIE + Qv (or MSTAR etc, depending on your implementation)

    Params:
      - rt_curve (default "RT")
      - phie_col (default "PHIE")
      - sw_col (default "SW_WS")
      - bvw_col (default "BVW_WS")
      - plus: Rw, m, n, B, Qv curve name, etc.
    """
    rt_curve = _get(p, "rt_curve", "RT")
    phie_col = _get(p, "phie_col", "PHIE")
    sw_col   = _get(p, "sw_col", "SW_WS")
    bvw_col  = _get(p, "bvw_col", "BVW_WS")

    _require_cols(df, [rt_curve, phie_col], "step_waxman_smits")

    Rt  = _as_float_series(df, rt_curve)
    PHIE = _as_float_series(df, phie_col)

    # ------------------------------------------------------
    # TODO: Call YOUR real Waxman–Smits implementation here.
    # You likely have something like:
    #   Sw = waxman_smits_sw_iterative(Rt, PHIE, Qv, Rw, B, m, n)
    #
    # Placeholder below uses Archie as a stand-in.
    # ------------------------------------------------------
    Rw = float(_get(p, "Rw", 0.08))
    m  = float(_get(p, "m", 2.0))
    n  = float(_get(p, "n", 2.0))
    a  = float(_get(p, "a", 1.0))

    # Archie placeholder (NOT Waxman–Smits)
    with np.errstate(divide="ignore", invalid="ignore"):
        Sw = ((a * Rw) / (Rt * (PHIE**m))).pow(1.0 / n)

    Sw = Sw.clip(lower=0.0, upper=1.0)
    BVW = (PHIE * Sw).clip(lower=0.0)

    df[sw_col] = Sw
    df[bvw_col] = BVW
    return df


# ==========================================================
# STEP 4 — Lithology optimization (volumes + depth plot inputs)
# ==========================================================
def step_lith_opt(df: pd.DataFrame, p: Dict[str, Any]) -> pd.DataFrame:
    """
    Adds: VOL_* columns, plus whatever diagnostics you want (misfit, flags).
    Expects: curves needed for your optimization (GR, RHOB, TNPH, DT, PEF, etc.)

    Params:
      - curve names
      - bounds / constraints
      - endmember responses
    """
    # Example curve keys you might use:
    gr_curve   = _get(p, "gr_curve", "GR")
    rhob_curve = _get(p, "rhob_curve", "RHOZ")
    tnph_curve = _get(p, "tnph_curve", "TNPH")

    _require_cols(df, [gr_curve, rhob_curve, tnph_curve], "step_lith_opt")

    # ------------------------------------------------------
    # TODO: Replace with YOUR optimization routine.
    # Typical pattern:
    #   for each depth row -> minimize -> volumes
    #   store volumes back into df
    # ------------------------------------------------------
    # Placeholder: fill NaNs for volumes
    df["VOL_QUARTZ"]   = np.nan
    df["VOL_CALCITE"]  = np.nan
    df["VOL_DOLOMITE"] = np.nan
    df["VOL_SHALE"]    = np.nan
    df["VOL_MUD"]      = np.nan
    df["LITH_MISFIT"]  = np.nan

    return df
