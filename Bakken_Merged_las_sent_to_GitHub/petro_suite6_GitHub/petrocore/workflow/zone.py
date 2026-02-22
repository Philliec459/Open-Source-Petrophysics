# petrocore/workflows/zone.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


def first_present(cols, candidates):
    s = set(cols)
    for c in candidates:
        if c in s:
            return c
    return None


def ensure_depth_column(df: pd.DataFrame, depth_col: str = "DEPT") -> pd.DataFrame:
    d = df.copy()
    if depth_col not in d.columns:
        d[depth_col] = d.index
    d[depth_col] = pd.to_numeric(d[depth_col], errors="coerce")
    d = d.dropna(subset=[depth_col])
    return d


def df_depth_limits_any(df: pd.DataFrame, depth_col: str = "DEPT") -> tuple[float, float]:
    if depth_col in df.columns:
        dep = pd.to_numeric(df[depth_col], errors="coerce").to_numpy()
    else:
        dep = pd.to_numeric(pd.Index(df.index), errors="coerce").to_numpy()
    dep = dep[np.isfinite(dep)]
    if dep.size == 0:
        raise ValueError("No finite depths found in DEPT or index.")
    return float(dep.min()), float(dep.max())


def curve_depth_limits_any(df: pd.DataFrame, curve: str, depth_col: str = "DEPT") -> tuple[float, float]:
    if curve not in df.columns:
        raise ValueError(f"{curve} not present in df.")
    d = ensure_depth_column(df, depth_col=depth_col)
    vals = pd.to_numeric(d[curve], errors="coerce").to_numpy()
    dep = d[depth_col].to_numpy(dtype=float)
    m = np.isfinite(vals) & np.isfinite(dep)
    if not np.any(m):
        raise ValueError(f"No finite values found for {curve}.")
    return float(dep[m].min()), float(dep[m].max())


def slice_zone_df(df: pd.DataFrame, top_depth: float, bottom_depth: float, depth_col: str = "DEPT") -> pd.DataFrame:
    top_depth, bottom_depth = (top_depth, bottom_depth) if top_depth <= bottom_depth else (bottom_depth, top_depth)
    d = ensure_depth_column(df, depth_col=depth_col)
    z = d.loc[(d[depth_col] >= top_depth) & (d[depth_col] <= bottom_depth)].copy()
    z = z.sort_values(depth_col).reset_index(drop=True)
    return z


def build_units_map_for_df(df: pd.DataFrame, base_units: Optional[dict] = None) -> dict:
    u = dict(base_units) if isinstance(base_units, dict) else {}
    for c in df.columns:
        u.setdefault(c, "")
    for c in ["PHIT_NMR", "PHIE_NMR", "FFI", "BVIE", "BVI_E", "CBW"]:
        if c in df.columns and not u.get(c, ""):
            u[c] = "v/v"
    return u


def nice_label(mnemonic: str, units_map: Optional[dict]) -> str:
    if not units_map:
        return mnemonic
    u = (units_map.get(mnemonic, "") or "").strip()
    return f"{mnemonic}\n[{u}]" if u else mnemonic
