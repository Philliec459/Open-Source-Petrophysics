from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd


def find_depth_col(df: pd.DataFrame) -> str | None:
    if df is None or df.empty:
        return None
    for col in df.columns:
        if str(col).strip().upper() in {"DEPT", "DEPTH", "MD"}:
            return col
    return None


def zoi_mask(df: pd.DataFrame, top: float | None, base: float | None) -> pd.Series:
    """Return a Boolean mask for the current zone of interest."""
    if df is None or df.empty:
        return pd.Series(dtype=bool)

    depth_col = find_depth_col(df)
    if depth_col is None or top is None or base is None:
        return pd.Series(True, index=df.index)

    z_top = min(float(top), float(base))
    z_base = max(float(top), float(base))
    depth = pd.to_numeric(df[depth_col], errors="coerce")
    return ((depth >= z_top) & (depth <= z_base)).fillna(False)


def apply_to_zoi(
    df: pd.DataFrame,
    compute_func: Callable[..., pd.DataFrame],
    *,
    top: float | None = None,
    base: float | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Apply a pure petrocore compute function only inside the ZoI and merge
    newly computed columns back into the full dataframe.
    """
    if df is None or df.empty:
        return df

    work_df = df.copy()
    mask = zoi_mask(work_df, top, base)
    if mask is None or not mask.any():
        return work_df

    subset = work_df.loc[mask].copy()
    out_subset = compute_func(subset, **kwargs)

    for col in out_subset.columns:
        if col not in work_df.columns:
            work_df[col] = pd.NA

    for col in out_subset.columns:
        if col not in subset.columns or not out_subset[col].equals(subset.get(col)):
            work_df.loc[mask, col] = out_subset[col].values

    return work_df
