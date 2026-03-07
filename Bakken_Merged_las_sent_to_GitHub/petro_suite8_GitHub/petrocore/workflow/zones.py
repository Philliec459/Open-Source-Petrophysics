# petrocore/workflows/zones.py
from __future__ import annotations
import pandas as pd

from petrocore.workflows.tops import ensure_depth_col


def slice_zone(
    df: pd.DataFrame,
    top_depth: float,
    base_depth: float,
    *,
    depth_col: str = "DEPT",
) -> pd.DataFrame:
    """Return analysis_df for the zone [top_depth, base_depth] with numeric DEPT column."""
    top_depth, base_depth = (top_depth, base_depth) if top_depth <= base_depth else (base_depth, top_depth)

    d = ensure_depth_col(df, depth_col=depth_col)
    z = d.loc[(d[depth_col] >= top_depth) & (d[depth_col] <= base_depth)].copy()
    z = z.sort_values(depth_col).reset_index(drop=True)
    return z
