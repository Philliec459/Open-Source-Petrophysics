# petrocore/workflows/tops.py
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


def strip_rename_suffix(folder_name: str) -> str:
    """Remove _rename / _renamed suffix from folder name."""
    if folder_name is None:
        return ""
    return re.sub(r"(_renamed|_rename)\s*$", "", str(folder_name), flags=re.IGNORECASE).strip()


def normalize_name(s: str) -> str:
    """Normalize for robust matching."""
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s).strip().upper())


def load_tops_master(
    tops_xlsx_path: str,
    *,
    well_name_col: str = "Welll Name",   # NOTE: your file uses 'Welll Name' (3 Ls)
    well_no_col: str = "Well No",
    formation_col: str = "Formation",
    top_col: str = "Top (ft)",
) -> pd.DataFrame:
    """Load the master tops table and normalize."""
    if not os.path.exists(tops_xlsx_path):
        raise FileNotFoundError(f"Tops file not found: {os.path.abspath(tops_xlsx_path)}")

    df = pd.read_excel(
        tops_xlsx_path,
        usecols=[well_name_col, well_no_col, formation_col, top_col],
    ).copy()

    df.columns = ["well_name", "well_no", "formation", "top_ft"]
    df["well_name_norm"] = df["well_name"].apply(normalize_name)
    df["formation"] = df["formation"].astype(str).str.strip()
    df["top_ft"] = pd.to_numeric(df["top_ft"], errors="coerce")
    df = df.dropna(subset=["top_ft"]).sort_values(["well_name_norm", "top_ft"]).reset_index(drop=True)
    return df


def get_tops_for_selected_well(tops_master: pd.DataFrame, folder_name: str) -> pd.DataFrame:
    """Filter tops for the selected well folder (without _renamed suffix)."""
    base = strip_rename_suffix(folder_name)
    base_norm = normalize_name(base)

    df_well = tops_master[tops_master["well_name_norm"] == base_norm].copy()
    df_well = df_well.sort_values("top_ft").reset_index(drop=True)
    return df_well


def ensure_depth_col(df: pd.DataFrame, depth_col: str = "DEPT") -> pd.DataFrame:
    """Guarantee a numeric DEPT column (create from index if needed)."""
    out = df.copy()
    if depth_col not in out.columns:
        out[depth_col] = out.index
    out[depth_col] = pd.to_numeric(out[depth_col], errors="coerce")
    out = out.dropna(subset=[depth_col]).sort_values(depth_col).reset_index(drop=True)
    return out


def attach_formation_from_tops(
    df: pd.DataFrame,
    tops_df_well: pd.DataFrame,
    *,
    depth_col: str = "DEPT",
    out_col: str = "FORMATION",
) -> pd.DataFrame:
    """
    Add FORMATION to df by assigning each depth to the most recent top.
    Requires tops_df_well columns: ['formation','top_ft'].
    """
    out = ensure_depth_col(df, depth_col=depth_col)

    if tops_df_well is None or tops_df_well.empty:
        out[out_col] = None
        return out

    t = tops_df_well[["top_ft", "formation"]].copy()
    t["top_ft"] = pd.to_numeric(t["top_ft"], errors="coerce")
    t = t.dropna(subset=["top_ft"]).sort_values("top_ft").reset_index(drop=True)

    # merge_asof assigns the last top <= depth
    out = pd.merge_asof(
        out.sort_values(depth_col),
        t.rename(columns={"top_ft": depth_col}).sort_values(depth_col),
        on=depth_col,
        direction="backward",
    )
    out = out.rename(columns={"formation": out_col})
    return out


def tops_for_plot(tops_df_well: pd.DataFrame) -> tuple[list[float], list[str]]:
    """Return (tops_depths, tops_names) for plotting."""
    if tops_df_well is None or tops_df_well.empty:
        return [], []
    depths = tops_df_well["top_ft"].astype(float).tolist()
    names = tops_df_well["formation"].astype(str).tolist()
    return depths, names
