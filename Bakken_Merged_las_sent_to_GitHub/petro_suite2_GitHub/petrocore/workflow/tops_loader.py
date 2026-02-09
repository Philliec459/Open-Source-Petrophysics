
# petrocore/workflows/tops_loader.py
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd


# -----------------------------
# Core helpers
# -----------------------------
def strip_rename_suffix(folder_name: str) -> str:
    """Remove _rename / _renamed suffix from folder name."""
    if folder_name is None:
        return ""
    return re.sub(r"(_renamed|_rename)\s*$", "", str(folder_name), flags=re.IGNORECASE).strip()


def normalize_name(s: str) -> str:
    """Normalize for robust matching (spaces + case)."""
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s).strip().upper())


def ensure_depth_column(df: pd.DataFrame, depth_col: str = "DEPT") -> pd.DataFrame:
    """Guarantee a numeric depth column named depth_col (create from index if needed)."""
    out = df.copy()
    if depth_col not in out.columns:
        out[depth_col] = out.index
    out[depth_col] = pd.to_numeric(out[depth_col], errors="coerce")
    out = out.dropna(subset=[depth_col]).sort_values(depth_col).reset_index(drop=True)
    return out


# -----------------------------
# Tops API
# -----------------------------
@dataclass(frozen=True)
class TopsColumns:
    # Your spreadsheet uses "Welll Name" in the snippet (3 Ls).
    # Keep this configurable because some sheets use "Well Name".
    well_name: str = "Welll Name"
    well_no: str = "Well No"
    formation: str = "Formation"
    top_ft: str = "Top (ft)"


def load_tops_master(tops_xlsx_path: str, cols: TopsColumns = TopsColumns()) -> pd.DataFrame:
    """Load the master tops table and normalize."""
    if not os.path.exists(tops_xlsx_path):
        raise FileNotFoundError(f"Tops file not found: {os.path.abspath(tops_xlsx_path)}")

    df = pd.read_excel(
        tops_xlsx_path,
        usecols=[cols.well_name, cols.well_no, cols.formation, cols.top_ft],
    ).copy()

    df.columns = ["well_name", "well_no", "formation", "top_ft"]
    df["well_name_norm"] = df["well_name"].apply(normalize_name)
    df["formation"] = df["formation"].astype(str).str.strip()
    df["top_ft"] = pd.to_numeric(df["top_ft"], errors="coerce")
    df = df.dropna(subset=["top_ft"]).sort_values(["well_name_norm", "top_ft"]).reset_index(drop=True)
    return df


def get_tops_for_selected_well(tops_master: pd.DataFrame, selected_well_folder: str) -> pd.DataFrame:
    """Filter tops for the currently selected well folder (without _renamed)."""
    base = strip_rename_suffix(selected_well_folder)
    base_norm = normalize_name(base)

    df_well = tops_master[tops_master["well_name_norm"] == base_norm].copy()
    df_well = df_well.sort_values("top_ft").reset_index(drop=True)
    return df_well


def tops_for_plot(tops_df_well: pd.DataFrame) -> tuple[list[float], list[str]]:
    """Return (tops_depths, tops_names) for plotting."""
    if tops_df_well is None or tops_df_well.empty:
        return [], []
    return (
        tops_df_well["top_ft"].astype(float).tolist(),
        tops_df_well["formation"].astype(str).tolist(),
    )


def attach_formation_from_tops(
    df: pd.DataFrame,
    tops_df_well: pd.DataFrame,
    *,
    depth_col: str = "DEPT",
    out_col: str = "FORMATION",
) -> pd.DataFrame:
    """
    Add FORMATION to df by assigning each depth to the most recent top.
    Uses merge_asof, so each depth gets the last formation top <= depth.
    """
    out = ensure_depth_column(df, depth_col=depth_col)

    if tops_df_well is None or tops_df_well.empty:
        out[out_col] = None
        return out

    t = tops_df_well[["top_ft", "formation"]].copy()
    t["top_ft"] = pd.to_numeric(t["top_ft"], errors="coerce")
    t = t.dropna(subset=["top_ft"]).sort_values("top_ft").reset_index(drop=True)

    out = pd.merge_asof(
        out.sort_values(depth_col),
        t.rename(columns={"top_ft": depth_col}).sort_values(depth_col),
        on=depth_col,
        direction="backward",
    )
    out = out.rename(columns={"formation": out_col})
    return out


def load_tops_for_workflow(
    *,
    tops_xlsx_path: str,
    selected_well_folder: str,
    merged_df: Optional[pd.DataFrame] = None,
    depth_col: str = "DEPT",
    formation_col: str = "FORMATION",
    cols: TopsColumns = TopsColumns(),
    attach_to_df: bool = True,
    logger: Optional[Callable[[str], None]] = None,
) -> dict:
    """
    High-level convenience function that your notebook OR Qt button can call.

    Returns a dict with:
      - selected_well_folder
      - selected_well_name
      - tops_df_well
      - tops_depths
      - tops
      - merged_df_with_tops  (if merged_df provided and attach_to_df=True)
      - report               (string)

    If attach_to_df=False or merged_df is None, merged_df_with_tops is None.
    """
    tops_master = load_tops_master(tops_xlsx_path, cols=cols)

    selected_well_name = strip_rename_suffix(selected_well_folder)
    tops_df_well = get_tops_for_selected_well(tops_master, selected_well_folder)
    tops_depths, tops = tops_for_plot(tops_df_well)

    report_lines = []
    report_lines.append(f"Tops file: {os.path.abspath(tops_xlsx_path)}")
    report_lines.append(f"Selected well folder: '{selected_well_folder}'")
    report_lines.append(f"Selected well name  : '{selected_well_name}'")

    merged_df_with_tops = None

    if tops_df_well.empty:
        report_lines.append("⚠️ No tops found for this well.")
    else:
        report_lines.append(f"✅ Tops loaded: {len(tops_df_well)} tops")
        report_lines.append("First few tops:")
        preview = tops_df_well[["well_name", "formation", "top_ft"]].head(12)
        report_lines.append(preview.to_string(index=False))

        if merged_df is not None and attach_to_df:
            merged_df_with_tops = attach_formation_from_tops(
                merged_df, tops_df_well, depth_col=depth_col, out_col=formation_col
            )
            report_lines.append(f"✅ Attached '{formation_col}' to merged_df (depth_col='{depth_col}').")

    report = "\n".join(report_lines) + "\n"

    if logger:
        logger(report)

    return {
        "selected_well_folder": selected_well_folder,
        "selected_well_name": selected_well_name,
        "tops_df_well": tops_df_well,
        "tops_depths": tops_depths,
        "tops": tops,
        "merged_df_with_tops": merged_df_with_tops,
        "report": report,
    }
