# petrocore/workflows/build_analysis.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd

from petrocore.workflows.tops_loader import (
    TopsColumns,
    load_tops_for_workflow,
    attach_formation_from_tops,
)
from petrocore.workflows.families import (
    FamilyCandidates,
    build_analysis_df_with_family_winners,
)


@dataclass(frozen=True)
class BuildAnalysisConfig:
    # --- depth ---
    depth_col: str = "DEPT"

    # --- tops ---
    tops_xlsx_path: Optional[str] = None
    tops_cols: TopsColumns = TopsColumns()
    formation_col: str = "FORMATION"
    attach_tops_to_merged_df: bool = True
    attach_tops_to_analysis_df: bool = True

    # --- family winners + interpolation ---
    candidates: FamilyCandidates = FamilyCandidates()
    max_gap_ft: float = 2.0
    required: tuple[str, ...] = ("rhob", "tnph", "rt")


def build_analysis_workflow(
    *,
    merged_df: pd.DataFrame,
    selected_well_folder: Optional[str],
    top_depth: float,
    bottom_depth: float,
    config: BuildAnalysisConfig = BuildAnalysisConfig(),
    logger: Optional[Callable[[str], None]] = None,
) -> dict:
    """
    End-to-end workflow:
      1) (Optional) Load tops for selected well + attach FORMATION
      2) Slice merged_df to zone -> analysis_df
      3) Pick family winners
      4) Interpolate short gaps only on family winners
      5) Return everything needed for the next buttons (chartbook PHIT, Vsh, etc.)

    Returns dict:
      - merged_df (possibly with FORMATION)
      - analysis_df
      - curves (dict)
      - selected_family_curves (list)
      - tops_df_well
      - tops_depths
      - tops
      - report (string)
    """
    if merged_df is None or merged_df.empty:
        raise ValueError("merged_df is missing or empty.")

    report_parts: list[str] = []
    def log(msg: str):
        if logger:
            logger(msg)

    # -----------------------------
    # 1) Tops (optional)
    # -----------------------------
    tops_df_well = None
    tops_depths, tops = [], []
    merged_df_out = merged_df

    if config.tops_xlsx_path and selected_well_folder:
        res = load_tops_for_workflow(
            tops_xlsx_path=config.tops_xlsx_path,
            selected_well_folder=selected_well_folder,
            merged_df=merged_df_out,
            depth_col=config.depth_col,
            formation_col=config.formation_col,
            cols=config.tops_cols,
            attach_to_df=config.attach_tops_to_merged_df,
            logger=None,  # we combine reports ourselves
        )
        tops_df_well = res["tops_df_well"]
        tops_depths = res["tops_depths"]
        tops = res["tops"]

        if res["merged_df_with_tops"] is not None:
            merged_df_out = res["merged_df_with_tops"]

        report_parts.append("=== TOPS ===")
        report_parts.append(res["report"].rstrip())

    else:
        report_parts.append("=== TOPS ===")
        report_parts.append("Tops step skipped (no tops_xlsx_path or no selected_well_folder).")

    # -----------------------------
    # 2–4) analysis_df + family winners + interpolation
    # -----------------------------
    analysis_df, curves, selected_family_curves, fam_report = build_analysis_df_with_family_winners(
        merged_df_out,
        top_depth=top_depth,
        bottom_depth=bottom_depth,
        depth_col=config.depth_col,
        candidates=config.candidates,
        max_gap_ft=config.max_gap_ft,
        required=config.required,
        logger=None,
    )

    # Optionally attach FORMATION to analysis_df too
    if config.attach_tops_to_analysis_df and (tops_df_well is not None) and (not tops_df_well.empty):
        analysis_df = attach_formation_from_tops(
            analysis_df,
            tops_df_well,
            depth_col=config.depth_col,
            out_col=config.formation_col,
        )

    report_parts.append("=== ANALYSIS DF ===")
    report_parts.append(fam_report.rstrip())

    report = "\n".join(report_parts).rstrip() + "\n"
    log(report)

    return {
        "merged_df": merged_df_out,
        "analysis_df": analysis_df,
        "curves": curves,
        "selected_family_curves": selected_family_curves,
        "tops_df_well": tops_df_well,
        "tops_depths": tops_depths,
        "tops": tops,
        "report": report,
    }
