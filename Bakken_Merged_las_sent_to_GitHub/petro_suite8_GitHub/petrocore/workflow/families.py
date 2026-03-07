
# petrocore/workflows/families.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd


# -----------------------------
# Small helpers
# -----------------------------
def first_present(cols, candidates):
    s = set(cols)
    for c in candidates:
        if c in s:
            return c
    return None


def ensure_depth_column(df: pd.DataFrame, depth_col: str = "DEPT") -> pd.DataFrame:
    """
    Guarantee a numeric depth column named depth_col.
    If depth_col doesn't exist, it is created from the index.
    """
    out = df.copy()
    if depth_col not in out.columns:
        out[depth_col] = out.index
    out[depth_col] = pd.to_numeric(out[depth_col], errors="coerce")
    out = out.dropna(subset=[depth_col]).sort_values(depth_col).reset_index(drop=True)
    return out


def slice_zone_df(df: pd.DataFrame, top_depth: float, bottom_depth: float, depth_col: str = "DEPT") -> pd.DataFrame:
    """
    Slice by depth using depth_col (created if needed).
    Returns sorted, reset-index df with numeric depth_col.
    """
    top_depth, bottom_depth = (top_depth, bottom_depth) if top_depth <= bottom_depth else (bottom_depth, top_depth)
    d = ensure_depth_column(df, depth_col=depth_col)
    z = d.loc[(d[depth_col] >= top_depth) & (d[depth_col] <= bottom_depth)].copy()
    z = z.sort_values(depth_col).reset_index(drop=True)
    return z


def format_curve_report(curves: dict) -> str:
    return (
        "\nCurves used in our analysis:\n"
        f"  Density : {curves.get('rhob')}\n"
        f"  Neutron : {curves.get('tnph')}\n"
        f"  Rt      : {curves.get('rt')}\n"
        f"  GR      : {curves.get('gr')}\n"
        f"  CGR     : {curves.get('cgr')}\n"
        f"  TCMR    : {curves.get('tcmr')}\n"
        f"  CMRP    : {curves.get('cmrp')}\n"
        f"  CBW     : {curves.get('cbw')}\n"
        f"  BVIE    : {curves.get('bvie')}\n"
        f"  FFI     : {curves.get('ffi')}\n"
        f"  PEF     : {curves.get('pef')}\n"
        f"  DTCO    : {curves.get('dtco')}\n"
        f"  DEPT    : {curves.get('dept')}\n"
    )


def interp_small_gaps_by_depth(
    df: pd.DataFrame,
    cols: list[str],
    *,
    depth_col: str = "DEPT",
    max_gap_ft: float = 2.0,
) -> pd.DataFrame:
    """
    Interpolate NaNs in cols using depth as x, but only fill gaps whose
    depth span <= max_gap_ft. Larger gaps remain NaN. No edge extrapolation.

    - Only fills NaNs bracketed by real values on both sides.
    - Leaves leading/trailing NaNs as NaN.
    """
    out = df.copy()
    out = ensure_depth_column(out, depth_col=depth_col)
    x = out[depth_col].to_numpy(dtype=float)

    for c in cols:
        if c not in out.columns:
            continue

        s0 = pd.to_numeric(out[c], errors="coerce")
        y0 = s0.to_numpy(dtype=float)

        good = np.isfinite(x) & np.isfinite(y0)
        if good.sum() < 2:
            continue

        xg = x[good]
        yg = y0[good]

        # sort & dedupe xg
        order = np.argsort(xg)
        xg, yg = xg[order], yg[order]
        xg_u, idx_u = np.unique(xg, return_index=True)
        yg_u = yg[idx_u]

        # fill all interior missing by 1D interpolation (we undo large gaps/edges next)
        y_fill = y0.copy()
        missing = np.isfinite(x) & ~np.isfinite(y0)
        if np.any(missing):
            y_fill[missing] = np.interp(x[missing], xg_u, yg_u)

        s_fill = pd.Series(y_fill)

        # undo fills for edges + large gaps
        isnan = ~np.isfinite(y0)
        if isnan.any():
            d = np.diff(np.r_[False, isnan, False].astype(int))
            starts = np.where(d == 1)[0]
            ends = np.where(d == -1)[0] - 1

            for st, en in zip(starts, ends):
                left = st - 1
                right = en + 1

                # edge gaps: no extrapolation
                if left < 0 or right >= len(y0):
                    s_fill.iloc[st : en + 1] = np.nan
                    continue

                # must be bracketed by real values
                if (not np.isfinite(y0[left])) or (not np.isfinite(y0[right])):
                    s_fill.iloc[st : en + 1] = np.nan
                    continue

                gap_ft = abs(x[right] - x[left])
                if gap_ft > max_gap_ft:
                    s_fill.iloc[st : en + 1] = np.nan

        out[c] = s_fill.to_numpy()

    return out


# -----------------------------
# Main “family winners” builder
# -----------------------------
@dataclass(frozen=True)
class FamilyCandidates:
    gr: tuple[str, ...] = ("HSGR", "GR", "SGR", "HGR")
    cgr: tuple[str, ...] = ("HCGR", "CGR")

    rhob: tuple[str, ...] = ("RHOZ", "RHOB")
    tnph: tuple[str, ...] = ("TNPH", "NPHI", "CNL", "NPOR")
    rt: tuple[str, ...] = ("AT90", "AF90", "AO90", "ILD", "RT")

    tcmr: tuple[str, ...] = ("PHIT_NMR", "TCMR", "MPHIS")
    cmrp: tuple[str, ...] = ("PHIE_NMR", "CMRP_3MS", "CMRP3MS", "CMRP", "MPHI")

    bvie: tuple[str, ...] = ("BVIE", "BVI_E")
    cbw: tuple[str, ...] = ("CBW",)
    ffi: tuple[str, ...] = ("FFI", "CMFF")

    pef: tuple[str, ...] = ("PEFZ", "PEF8", "PEF", "PE")
    dtco: tuple[str, ...] = ("DTCO", "DTC", "DT", "DTCO3")


def build_analysis_df_with_family_winners(
    merged_df: pd.DataFrame,
    *,
    top_depth: float,
    bottom_depth: float,
    depth_col: str = "DEPT",
    candidates: FamilyCandidates = FamilyCandidates(),
    max_gap_ft: float = 2.0,
    required: tuple[str, ...] = ("rhob", "tnph", "rt"),
    logger: Optional[Callable[[str], None]] = None,
) -> tuple[pd.DataFrame, dict, list[str], str]:
    """
    Build analysis_df for the selected zone and pick curve-family winners.
    Then interpolate short null gaps ONLY on the selected family winners.

    Returns:
      analysis_df               DataFrame (zone-sliced, with numeric DEPT col)
      curves                    dict of chosen curve names
      selected_family_curves     list of curve names interpolated
      report                     formatted text report (for print or QTextEdit)

    Notes:
      - Works whether merged_df depth is in index or already in a DEPT column.
      - Does not require GR/CGR/NMR/etc. to exist; only the fields in `required`.
    """
    if merged_df is None or merged_df.empty:
        raise ValueError("merged_df is missing or empty.")

    # 1) Slice zone and normalize column names
    analysis_df = slice_zone_df(merged_df, top_depth, bottom_depth, depth_col=depth_col)
    analysis_df.columns = [str(c).strip() for c in analysis_df.columns]
    analysis_df = ensure_depth_column(analysis_df, depth_col=depth_col)

    # 2) Pick family winners
    cols = analysis_df.columns

    gr_curve = first_present(cols, candidates.gr)
    cgr_curve = first_present(cols, candidates.cgr)

    rhob_curve = first_present(cols, candidates.rhob)
    tnph_curve = first_present(cols, candidates.tnph)
    rt_curve = first_present(cols, candidates.rt)

    tcmr_curve = first_present(cols, candidates.tcmr)
    cmrp_curve = first_present(cols, candidates.cmrp)

    cbw_curve = first_present(cols, candidates.cbw)
    bvie_curve = first_present(cols, candidates.bvie)
    ffi_curve = first_present(cols, candidates.ffi)

    pef_curve = first_present(cols, candidates.pef)
    dtco_curve = first_present(cols, candidates.dtco)

    curves = {
        "gr": gr_curve,
        "cgr": cgr_curve,
        "rhob": rhob_curve,
        "tnph": tnph_curve,
        "rt": rt_curve,
        "tcmr": tcmr_curve,
        "cmrp": cmrp_curve,
        "cbw": cbw_curve,
        "bvie": bvie_curve,
        "ffi": ffi_curve,
        "pef": pef_curve,
        "dtco": dtco_curve,
        "dept": depth_col,  # we guarantee this column exists
    }

    # 3) Enforce required curves
    missing = [k for k in required if not curves.get(k)]
    if missing:
        preview_cols = list(analysis_df.columns)[:80]
        msg = (
            f"Missing required curve families in zone {top_depth:g}–{bottom_depth:g} ft: {missing}\n"
            f"First 80 columns seen:\n{preview_cols}"
        )
        if logger:
            logger(msg)
        raise ValueError(msg)

    # 4) Interpolate only family winners (short gaps only)
    selected_family_curves = [
        curves.get("gr"),
        curves.get("cgr"),
        curves.get("rhob"),
        curves.get("tnph"),
        curves.get("rt"),
        curves.get("tcmr"),
        curves.get("cmrp"),
        curves.get("cbw"),
        curves.get("bvie"),
        curves.get("ffi"),
        curves.get("pef"),
        curves.get("dtco"),
    ]
    # drop Nones, duplicates, and depth col
    selected_family_curves = [c for c in dict.fromkeys(selected_family_curves) if c and c != depth_col]

    before_nulls = int(analysis_df[selected_family_curves].isna().sum().sum()) if selected_family_curves else 0
    analysis_df = interp_small_gaps_by_depth(
        analysis_df,
        cols=selected_family_curves,
        depth_col=depth_col,
        max_gap_ft=max_gap_ft,
    )
    after_nulls = int(analysis_df[selected_family_curves].isna().sum().sum()) if selected_family_curves else 0

    # 5) Report text (your printout + a bit of QC)
    report = format_curve_report(curves)
    report += (
        f"\nZone: {min(top_depth, bottom_depth):.1f} – {max(top_depth, bottom_depth):.1f} ft"
        f"\nRows: {len(analysis_df):,}"
        f"\nInterpolated SMALL gaps only (<= {max_gap_ft:g} ft) on {len(selected_family_curves)} family-winner curves"
        f"\nNulls (selected curves): before={before_nulls:,}  after={after_nulls:,}\n"
    )

    # Optional: include remaining nulls per curve (only if any remain)
    if selected_family_curves:
        remain = analysis_df[selected_family_curves].isna().sum()
        remain = remain[remain > 0].sort_values(ascending=False)
        if len(remain):
            report += "\nRemaining nulls (selected family curves):\n"
            report += remain.to_string() + "\n"

    if logger:
        logger(report)

    return analysis_df, curves, selected_family_curves, report
