# petrocore/workflows/pickett_mstar_panel.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class PickettConfig:
    dept_col: str = "DEPT"
    phit_col: str = "PHIT"
    vsh_col: str = "VSH_HL"
    qv_col: str = "Qv"

    rt_col: str = "RT"          # pass your family-winner (AT90/ILD/RT)
    cbw_col: Optional[str] = "CBWapp"  # optional, not required for plots

    out_mstar: str = "MSTAR"    # your final M*
    out_swt: str = "Swt"        # optional output
    out_bvw: str = "BVW"        # optional output
    out_bvo: str = "BVO"        # optional output
    out_mstar_app: str = "MSTARAPP"  # optional output


# -----------------------------
# Compute (Qt-friendly)
# -----------------------------
def compute_bvw_bvo_swt_mstar(
    *,
    analysis_df: pd.DataFrame,
    cfg: PickettConfig,
    m: float,
    n: float,
    rw: float,
    mslope: float,
    B: float,
    clamp_bvw_to_phit: bool = True,
    logger: Optional[Callable[[str], None]] = None,
) -> dict:
    """
    Mirrors your logic:

      BVW = PHIT * ((1/PHIT^m) * (rw/RT))^(1/n)
      if BVW > PHIT: BVW = PHIT
      Swt = BVW/PHIT
      BVO = PHIT*(1-Swt)

      MSTARAPP = log10( rw/(RT*(1+rw*B*Qv)) ) / log10(PHIT)
      MSTAR    = VSH_HL*mslope + m

    Adds columns to analysis_df and returns updated df.
    """
    if analysis_df is None or analysis_df.empty:
        raise ValueError("analysis_df is empty.")

    for c in (cfg.phit_col, cfg.vsh_col, cfg.qv_col, cfg.rt_col):
        if c not in analysis_df.columns:
            raise ValueError(f"Missing required column: '{c}'")

    if not (np.isfinite(m) and m > 0):
        raise ValueError("m must be > 0.")
    if not (np.isfinite(n) and n > 0):
        raise ValueError("n must be > 0.")
    if not (np.isfinite(rw) and rw > 0):
        raise ValueError("rw must be > 0.")
    if not np.isfinite(mslope):
        raise ValueError("mslope must be finite.")
    if not (np.isfinite(B) and B >= 0):
        raise ValueError("B must be finite and >= 0.")

    df = analysis_df.copy()

    phit = pd.to_numeric(df[cfg.phit_col], errors="coerce").to_numpy(dtype=float)
    vsh  = pd.to_numeric(df[cfg.vsh_col], errors="coerce").to_numpy(dtype=float)
    qv   = pd.to_numeric(df[cfg.qv_col], errors="coerce").to_numpy(dtype=float)
    rt   = pd.to_numeric(df[cfg.rt_col], errors="coerce").to_numpy(dtype=float)

    # BVW / Swt
    BVW = np.full_like(phit, np.nan, dtype=float)
    Swt = np.full_like(phit, np.nan, dtype=float)

    ok = np.isfinite(phit) & (phit > 0) & np.isfinite(rt) & (rt > 0)
    # BVW = PHIT * ( (1/PHIT^m) * (rw/RT) )^(1/n)
    BVW[ok] = phit[ok] * np.power((1.0 / np.power(phit[ok], m)) * (rw / rt[ok]), 1.0 / n)

    if clamp_bvw_to_phit:
        BVW = np.where(np.isfinite(BVW) & np.isfinite(phit), np.minimum(BVW, phit), BVW)

    Swt[ok] = BVW[ok] / phit[ok]
    Swt = np.clip(Swt, 0.0, 1.0)

    BVO = np.where(np.isfinite(phit) & np.isfinite(Swt), phit * (1.0 - Swt), np.nan)
    BVO = np.clip(BVO, 0.0, 1.0)

    # MSTARAPP (protect logs)
    MSTARAPP = np.full_like(phit, np.nan, dtype=float)
    ok2 = ok & np.isfinite(qv) & (phit > 0) & np.isfinite(phit)
    denom = rt * (1.0 + rw * B * qv)
    good_denom = ok2 & np.isfinite(denom) & (denom > 0) & (phit > 0) & (phit != 1.0)
    # log10(rw/denom)/log10(phit)
    MSTARAPP[good_denom] = np.log10(rw / denom[good_denom]) / np.log10(phit[good_denom])

    # Your final MSTAR trend
    MSTAR = vsh * mslope + m

    df[cfg.out_bvw] = BVW
    df[cfg.out_swt] = Swt
    df[cfg.out_bvo] = BVO
    df[cfg.out_mstar_app] = MSTARAPP
    df[cfg.out_mstar] = MSTAR

    report = (
        "=== Pickett / BVW / M* ===\n"
        f"Rt   : {cfg.rt_col}\n"
        f"PHIT : {cfg.phit_col}\n"
        f"Vsh  : {cfg.vsh_col}\n"
        f"Qv   : {cfg.qv_col}\n"
        f"m,n,rw,mslope,B : {m:g}, {n:g}, {rw:g}, {mslope:g}, {B:g}\n"
        f"Outputs: {cfg.out_bvw}, {cfg.out_bvo}, {cfg.out_swt}, {cfg.out_mstar_app}, {cfg.out_mstar}\n"
    )
    if logger:
        logger(report)

    return {"analysis_df": df, "report": report}


# -----------------------------
# Plot (Panel-friendly)
# -----------------------------
def plot_pickett_and_mstar(
    *,
    analysis_df: pd.DataFrame,
    cfg: PickettConfig,
    m: float,
    n: float,
    rw: float,
    mslope: float,
    B: float,
) -> plt.Figure:
    """
    Creates your 3-panel figure:
      left  : PHIT + BVW fills vs depth
      middle: Pickett plot with Sw lines (Archie-style lines you draw)
      right : Vsh vs MSTARAPP + m* trend line
    """
    df = analysis_df

    # Ensure outputs exist (compute if not)
    need_cols = [cfg.out_bvw, cfg.out_bvo, cfg.out_swt, cfg.out_mstar_app, cfg.out_mstar]
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        res = compute_bvw_bvo_swt_mstar(
            analysis_df=df,
            cfg=cfg,
            m=m, n=n, rw=rw, mslope=mslope, B=B,
        )
        df = res["analysis_df"]

    y = pd.to_numeric(df[cfg.dept_col], errors="coerce").to_numpy(dtype=float)
    phit = pd.to_numeric(df[cfg.phit_col], errors="coerce").to_numpy(dtype=float)
    bvw = pd.to_numeric(df[cfg.out_bvw], errors="coerce").to_numpy(dtype=float)
    vsh = pd.to_numeric(df[cfg.vsh_col], errors="coerce").to_numpy(dtype=float)
    mstarapp = pd.to_numeric(df[cfg.out_mstar_app], errors="coerce").to_numpy(dtype=float)

    rt = pd.to_numeric(df[cfg.rt_col], errors="coerce").to_numpy(dtype=float)

    fig, axs = plt.subplot_mosaic([["left", "middle", "right"]])
    fig.suptitle("Saturations from analysis_df", color="blue", fontsize=16)
    fig.subplots_adjust(top=0.90, wspace=0.2, hspace=0.15)
    fig.set_figheight(7)
    fig.set_figwidth(15)

    # ---- Left: BVW/BVO vs depth ----
    axs["left"].plot(phit, y, "-r", lw=1)
    axs["left"].plot(bvw, y, "-k", lw=1)
    axs["left"].set_title("Bulk Volume Plot", color="blue")
    axs["left"].set_xlabel("BVO/BVW", color="blue")
    axs["left"].set_ylabel("Depth", color="blue")
    axs["left"].set_xlim(0.5, 0.0)
    axs["left"].set_ylim(np.nanmax(y), np.nanmin(y))
    axs["left"].fill_betweenx(y, phit, bvw, color="green", label="BVO")
    axs["left"].fill_betweenx(y, bvw, 0, color="cyan", label="BVW")
    axs["left"].legend()
    axs["left"].grid(True)

    # ---- Right: Vsh vs MSTARAPP + line ----
    axs["right"].set_title("Vsh vs. Mstar_Apparent", color="blue")
    axs["right"].plot(vsh, mstarapp, "r.", label="")
    xline = np.linspace(0.0, 1.0, 200)
    axs["right"].plot(xline, xline * mslope + m, "k-", label="")
    axs["right"].set_xlim(0.0, 1.0)
    axs["right"].set_ylim(0, 7)
    axs["right"].set_ylabel("Mstar Apparent", color="blue")
    axs["right"].set_xlabel("Vsh [v/v]", color="blue")
    axs["right"].grid(True, which="both", ls="-", color="gray")

    # ---- Middle: Pickett + Sw lines ----
    axs["middle"].loglog(rt, phit, "ro")
    axs["middle"].set_xlim(0.01, 1000)
    axs["middle"].set_ylim(0.01, 1)
    axs["middle"].set_title("Pickett Plot", color="blue")
    axs["middle"].set_ylabel("PHIT [v/v]", color="blue")
    axs["middle"].set_xlabel(f"{cfg.rt_col} [ohm-m]", color="blue")
    axs["middle"].grid(True, which="both", ls="-", color="gray")

    sw_plot = (1.0, 0.8, 0.6, 0.4, 0.2)
    phit_plot = np.array([0.01, 1.0], dtype=float)

    for sw in sw_plot:
        rt_line = (rw / (sw**n)) * (1.0 / (phit_plot**m))
        axs["middle"].plot(rt_line, phit_plot, linewidth=2, label=f"SW {int(sw*100)}%")
    axs["middle"].legend(loc="best")

    fig.tight_layout()
    plt.close(fig)
    return fig


# -----------------------------
# Panel wrapper (optional)
# -----------------------------
def pickett_panel_ui(
    analysis_df: pd.DataFrame,
    *,
    cfg: PickettConfig,
    m0: float,
    n0: float,
    rw0: float,
    mslope0: float,
    B0: float,
):
    """
    Returns (ui, sliders) so you can display ui in notebook and then read values.
    """
    import panel as pn
    from bokeh.models.formatters import PrintfTickFormatter

    m_slider = pn.widgets.FloatSlider(
        name="Cementation Exponent 'm_cem' = ",
        start=1.00, end=3.00, step=0.01, value=m0,
        format=PrintfTickFormatter(format="%.2f"),
    )
    n_slider = pn.widgets.FloatSlider(
        name="Saturation Exponent 'n_sat' = ",
        start=1.00, end=3.00, step=0.01, value=n0,
        format=PrintfTickFormatter(format="%.2f"),
    )
    rw_slider = pn.widgets.FloatSlider(
        name="Rw = ",
        start=0.01, end=0.1, step=0.001, value=rw0,
        format=PrintfTickFormatter(format="%.3f"),
    )
    mslope_slider = pn.widgets.FloatSlider(
        name="m* Slope = ",
        start=0.01, end=4.0, step=0.01, value=mslope0,
        format=PrintfTickFormatter(format="%.3f"),
    )

    def _view(m: float, n: float, rw: float, mslope: float):
        fig = plot_pickett_and_mstar(
            analysis_df=analysis_df,
            cfg=cfg,
            m=m, n=n, rw=rw, mslope=mslope, B=B0,
        )
        return fig

    ui = pn.interact(_view, m=m_slider, n=n_slider, rw=rw_slider, mslope=mslope_slider)
    sliders = {"m": m_slider, "n": n_slider, "rw": rw_slider, "mslope": mslope_slider}
    return ui, sliders
