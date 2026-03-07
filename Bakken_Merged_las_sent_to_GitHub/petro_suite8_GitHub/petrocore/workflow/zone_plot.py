# petrocore/workflows/zone_plot.py
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

from petrocore.workflow.zone import (
    first_present, slice_zone_df, nice_label
)


def plot_zone_template_from_df(
    df_in: pd.DataFrame,
    units_map: Optional[dict],
    top_depth: float,
    bottom_depth: float,
    *,
    title: Optional[str] = None,
    depth_col: str = "DEPT",
    tops_depths: Optional[list[float]] = None,
    tops: Optional[list[str]] = None,
) -> plt.Figure:

    if df_in is None or df_in.empty:
        raise ValueError("merged_df is empty.")

    top_depth, bottom_depth = (top_depth, bottom_depth) if top_depth <= bottom_depth else (bottom_depth, top_depth)

    z = slice_zone_df(df_in, top_depth, bottom_depth, depth_col=depth_col)
    if z.empty:
        raise ValueError(f"No data in zone {top_depth}–{bottom_depth}.")

    depth = z[depth_col].to_numpy(dtype=float)

    # Candidate lists (same as yours)
    gr_cands  = ["GR", "SGR", "HGR", "HSGR"]
    cgr_cands = ["CGR", "HCGR"]

    cali_cands = ["CALI","CALS","HCAL","CAL","CALIPER","HD1_PPC1","HD2_PPC1"]
    bs_cands   = ["BS","BITSIZE","BIT","BIT_SIZE"]
    dcal_cands = ["DCAL","DCALI","D_CAL","CALD"]

    rhob_cands = ["RHOB", "RHOZ"]
    drho_cands = ["DRHO", "HDRA", "HDRH", "DROH", "DRH", "RHOC"]
    tnph_cands = ["TNPH", "NPHI", "NPOR", "CNL"]
    tcmr_cands = ["PHIT_NMR", "TCMR", "MPHIS"]
    cmrp_cands = ["PHIE_NMR", "CMRP_3MS", "CMRP3MS", "CMRP", "MPHI"]

    rxo_cands  = ["RXOZ", "Rxo", "RXO", "RxoZ"]
    at90_cands = ["AT90", "AF90", "AO90", "ILD", "RT"]
    at60_cands = ["AT60", "AF60", "AO60"]
    at30_cands = ["AT30", "AF30", "AO30"]
    at20_cands = ["AT20", "AF20", "AO20"]
    at10_cands = ["AT10", "AF10", "AO10"]

    bvie_cands = ["BVIE", "BVI_E"]
    cbw_cands  = ["CBW"]
    ffi_cands  = ["FFI", "CMFF"]

    # Pick curves
    gr   = first_present(z.columns, gr_cands)
    cgr  = first_present(z.columns, cgr_cands)

    cali = first_present(z.columns, cali_cands)
    bs   = first_present(z.columns, bs_cands)
    dcal = first_present(z.columns, dcal_cands)

    rhob = first_present(z.columns, rhob_cands)
    drho = first_present(z.columns, drho_cands)
    tnph = first_present(z.columns, tnph_cands)
    tcmr = first_present(z.columns, tcmr_cands)
    cmrp = first_present(z.columns, cmrp_cands)

    rxo  = first_present(z.columns, rxo_cands)
    at90 = first_present(z.columns, at90_cands)
    at60 = first_present(z.columns, at60_cands)
    at30 = first_present(z.columns, at30_cands)
    at20 = first_present(z.columns, at20_cands)
    at10 = first_present(z.columns, at10_cands)

    cbw_name  = first_present(z.columns, cbw_cands)
    bvie_name = first_present(z.columns, bvie_cands)
    ffi_name  = first_present(z.columns, ffi_cands)

    # -----------------------------
    # Create 4 tracks
    # -----------------------------
    fig, ax = plt.subplots(
        nrows=1, ncols=4, figsize=(18, 20), sharey=True,
        gridspec_kw={"width_ratios": [1.8, 1.9, 1.7, 1.7]}
    )

    if title is None:
        title = f"Zone Template Plot — {top_depth:g}–{bottom_depth:g} ft"
    fig.suptitle(title, fontsize=18, color="blue", y=0.995)
    fig.subplots_adjust(top=0.80, wspace=0.24)

    # Common Y formatting
    for a in ax:
        a.set_ylim(top_depth, bottom_depth)
        a.invert_yaxis()
        a.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

        if tops_depths is not None:
            for d in tops_depths:
                if top_depth <= d <= bottom_depth:
                    a.axhline(d, color="red", linewidth=2.0)

    # Top labels in Track 1
    if (tops_depths is not None) and (tops is not None):
        for d, nm in zip(tops_depths, tops):
            if top_depth <= d <= bottom_depth:
                ax[0].text(
                    0.985, d, str(nm),
                    transform=ax[0].get_yaxis_transform(),
                    ha="right", va="center",
                    color="black", fontsize=20,
                    zorder=50,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.65, pad=1.5)
                )

    # ==========================================================
    # Track 1: GR/CGR fills + Hole condition
    # ==========================================================
    ax[0].set_title("GR/CGR + Hole", fontsize=14, color="green")
    ax[0].set_xlim(0.0, 200.0)
    ax[0].grid(True, axis="x", linestyle="--", linewidth=0.7, alpha=0.6)

    a_fill_gr = ax[0].twiny()
    a_fill_gr.set_xlim(0.0, 200.0)
    a_fill_gr.spines["top"].set_visible(False)
    a_fill_gr.get_xaxis().set_visible(False)

    gr_vals  = pd.to_numeric(z[gr], errors="coerce").to_numpy() if gr else None
    cgr_vals = pd.to_numeric(z[cgr], errors="coerce").to_numpy() if cgr else None

    if cgr_vals is not None:
        a_fill_gr.fill_betweenx(depth, 0.0, cgr_vals, where=np.isfinite(cgr_vals), alpha=1.0)

    if (gr_vals is not None) and (cgr_vals is not None):
        a_fill_gr.fill_betweenx(
            depth, cgr_vals, gr_vals,
            where=np.isfinite(gr_vals) & np.isfinite(cgr_vals) & (gr_vals >= cgr_vals),
            alpha=1.0
        )

    if (gr_vals is not None) and (cgr_vals is None):
        a_fill_gr.fill_betweenx(depth, 0.0, gr_vals, where=np.isfinite(gr_vals), alpha=0.25)

    if gr:
        a = ax[0].twiny()
        a.set_xlim(0.0, 200.0)
        a.plot(gr_vals, depth, linewidth=2.5)
        a.set_xlabel(nice_label(gr, units_map))
        a.spines["top"].set_position(("outward", 0))

    if cgr:
        a = ax[0].twiny()
        a.set_xlim(0.0, 200.0)
        a.plot(cgr_vals, depth, linewidth=2.5)
        a.set_xlabel(nice_label(cgr, units_map))
        a.spines["top"].set_position(("outward", 35))

    # hole cond
    cali_min, cali_max = 5.0, 25.0
    dcal_min, dcal_max = -2.0, 2.0

    if cali:
        a = ax[0].twiny()
        a.set_xlim(cali_min, cali_max)
        a.plot(pd.to_numeric(z[cali], errors="coerce").to_numpy(), depth, linewidth=1.6)
        a.set_xlabel(nice_label(cali, units_map))
        a.spines["top"].set_position(("outward", 80))

    if bs:
        a = ax[0].twiny()
        a.set_xlim(cali_min, cali_max)
        a.plot(pd.to_numeric(z[bs], errors="coerce").to_numpy(), depth, linewidth=1.1, linestyle="--")
        a.set_xlabel(nice_label(bs, units_map))
        a.spines["top"].set_position(("outward", 115))

    if cali and bs:
        cali_h = pd.to_numeric(z[cali], errors="coerce").to_numpy()
        bs_h   = pd.to_numeric(z[bs], errors="coerce").to_numpy()
        a_fill_h = ax[0].twiny()
        a_fill_h.set_xlim(cali_min, cali_max)
        a_fill_h.spines["top"].set_visible(False)
        a_fill_h.get_xaxis().set_visible(False)
        a_fill_h.fill_betweenx(
            depth, bs_h, cali_h,
            where=np.isfinite(cali_h) & np.isfinite(bs_h) & (cali_h > bs_h),
            alpha=0.25
        )

    if dcal:
        DC = pd.to_numeric(z[dcal], errors="coerce").to_numpy()

        a_fill_d = ax[0].twiny()
        a_fill_d.set_xlim(dcal_min, dcal_max)
        a_fill_d.spines["top"].set_visible(False)
        a_fill_d.get_xaxis().set_visible(False)

        a_fill_d.fill_betweenx(depth, 0.0, np.clip(DC, 0.0, None),
                               where=np.isfinite(DC) & (DC > 0), alpha=0.45)
        a_fill_d.fill_betweenx(depth, np.clip(DC, None, 0.0), 0.0,
                               where=np.isfinite(DC) & (DC < 0), alpha=0.45)

        a = ax[0].twiny()
        a.set_xlim(dcal_min, dcal_max)
        a.plot(DC, depth, linewidth=1.4, linestyle="--")
        a.set_xlabel(nice_label(dcal, units_map))
        a.spines["top"].set_position(("outward", 150))

    # ==========================================================
    # Track 2: RHOB / TNPH / NMR / DRHO
    # ==========================================================
    ax[1].set_title("RHOB / TNPH / NMR / DRHO", fontsize=14, color="blue")

    def setup_porosity_axis(a):
        ticks = [-0.15, 0.00, 0.15, 0.30, 0.45]
        a.set_xlim(-0.15, 0.45)
        a.invert_xaxis()
        a.set_xticks(ticks)
        a.grid(True, axis="x", linestyle="--", linewidth=0.7, alpha=0.8)

    overlay_done = False

    if tnph:
        a = ax[1].twiny()
        setup_porosity_axis(a)
        a.plot(pd.to_numeric(z[tnph], errors="coerce").to_numpy(), depth, linewidth=1.8)
        a.set_xlabel(nice_label(tnph, units_map))
        a.spines["top"].set_position(("outward", 0))
        overlay_done = True

    if tcmr:
        a = ax[1].twiny()
        setup_porosity_axis(a)
        a.plot(pd.to_numeric(z[tcmr], errors="coerce").to_numpy(), depth, linewidth=0.9)
        a.set_xlabel(nice_label(tcmr, units_map))
        a.spines["top"].set_position(("outward", 35))
        overlay_done = True

    if cmrp:
        a = ax[1].twiny()
        setup_porosity_axis(a)
        a.plot(pd.to_numeric(z[cmrp], errors="coerce").to_numpy(), depth, linewidth=0.9)
        a.set_xlabel(nice_label(cmrp, units_map))
        a.spines["top"].set_position(("outward", 70))
        overlay_done = True

    if rhob:
        a = ax[1].twiny()
        a.set_xlim(1.95, 2.95)
        a.plot(pd.to_numeric(z[rhob], errors="coerce").to_numpy(), depth, linewidth=1.8)
        a.set_xlabel(nice_label(rhob, units_map))
        a.spines["top"].set_position(("outward", 105))
        overlay_done = True

    if drho:
        a = ax[1].twiny()
        a.set_xlim(-0.15, 0.15)
        a.plot(pd.to_numeric(z[drho], errors="coerce").to_numpy(), depth, linewidth=0.7, linestyle="--")
        a.set_xlabel(nice_label(drho, units_map))
        a.spines["top"].set_position(("outward", 140))
        overlay_done = True

    if not overlay_done:
        ax[1].text(0.5, 0.5, "No RHOB / TNPH / NMR / DRHO", transform=ax[1].transAxes, ha="center", va="center")

    # ==========================================================
    # Track 3: Resistivity (log)
    # ==========================================================
    ax[2].set_title("Resistivity (log)", fontsize=14, color="blue")
    ax[2].set_xscale("log")
    ax[2].set_xlim(0.2, 2000)

    ax[2].xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=10))
    ax[2].xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax[2].xaxis.set_minor_formatter(NullFormatter())

    ax[2].grid(which="major", axis="x", linestyle="-", linewidth=2.2, alpha=1.0)
    ax[2].grid(which="minor", axis="x", linestyle=":", linewidth=0.7, alpha=0.9)

    res_curves = [
        (rxo,  "-",  1.0,   0),
        (at10, "--", 1.0,  35),
        (at20, "-.", 1.0,  70),
        (at30, ":",  1.0, 105),
        (at60, (0, (5, 2)), 1.0, 140),
        (at90, "--", 2.8, 175),
    ]

    any_res = False
    for c, ls, lw, off in res_curves:
        if c:
            a = ax[2].twiny()
            a.set_xscale("log")
            a.set_xlim(0.2, 2000)
            a.plot(pd.to_numeric(z[c], errors="coerce").to_numpy(), depth, linestyle=ls, linewidth=lw)
            a.set_xlabel(nice_label(c, units_map))
            a.spines["top"].set_position(("outward", off))
            any_res = True

    if not any_res:
        ax[2].text(0.5, 0.5, "No resistivity curves", transform=ax[2].transAxes, ha="center", va="center")

    # ==========================================================
    # Track 4: NMR Partition
    # ==========================================================
    ax[3].set_title("NMR Partition", fontsize=14, color="blue")
    ax[3].set_xlim(0.0, 0.30)
    ax[3].invert_xaxis()
    ax[3].grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.6)

    phit = pd.to_numeric(z[tcmr], errors="coerce").to_numpy() if tcmr else None
    phie = pd.to_numeric(z[cmrp], errors="coerce").to_numpy() if cmrp else None
    bvie = pd.to_numeric(z[bvie_name], errors="coerce").to_numpy() if bvie_name else None

    if bvie is None and (phie is not None) and ffi_name:
        ff = pd.to_numeric(z[ffi_name], errors="coerce").to_numpy()
        ff_clip = np.where(np.isfinite(ff) & np.isfinite(phie), np.clip(ff, 0.0, phie), np.nan)
        bvie = np.where(np.isfinite(phie) & np.isfinite(ff_clip), np.maximum(phie - ff_clip, 0.0), np.nan)

    a_fill = ax[3].twiny()
    a_fill.set_xlim(0.0, 0.30)
    a_fill.invert_xaxis()
    a_fill.spines["top"].set_visible(False)
    a_fill.get_xaxis().set_visible(False)

    any_nmr = False

    if bvie is not None:
        a_fill.fill_betweenx(depth, 0.0, bvie, where=np.isfinite(bvie), alpha=1.0)
        any_nmr = True

    if (phie is not None) and (bvie is not None):
        a_fill.fill_betweenx(depth, bvie, phie,
                             where=np.isfinite(phie) & np.isfinite(bvie) & (phie >= bvie),
                             alpha=1.0)
        any_nmr = True

    if (phit is not None) and (phie is not None):
        a_fill.fill_betweenx(depth, phie, phit,
                             where=np.isfinite(phit) & np.isfinite(phie) & (phit >= phie),
                             alpha=0.65)
        any_nmr = True

    for curve_name, off, lw, ls in [
        (tcmr,      0,   0.9, "-"),
        (cmrp,     35,   0.9, "-"),
        (bvie_name, 70,  0.9, "-."),
    ]:
        if curve_name and curve_name in z.columns:
            a = ax[3].twiny()
            a.set_xlim(0.0, 0.30)
            a.invert_xaxis()
            a.plot(pd.to_numeric(z[curve_name], errors="coerce").to_numpy(), depth, linewidth=lw, linestyle=ls)
            a.set_xlabel(nice_label(curve_name, units_map))
            a.spines["top"].set_position(("outward", off))

    if not any_nmr:
        ax[3].text(0.5, 0.5, "No NMR curves", transform=ax[3].transAxes, ha="center", va="center")

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.close(fig)
    return fig
