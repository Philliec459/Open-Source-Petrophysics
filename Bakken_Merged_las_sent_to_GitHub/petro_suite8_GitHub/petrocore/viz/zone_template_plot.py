from __future__ import annotations



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter








# -----------------------------
# Helpers
# -----------------------------
def nice_label(mnemonic, units_map):
    u = (units_map.get(mnemonic, "") or "").strip() if isinstance(units_map, dict) else ""
    return f"{mnemonic}\n[{u}]" if u else mnemonic

def first_present(cols, candidates):
    s = set(cols)
    for c in candidates:
        if c in s:
            return c
    return None

def add_tops(ax, top_depth, bottom_depth):
    if "tops_depths" in globals() and "tops" in globals():
        for d in tops_depths:
            if top_depth <= d <= bottom_depth:
                ax.axhline(d, color="red", linewidth=1.0)

def add_tops_labels(ax, top_depth, bottom_depth):
    """
    Draw top names in the GR track, right-justified.
    Uses x in axes fraction (0..1) and y in data (depth).
    """
    if "tops_depths" in globals() and "tops" in globals():
        for d, nm in zip(tops_depths, tops):
            if top_depth <= d <= bottom_depth:
                ax.text(
                    0.985, d, str(nm),
                    transform=ax.get_yaxis_transform(),
                    ha="right", va="center",
                    color="black", fontsize=20,
                    zorder=50,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.65, pad=1.5)
                )

def setup_porosity_axis(a):
    """
    Standard porosity axis:
    - Range: 0.45 → -0.15 (inverted)
    - Gridlines at [-0.15, 0.00, 0.15, 0.30, 0.45]
    """
    ticks = [-0.15, 0.00, 0.15, 0.30, 0.45]
    a.set_xlim(-0.15, 0.45)
    a.invert_xaxis()
    a.set_xticks(ticks)
    a.grid(True, axis="x", linestyle="--", linewidth=0.7, alpha=0.8)

# -----------------------------
# Plot function (FROM DF with DEPT column)
# -----------------------------
def plot_zone_template_from_df(df_in, units_map, top_depth, bottom_depth, title=None, depth_col="DEPT"):
    if df_in is None or df_in.empty:
        raise ValueError("Input dataframe is empty.")
    if depth_col not in df_in.columns:
        raise ValueError(f"Dataframe must contain a '{depth_col}' column.")

    # Copy, coerce depth to numeric, sort, slice
    df = df_in.copy()
    df[depth_col] = pd.to_numeric(df[depth_col], errors="coerce")
    df = df.dropna(subset=[depth_col]).sort_values(depth_col)

    z = df.loc[(df[depth_col] >= top_depth) & (df[depth_col] <= bottom_depth)].copy()
    if z.empty:
        raise ValueError(f"No data in zone {top_depth}–{bottom_depth} (based on {depth_col}).")

    depth = z[depth_col].to_numpy()

    # -----------------------------
    # Candidate lists
    # -----------------------------
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

    cbw_cands  = ["CBW"]
    bvie_cands = ["BVIE","BVI_E"]
    ffi_cands  = ["FFI", "CMFF"]
    bfv_cands  = ["BFV","BFV_3MS","BFV3MS","MBVI"]

    phit_cands = ["PHIT", "PHIT_RMS", "PHIT_NMR", "TCMR"]
    sw_cands   = ["SW_IT", "SW_CP", "SW_ARCHIE"]
    bvw_cands  = ["BVWe_CP", "BVWe_IT"]
    phie_cands = ["PHIE", "PHIE_NMR", "CMRP_3MS", "CMRP", "MPHI"]

    # -----------------------------
    # Pick curves
    # -----------------------------
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

    cbw  = first_present(z.columns, cbw_cands)
    bvie = first_present(z.columns, bvie_cands)
    ffi  = first_present(z.columns, ffi_cands)
    bfv  = first_present(z.columns, bfv_cands)

    phit_name = first_present(z.columns, phit_cands)
    sw_name   = first_present(z.columns, sw_cands)
    bvw_name  = first_present(z.columns, bvw_cands)
    phie_name = first_present(z.columns, phie_cands)

    # -----------------------------
    # Create 5 tracks
    # -----------------------------
    fig, ax = plt.subplots(
        nrows=1, ncols=5, figsize=(21, 20), sharey=True,
        gridspec_kw={"width_ratios": [1.8, 1.9, 1.7, 1.7, 1.4]}
    )

    if title is None:
        title = f"Zone Template Plot — {top_depth:g}–{bottom_depth:g} ft"
    fig.suptitle(title, fontsize=18, color="blue", y=0.995)
    fig.subplots_adjust(top=0.88, wspace=0.24)

    # Common Y formatting
    for a in ax:
        a.set_ylim(top_depth, bottom_depth)
        a.invert_yaxis()
        a.yaxis.grid(True, linestyle="--", linewidth=0.5)
        a.get_xaxis().set_visible(True)
        add_tops(a, top_depth, bottom_depth)
    add_tops_labels(ax[0], top_depth, bottom_depth)

    # ==========================================================
    # Track 1: GR/CGR + Hole (CALI/BS/DCAL)
    # ==========================================================
    ax[0].set_title("GR/CGR + Hole", fontsize=14, color="green")

    x_left_gr, x_right_gr = 0.0, 200.0
    ax[0].set_xlim(x_left_gr, x_right_gr)
    ax[0].grid(True, axis="x", linestyle="--", linewidth=0.7, alpha=0.6)

    a_fill_gr = ax[0].twiny()
    a_fill_gr.set_xlim(x_left_gr, x_right_gr)
    a_fill_gr.spines["top"].set_visible(False)
    a_fill_gr.get_xaxis().set_visible(False)
    a_fill_gr.grid(False)

    gr_vals  = z[gr].astype(float).to_numpy() if gr else None
    cgr_vals = z[cgr].astype(float).to_numpy() if cgr else None

    if cgr_vals is not None:
        a_fill_gr.fill_betweenx(depth, 0.0, cgr_vals, where=np.isfinite(cgr_vals),
                                facecolor="green", alpha=1.0)

    if (gr_vals is not None) and (cgr_vals is not None):
        a_fill_gr.fill_betweenx(
            depth, cgr_vals, gr_vals,
            where=np.isfinite(gr_vals) & np.isfinite(cgr_vals) & (gr_vals >= cgr_vals),
            facecolor="magenta", alpha=1.0
        )

    if (gr_vals is not None) and (cgr_vals is None):
        a_fill_gr.fill_betweenx(depth, 0.0, gr_vals, where=np.isfinite(gr_vals),
                                facecolor="green", alpha=0.25)

    if gr:
        a = ax[0].twiny()
        a.set_xlim(x_left_gr, x_right_gr)
        a.plot(gr_vals, depth, color="green", linewidth=2.5)
        a.set_xlabel(nice_label(gr, units_map), color="green")
        a.tick_params(axis="x", colors="green")
        a.spines["top"].set_position(("outward", 0))
        a.grid(False)

    if cgr:
        a = ax[0].twiny()
        a.set_xlim(x_left_gr, x_right_gr)
        a.plot(cgr_vals, depth, color="magenta", linewidth=2.5)
        a.set_xlabel(nice_label(cgr, units_map), color="magenta")
        a.tick_params(axis="x", colors="magenta")
        a.spines["top"].set_position(("outward", 35))
        a.grid(False)

    # Hole overlay scales (your latest preference)
    cali_min, cali_max = 5.0, 25.0
    dcal_min, dcal_max = -2.0, 2.0

    if cali:
        a = ax[0].twiny()
        a.set_xlim(cali_min, cali_max)
        a.plot(z[cali].astype(float).to_numpy(), depth, color="brown", linewidth=2.0)
        a.set_xlabel(nice_label(cali, units_map), color="brown")
        a.tick_params(axis="x", colors="brown")
        a.spines["top"].set_position(("outward", 80))
        a.grid(False)

    if bs:
        a = ax[0].twiny()
        a.set_xlim(cali_min, cali_max)
        a.plot(z[bs].astype(float).to_numpy(), depth, color="black", linewidth=1.0, linestyle="--")
        a.set_xlabel(nice_label(bs, units_map), color="black")
        a.tick_params(axis="x", colors="black")
        a.spines["top"].set_position(("outward", 115))
        a.grid(False)

    if cali and bs:
        cali_vals_h = z[cali].astype(float).to_numpy()
        bs_vals_h   = z[bs].astype(float).to_numpy()

        a_fill_h = ax[0].twiny()
        a_fill_h.set_xlim(cali_min, cali_max)
        a_fill_h.spines["top"].set_visible(False)
        a_fill_h.get_xaxis().set_visible(False)
        a_fill_h.grid(False)

        a_fill_h.fill_betweenx(
            depth, bs_vals_h, cali_vals_h,
            where=np.isfinite(cali_vals_h) & np.isfinite(bs_vals_h) & (cali_vals_h > bs_vals_h),
            facecolor="yellow", alpha=0.25
        )

    if dcal:
        DC = z[dcal].astype(float).to_numpy()

        a_fill_d = ax[0].twiny()
        a_fill_d.set_xlim(dcal_min, dcal_max)
        a_fill_d.spines["top"].set_visible(False)
        a_fill_d.get_xaxis().set_visible(False)
        a_fill_d.grid(False)

        a_fill_d.fill_betweenx(
            depth, 0.0, np.clip(DC, 0.0, None),
            where=np.isfinite(DC) & (DC > 0),
            facecolor="yellow", alpha=0.45
        )
        a_fill_d.fill_betweenx(
            depth, np.clip(DC, None, 0.0), 0.0,
            where=np.isfinite(DC) & (DC < 0),
            facecolor="saddlebrown", alpha=0.45
        )

        a = ax[0].twiny()
        a.set_xlim(dcal_min, dcal_max)
        a.plot(DC, depth, color="purple", linewidth=4.0, linestyle="--")
        a.set_xlabel(nice_label(dcal, units_map), color="purple")
        a.tick_params(axis="x", colors="purple")
        a.spines["top"].set_position(("outward", 150))
        a.grid(False)

    # ==========================================================
    # Track 2: RHOB / TNPH / NMR / DRHO
    # ==========================================================
    ax[1].set_title("RHOB / TNPH / NMR / DRHO", fontsize=14, color="blue")
    overlay_done = False

    if tnph:
        a = ax[1].twiny()
        setup_porosity_axis(a)
        a.plot(z[tnph].to_numpy(), depth, color="green", linewidth=2.0)
        a.set_xlabel(nice_label(tnph, units_map), color="green")
        a.tick_params(axis="x", colors="green")
        a.spines["top"].set_position(("outward", 0))
        overlay_done = True

    if tcmr:
        a = ax[1].twiny()
        setup_porosity_axis(a)
        a.plot(z[tcmr].to_numpy(), depth, color="black", linewidth=0.5)
        a.set_xlabel(nice_label(tcmr, units_map), color="black")
        a.tick_params(axis="x", colors="black")
        a.spines["top"].set_position(("outward", 35))
        overlay_done = True

    if cmrp:
        a = ax[1].twiny()
        setup_porosity_axis(a)
        a.plot(z[cmrp].to_numpy(), depth, color="brown", linewidth=0.5)
        a.set_xlabel(nice_label(cmrp, units_map), color="brown")
        a.tick_params(axis="x", colors="brown")
        a.spines["top"].set_position(("outward", 70))
        overlay_done = True

    if rhob:
        a = ax[1].twiny()
        a.set_xlim(1.95, 2.95)
        a.plot(z[rhob].to_numpy(), depth, color="red", linewidth=2)
        a.set_xlabel(nice_label(rhob, units_map), color="red")
        a.tick_params(axis="x", colors="red")
        a.spines["top"].set_position(("outward", 105))
        a.grid(False)
        overlay_done = True

    if drho:
        a = ax[1].twiny()
        a.set_xlim(-0.05, 0.25)
        a.plot(z[drho].to_numpy(), depth, color="purple", linewidth=3, linestyle="--")
        a.set_xlabel(nice_label(drho, units_map), color="purple")
        a.tick_params(axis="x", colors="purple")
        a.spines["top"].set_position(("outward", 140))
        a.grid(False)
        overlay_done = True

    if not overlay_done:
        ax[1].text(0.5, 0.5, "No RHOB / TNPH / NMR / DRHO",
                   transform=ax[1].transAxes, ha="center", va="center")

    # ==========================================================
    # Track 3: Resistivity (log)
    # ==========================================================
    ax[2].set_title("Resistivity (log)", fontsize=14, color="blue")
    ax[2].set_xscale("log")
    ax[2].set_xlim(0.2, 2000)
    ax[2].xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=10))
    ax[2].xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax[2].xaxis.set_minor_formatter(NullFormatter())
    ax[2].grid(which="major", axis="x", linestyle="-", linewidth=1.8, alpha=1.0)
    ax[2].grid(which="minor", axis="x", linestyle=":", linewidth=0.7, alpha=0.9)

    res_curves = [
        ("RXO",  rxo,  "blue",   "-",         1.0,   0),
        ("AT10", at10, "green",  "--",        1.0,  35),
        ("AT20", at20, "orange", "-.",        1.0,  70),
        ("AT30", at30, "red",    ":",         1.0, 105),
        ("AT60", at60, "purple", (0, (5, 2)), 1.0, 140),
        ("AT90", at90, "black",  "--",        3.2, 175),
    ]

    any_res = False
    for _, c, color, ls, lw, off in res_curves:
        if c:
            a = ax[2].twiny()
            a.set_xscale("log")
            a.set_xlim(0.2, 2000)
            a.plot(z[c].to_numpy(), depth, color=color, linestyle=ls, linewidth=lw)
            a.set_xlabel(nice_label(c, units_map), color=color)
            a.tick_params(axis="x", colors=color)
            a.spines["top"].set_position(("outward", off))
            a.grid(False)
            any_res = True

    if not any_res:
        ax[2].text(0.5, 0.5, "No resistivity curves",
                   transform=ax[2].transAxes, ha="center", va="center")

    # ==========================================================
    # Track 4: NMR Partition
    # ==========================================================
    ax[3].set_title("NMR Partition", fontsize=14, color="blue")
    x_left, x_right = 0.0, 0.30
    ax[3].set_xlim(x_left, x_right)
    ax[3].invert_xaxis()
    ax[3].grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.6)

    any_nmr = False

    phit = z[tcmr].astype(float).to_numpy() if tcmr else None
    phie = z[cmrp].astype(float).to_numpy() if cmrp else None

    bvie_name = first_present(z.columns, ["BVI_E", "BVIE"])
    cmff_name = first_present(z.columns, ["FFI", "CMFF"])

    bvie_arr = z[bvie_name].astype(float).to_numpy() if bvie_name else None

    if bvie_arr is None and (phie is not None) and cmff_name:
        cmff = z[cmff_name].astype(float).to_numpy()
        cmff_clip = np.where(np.isfinite(cmff) & np.isfinite(phie), np.clip(cmff, 0.0, phie), np.nan)
        bvie_arr = np.where(np.isfinite(phie) & np.isfinite(cmff_clip), np.maximum(phie - cmff_clip, 0.0), np.nan)

    a_fill = ax[3].twiny()
    a_fill.set_xlim(x_left, x_right)
    a_fill.invert_xaxis()
    a_fill.spines["top"].set_visible(False)
    a_fill.get_xaxis().set_visible(False)
    a_fill.grid(False)

    if bvie_arr is not None:
        a_fill.fill_betweenx(depth, 0.0, bvie_arr, where=np.isfinite(bvie_arr),
                             facecolor="blue", alpha=1.0)
        any_nmr = True

    if (phie is not None) and (bvie_arr is not None):
        a_fill.fill_betweenx(depth, bvie_arr, phie,
                             where=np.isfinite(phie) & np.isfinite(bvie_arr) & (phie >= bvie_arr),
                             facecolor="yellow", alpha=1.0)
        any_nmr = True

    if (phit is not None) and (phie is not None):
        a_fill.fill_betweenx(depth, phie, phit,
                             where=np.isfinite(phit) & np.isfinite(phie) & (phit >= phie),
                             facecolor="gray", alpha=0.65)
        any_nmr = True

    for curve_name, color, off, lw, ls in [
        (tcmr,      "black",  0, 0.9, "-"),
        (cmrp,      "brown", 35, 0.9, "-"),
        (bvie_name, "navy",  70, 0.9, "-."),
    ]:
        if curve_name and curve_name in z.columns:
            a = ax[3].twiny()
            a.set_xlim(x_left, x_right)
            a.invert_xaxis()
            a.plot(z[curve_name].to_numpy(), depth, color=color, linewidth=lw, linestyle=ls)
            a.set_xlabel(nice_label(curve_name, units_map), color=color)
            a.tick_params(axis="x", colors=color)
            a.spines["top"].set_position(("outward", off))
            a.grid(False)
            any_nmr = True

    if not any_nmr:
        ax[3].text(0.5, 0.5, "No NMR curves",
                   transform=ax[3].transAxes, ha="center", va="center")

    # ==========================================================
    # Track 5: PHIT / BVW shading
    # ==========================================================
    ax[4].set_title("PHIT / BVW", fontsize=14, color="blue")
    x_left, x_right = 0.0, 0.3
    ax[4].set_xlim(x_left, x_right)
    ax[4].invert_xaxis()
    ax[4].grid(True, axis="x", linestyle="--", linewidth=0.6, alpha=0.7)

    any_bvw = False

    phit_arr = z[phit_name].astype(float).to_numpy() if phit_name else None

    bvw_arr = z[bvw_name].astype(float).to_numpy() if bvw_name else None
    if bvw_arr is None and (phit_arr is not None) and sw_name:
        sw = z[sw_name].astype(float).to_numpy()
        bvw_arr = phit_arr * sw

    phie_arr = z[phie_name].astype(float).to_numpy() if phie_name else None

    a_fill = ax[4].twiny()
    a_fill.set_xlim(x_left, x_right)
    a_fill.invert_xaxis()
    a_fill.spines["top"].set_visible(False)
    a_fill.get_xaxis().set_visible(False)
    a_fill.grid(False)

    if bvw_arr is not None:
        bvw_clip = np.clip(bvw_arr, 0.0, None)
        a_fill.fill_betweenx(depth, 0.0, bvw_clip, where=np.isfinite(bvw_clip),
                             facecolor="cyan", alpha=1)
        any_bvw = True

    if (phie_arr is not None) and (bvw_arr is not None):
        a_fill.fill_betweenx(depth, bvw_arr, phie_arr,
                             where=np.isfinite(phie_arr) & np.isfinite(bvw_arr) & (phie_arr >= bvw_arr),
                             facecolor="lime", alpha=1)
        any_bvw = True

    if (phit_arr is not None) and (phie_arr is not None):
        a_fill.fill_betweenx(depth, phie_arr, phit_arr,
                             where=np.isfinite(phit_arr) & np.isfinite(phie_arr) & (phit_arr >= phie_arr),
                             facecolor="gray", alpha=1)
        any_bvw = True

    if phit_arr is not None:
        a = ax[4].twiny()
        a.set_xlim(x_left, x_right)
        a.invert_xaxis()
        a.plot(phit_arr, depth, color="black", linewidth=1.2)
        a.set_xlabel(nice_label(phit_name, units_map), color="black")
        a.tick_params(axis="x", colors="black")
        a.spines["top"].set_position(("outward", 0))
        a.grid(False)
        any_bvw = True

    if bvw_arr is not None:
        a = ax[4].twiny()
        a.set_xlim(x_left, x_right)
        a.invert_xaxis()
        a.plot(bvw_arr, depth, color="blue", linewidth=2, linestyle="-")
        label = bvw_name if bvw_name else "BVWe=PHIT*Sw-CBW"
        a.set_xlabel(nice_label(label, units_map), color="blue")
        a.tick_params(axis="x", colors="blue")
        a.spines["top"].set_position(("outward", 70))
        a.grid(False)
        any_bvw = True

    if analysis_df['BVWe_CP'] is not None:
        a = ax[4].twiny()
        a.set_xlim(x_left, x_right)
        a.invert_xaxis()
        a.plot(analysis_df['BVWe_CP'], depth, color="magenta", linewidth=1.5, linestyle="-")
        label = bvw_name if bvw_name else "BVWe=PHIT*Sw-CBW"
        a.set_xlabel(nice_label(label, units_map), color="magenta")
        a.tick_params(axis="x", colors="magenta")
        a.spines["top"].set_position(("outward", 105))
        a.grid(False)
        any_bvw = True

    if phie_arr is not None:
        a = ax[4].twiny()
        a.set_xlim(x_left, x_right)
        a.invert_xaxis()
        a.plot(phie_arr, depth, color="orange", linewidth=1.2, linestyle="--")
        a.set_xlabel(nice_label("PHIE", units_map), color="orange")
        a.tick_params(axis="x", colors="orange")
        a.spines["top"].set_position(("outward", 35))
        a.grid(False)
        any_bvw = True

    if not any_bvw:
        ax[4].text(0.5, 0.5, "No PHIT/BVW/Sw",
                   transform=ax[4].transAxes, ha="center", va="center")

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    return fig



plt.show()
