# petrocore/workflows/vsh_qc_plot.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec


@dataclass(frozen=True)
class VshQcPlotCols:
    dept: str = "DEPT"
    gr: Optional[str] = None     # GR or CGR
    dtco: Optional[str] = None
    rhob: Optional[str] = None
    tnph: Optional[str] = None
    cmrp: Optional[str] = None
    tcmr: Optional[str] = None
    bvie: Optional[str] = None

    vsh_gr: str = "VSH_GR"
    vsh_nd: str = "VSH_ND"
    vsh_dtden: str = "VSH_DTDEN"
    vsh_cmr: str = "VSH_CMR"
    vsh_hl: str = "VSH_HL"


def shale_qc_plot(
    analysis_df: pd.DataFrame,
    cols: VshQcPlotCols,
    *,
    gr_clean: float,
    gr_shale: float,
    dt_matrix: float,
    dt_shale: float,
    neut_shale: float,
    den_matrix: float = 2.65,
    den_shale: float = 2.65,
    den_fl: float = 1.10,
    dt_fl: float = 140.0,
    neut_matrix: float = -0.04,
    neut_fl: float = 1.00,
    mphi_shale: float = 0.00,
    title: str = "Shale Parameter Plot: Vsh Methods + Endpoints",
) -> plt.Figure:

    df = analysis_df
    dept = pd.to_numeric(df[cols.dept], errors="coerce").to_numpy(dtype=float)

    def a(col):
        if not col or col not in df.columns:
            return None
        return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)

    gr   = a(cols.gr)
    dtco = a(cols.dtco)
    rhob = a(cols.rhob)
    tnph = a(cols.tnph)
    cmrp = a(cols.cmrp)
    tcmr = a(cols.tcmr)
    bvie = a(cols.bvie)

    vsh_gr  = a(cols.vsh_gr)
    vsh_nd  = a(cols.vsh_nd)
    vsh_dt  = a(cols.vsh_dtden)
    vsh_cmr = a(cols.vsh_cmr)
    vsh_hl  = a(cols.vsh_hl)

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(title, color="blue", fontsize=18)

    gs = gridspec.GridSpec(4, 4)
    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[1, 1])
    ax3 = fig.add_subplot(gs[0, 1])
    ax4 = fig.add_subplot(gs[2, 1])
    ax5 = fig.add_subplot(gs[3, 1])
    ax6 = fig.add_subplot(gs[:, 2], sharey=ax1)
    ax7 = fig.add_subplot(gs[:, 3], sharey=ax1)

    # ---- Track GR/DT ----
    ax1.grid(True)
    ax1.set_ylabel("DEPTH")
    ax1.invert_yaxis()

    if gr is not None:
        ax1.plot(gr, dept, color="green", lw=3.0, label=(cols.gr or "GR"))
        ax1.axvline(gr_clean, color="lime", lw=3.0, label="gr_clean")
        ax1.axvline(gr_shale, color="olive", lw=3.0, ls="-.", label="gr_shale")
        ax1.set_xlim(0, 200)
        ax1.set_title("GR/CGR + DTCO", color="blue")

    if dtco is not None:
        ax1.plot(dtco, dept, color="blue", lw=3.0, ls="--", label=(cols.dtco or "DTCO"))
        ax1.axvline(dt_matrix, color="cyan", lw=3.0, label="dt_matrix")
        ax1.axvline(dt_shale, color="dodgerblue", lw=3.0, label="dt_shale")

    ax1.legend(loc="best")

    # ---- GR histogram ----
    if gr is not None:
        ax2.hist(gr[np.isfinite(gr)], bins=100, color="green")
        ax2.axvline(gr_clean, color="blue")
        ax2.axvline(gr_shale, color="brown", ls="-.")
        ax2.axvspan(gr_clean, gr_shale, alpha=0.1, color="yellow")
        ax2.set_xlim(0, 200)
        ax2.set_title("GR/CGR Histogram", color="green")
        ax2.grid(True)

    # ---- DTCO-RHOB xplot ----
    if (dtco is not None) and (rhob is not None):
        ax3.plot(dtco, rhob, "ro", ms=3)
        ax3.set_title("DTCO - RHOB Xplot", color="blue")
        ax3.set_xlim(40, 140)
        ax3.set_ylim(3, 1.0)
        ax3.set_xlabel("DTCO")
        ax3.set_ylabel("RHOB")
        ax3.grid(True)

        # triangle sketch (matches your intent)
        ax3.plot([dt_matrix, dt_fl], [den_matrix, den_fl], "b-o")
        ax3.plot([dt_matrix, dt_shale], [den_matrix, den_shale], "b-o")
        ax3.plot([dt_shale, dt_fl], [den_shale, den_fl], "b-o")

    # ---- NPHI-CMRP xplot ----
    if (tnph is not None) and (cmrp is not None):
        ax4.plot(tnph, cmrp, "ro", ms=3)
        ax4.set_title("NPHI - CMRP Xplot", color="blue")
        ax4.set_xlim(-0.05, 1.0)
        ax4.set_ylim(0, 1.0)
        ax4.set_ylabel("CMRP")
        ax4.grid(True)

    # ---- NPHI-RHOB xplot ----
    if (tnph is not None) and (rhob is not None):
        ax5.plot(tnph, rhob, "ro", ms=3)
        ax5.set_title("NPHI - RHOB Xplot", color="blue")
        ax5.set_xlim(-0.05, 1.0)
        ax5.set_ylim(3, 1.0)
        ax5.set_xlabel("NPHI")
        ax5.set_ylabel("RHOB")
        ax5.grid(True)

        ax5.plot([neut_matrix, neut_fl], [den_matrix, den_fl], "b-o")
        ax5.plot([neut_matrix, neut_shale], [den_matrix, den_shale], "b-o")
        ax5.plot([neut_shale, neut_fl], [den_shale, den_fl], "b-o")

    # ---- Vsh track ----
    ax6.grid(True)
    ax6.set_xlim(0, 1)
    ax6.set_title("Calculated Shale Volumes", color="blue")
    if vsh_gr is not None:  ax6.plot(vsh_gr, dept, color="green", lw=2, label="VSH_GR")
    if vsh_dt is not None:  ax6.plot(vsh_dt, dept, color="blue",  lw=2, label="VSH_DTDEN")
    if vsh_nd is not None:  ax6.plot(vsh_nd, dept, color="red",   lw=2, label="VSH_ND")
    if vsh_cmr is not None: ax6.plot(vsh_cmr, dept, color="purple",lw=2, label="VSH_CMR")
    if vsh_hl is not None:  ax6.plot(vsh_hl, dept, color="black", lw=3, label="VSH_HL")
    ax6.legend(loc="best", fontsize=10)

    # ---- NMR track (optional) ----
    ax7.grid(True)
    ax7.invert_yaxis()
    ax7.set_xlim(0.30, 0.0)
    ax7.set_title("NMR Log", color="blue")
    if tcmr is not None: ax7.plot(tcmr, dept, "k-", lw=1, label="TCMR")
    if cmrp is not None: ax7.plot(cmrp, dept, "brown", lw=1, label="CMRP")
    if bvie is not None: ax7.plot(bvie, dept, "navy", lw=1, label="BVIE")
    ax7.legend(loc="best", fontsize=10)

    fig.tight_layout()
    plt.close(fig)
    return fig
