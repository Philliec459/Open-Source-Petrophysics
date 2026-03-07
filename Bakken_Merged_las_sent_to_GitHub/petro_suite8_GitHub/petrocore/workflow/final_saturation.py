# petrocore/workflows/final_saturation.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from petrocore.workflows.pickett_mstar_panel import (
    PickettConfig,
    compute_bvw_bvo_swt_mstar,
)
from petrocore.workflows.waxman_smits import (
    WaxmanSmitsConfig,
    compute_waxman_smits,
)


# -------------------------------------------------
# Config
# -------------------------------------------------
@dataclass(frozen=True)
class FinalSaturationConfig:
    dept_col: str = "DEPT"

    phit_col: str = "PHIT"
    vsh_col: str = "VSH_HL"
    qv_col: str = "Qv"
    cbw_col: str = "CBWapp"
    rt_col: str = "RT"   # family winner

    # outputs
    sw_ws: str = "SW_WS"
    bvwt_ws: str = "BVWT_WS"
    bvwe_ws: str = "BVWe_WS"


# -------------------------------------------------
# Master compute
# -------------------------------------------------
def compute_final_saturation(
    *,
    analysis_df: pd.DataFrame,
    cfg: FinalSaturationConfig,
    # Pickett / M*
    m: float,
    n: float,
    rw: float,
    mslope: float,
    B: float,
    # Waxman–Smits
    rw_ws: Optional[float] = None,
    m_ws: Optional[float] = None,
    n_ws: Optional[float] = None,
    B_ws: Optional[float] = None,
    logger: Optional[Callable[[str], None]] = None,
) -> dict:
    """
    Runs:
      1) Pickett BVW/BVO/Swt/MSTARAPP/MSTAR
      2) Waxman–Smits Sw, BVWT, BVWe

    Returns updated analysis_df.
    """

    df = analysis_df

    # -------------------------
    # 1) Pickett + M*
    # -------------------------
    p_cfg = PickettConfig(
        dept_col=cfg.dept_col,
        phit_col=cfg.phit_col,
        vsh_col=cfg.vsh_col,
        qv_col=cfg.qv_col,
        rt_col=cfg.rt_col,
    )

    res1 = compute_bvw_bvo_swt_mstar(
        analysis_df=df,
        cfg=p_cfg,
        m=m,
        n=n,
        rw=rw,
        mslope=mslope,
        B=B,
        logger=logger,
    )
    df = res1["analysis_df"]

    # -------------------------
    # 2) Waxman–Smits Sw
    # -------------------------
    ws_cfg = WaxmanSmitsConfig(
        rt_col=cfg.rt_col,
        phit_col=cfg.phit_col,
        qv_col=cfg.qv_col,
        cbw_col=cfg.cbw_col,
        out_sw=cfg.sw_ws,
        out_bvwt=cfg.bvwt_ws,
        out_bvwe=cfg.bvwe_ws,
    )

    res2 = compute_waxman_smits(
        analysis_df=df,
        cfg=ws_cfg,
        rw=rw_ws if rw_ws is not None else rw,
        m=m_ws if m_ws is not None else m,
        n=n_ws if n_ws is not None else n,
        B=B_ws if B_ws is not None else B,
        logger=logger,
    )
    df = res2["analysis_df"]

    if logger:
        logger("=== FINAL SATURATION COMPLETE ===")

    return {"analysis_df": df}


# -------------------------------------------------
# Final Results Plot
# -------------------------------------------------
def plot_final_saturation(
    *,
    analysis_df: pd.DataFrame,
    cfg: FinalSaturationConfig,
) -> plt.Figure:
    """
    Tracks:
      1) PHIT, BVW, BVWT_WS vs depth
      2) Sw Pickett vs Sw Waxman–Smits
      3) Vsh vs MSTAR
      4) Pickett crossplot
    """

    df = analysis_df

    y = pd.to_numeric(df[cfg.dept_col], errors="coerce").to_numpy()

    phit = pd.to_numeric(df[cfg.phit_col], errors="coerce").to_numpy()
    bvw = pd.to_numeric(df.get("BVW", np.nan), errors="coerce").to_numpy()
    bvwt = pd.to_numeric(df[cfg.bvwt_ws], errors="coerce").to_numpy()

    sw_pickett = pd.to_numeric(df.get("Swt", np.nan), errors="coerce").to_numpy()
    sw_ws = pd.to_numeric(df[cfg.sw_ws], errors="coerce").to_numpy()

    vsh = pd.to_numeric(df[cfg.vsh_col], errors="coerce").to_numpy()
    mstar = pd.to_numeric(df.get("MSTAR", np.nan), errors="coerce").to_numpy()

    rt = pd.to_numeric(df[cfg.rt_col], errors="coerce").to_numpy()

    fig, ax = plt.subplots(1, 4, figsize=(20, 8), sharey=True)
    fig.suptitle("Final Saturation Results", fontsize=16, color="blue")

    # ---- Track 1: Volumes ----
    ax[0].plot(phit, y, "r-", lw=1, label="PHIT")
    ax[0].plot(bvw, y, "k-", lw=1, label="BVW (Pickett)")
    ax[0].plot(bvwt, y, "b-", lw=1, label="BVWT (WS)")
    ax[0].set_xlim(0.5, 0.0)
    ax[0].set_ylim(np.nanmax(y), np.nanmin(y))
    ax[0].set_title("Bulk Volumes")
    ax[0].grid(True)
    ax[0].legend()

    # ---- Track 2: Sw ----
    ax[1].plot(sw_pickett, y, "g.", label="Sw Pickett")
    ax[1].plot(sw_ws, y, "r.", label="Sw WS")
    ax[1].set_xlim(1.0, 0.0)
    ax[1].set_title("Water Saturation")
    ax[1].grid(True)
    ax[1].legend()

    # ---- Track 3: MSTAR ----
    ax[2].plot(vsh, mstar, "k.")
    ax[2].set_xlim(0, 1)
    ax[2].set_ylim(0, np.nanmax(mstar) * 1.1)
    ax[2].set_xlabel("Vsh")
    ax[2].set_ylabel("MSTAR")
    ax[2].set_title("Vsh vs MSTAR")
    ax[2].grid(True)

    # ---- Track 4: Pickett ----
    ax[3].loglog(rt, phit, "ro")
    ax[3].set_xlim(0.01, 1000)
    ax[3].set_ylim(0.01, 1)
    ax[3].set_xlabel("Rt")
    ax[3].set_ylabel("PHIT")
    ax[3].set_title("Pickett")
    ax[3].grid(True, which="both")

    plt.tight_layout()
    plt.close(fig)
    return fig
