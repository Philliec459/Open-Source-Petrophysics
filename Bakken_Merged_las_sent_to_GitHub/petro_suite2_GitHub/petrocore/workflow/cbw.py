# petrocore/workflows/cbw.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CBWConfig:
    vsh_col: str = "VSH_HL"
    cbw_col: str = "CBW"          # measured/loaded CBW (from NMR partition), used only for QC plot externally
    phit_col: str = "PHIT"

    out_cbwapp: str = "CBWapp"
    out_phie: str = "PHIE"
    out_swb: str = "Swb"
    out_qv: str = "Qv"

    # Qv model constants (your Hill, Shirley & Klein form)
    qv_cap: float = 5.0


def compute_cbw_from_vsh_intercept(
    *,
    analysis_df: pd.DataFrame,
    cbw_intercept: float,
    den_fl: float,
    SAL: float,
    cfg: CBWConfig = CBWConfig(),
    logger: Optional[Callable[[str], None]] = None,
) -> dict:
    """
    Implements your workflow:

      CBWapp = VSH_HL * CBW_Int
      PHIE   = PHIT - CBW_Int * VSH_HL
      Swb    = 1 - PHIE/PHIT
      Qv     = Swb / (0.6425/sqrt(den_fl*SAL) + 0.22)

    Notes:
      - Clips: CBWapp, PHIE, Swb -> [0,1]; Qv -> [0, qv_cap]
      - Handles PHIT<=0 safely (Swb, Qv -> NaN there)
    """
    if analysis_df is None or analysis_df.empty:
        raise ValueError("analysis_df is empty.")

    for c in (cfg.vsh_col, cfg.phit_col):
        if c not in analysis_df.columns:
            raise ValueError(f"Missing required column in analysis_df: '{c}'")

    if not np.isfinite(cbw_intercept) or cbw_intercept < 0:
        raise ValueError("cbw_intercept must be a finite non-negative number.")
    if not np.isfinite(den_fl) or den_fl <= 0:
        raise ValueError("den_fl must be > 0.")
    if not np.isfinite(SAL) or SAL <= 0:
        raise ValueError("SAL must be > 0.")

    df = analysis_df.copy()

    vsh = pd.to_numeric(df[cfg.vsh_col], errors="coerce").to_numpy(dtype=float)
    phit = pd.to_numeric(df[cfg.phit_col], errors="coerce").to_numpy(dtype=float)

    # CBWapp and PHIE
    cbwapp = np.clip(vsh * cbw_intercept, 0.0, 1.0)
    phie = np.clip(phit - cbw_intercept * vsh, 0.0, 1.0)

    # Swb (avoid divide by zero)
    swb = np.full_like(phit, np.nan, dtype=float)
    good = np.isfinite(phit) & (phit > 0) & np.isfinite(phie)
    swb[good] = np.clip(1.0 - (phie[good] / phit[good]), 0.0, 1.0)

    # Qv from Swb
    denom = (0.6425 / np.sqrt(den_fl * SAL)) + 0.22
    qv = np.full_like(phit, np.nan, dtype=float)
    ok = np.isfinite(swb)
    qv[ok] = np.clip(swb[ok] / denom, 0.0, cfg.qv_cap)

    df[cfg.out_cbwapp] = cbwapp
    df[cfg.out_phie] = phie
    df[cfg.out_swb] = swb
    df[cfg.out_qv] = qv

    n_valid = int(np.isfinite(qv).sum())

    report = (
        "=== CBW from Vsh Intercept ===\n"
        f"Vsh column     : {cfg.vsh_col}\n"
        f"PHIT column    : {cfg.phit_col}\n"
        f"CBW intercept  : {cbw_intercept:.4f}\n"
        f"den_fl, SAL    : {den_fl:g}, {SAL:g}\n"
        f"Outputs        : {cfg.out_cbwapp}, {cfg.out_phie}, {cfg.out_swb}, {cfg.out_qv}\n"
        f"Valid Qv       : {n_valid:,} / {len(df):,}\n"
    )

    if logger:
        logger(report)

    return {"analysis_df": df, "report": report}
