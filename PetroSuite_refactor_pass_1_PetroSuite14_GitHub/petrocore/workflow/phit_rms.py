from __future__ import annotations

import numpy as np
import pandas as pd


def compute_phit_rms(
    df: pd.DataFrame,
    rhob_curve: str,
    nphi_curve: str,
    matrix_density: float = 2.71,
    fluid_density: float = 1.1,
) -> pd.DataFrame:

    out = df.copy()

    if rhob_curve not in out.columns:
        raise KeyError(f"Density curve not found: {rhob_curve}")

    if nphi_curve not in out.columns:
        raise KeyError(f"Neutron curve not found: {nphi_curve}")

    rhob = pd.to_numeric(out[rhob_curve], errors="coerce")
    nphi = pd.to_numeric(out[nphi_curve], errors="coerce")

    por_den = (matrix_density - rhob) / (matrix_density - fluid_density)
    phit = np.sqrt((por_den**2 + nphi**2) / 2.0)

    out["POR_DEN"] = por_den
    out["PHIT"] = phit

    return out