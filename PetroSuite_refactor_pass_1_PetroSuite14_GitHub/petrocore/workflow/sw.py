
# petrocore/workflow/sw.py

import numpy as np


def compute_sw_waxman_smits(
    Rt,
    phi,
    Rw,
    B,
    Qv,
    m=2.0,
    n=2.0,
    max_iter=20,
    tol=1e-5,
):
    """
    Waxman–Smits water saturation

    Parameters:
        Rt   : resistivity array
        phi  : porosity array
        Rw   : water resistivity
        B    : equivalent conductance
        Qv   : cation exchange capacity (from CBW or Vsh)
        m, n : Archie exponents

    Returns:
        Sw array
    """

    Rt = np.asarray(Rt, dtype=float)
    phi = np.asarray(phi, dtype=float)
    Qv = np.asarray(Qv, dtype=float)

    Sw = np.full_like(Rt, 1.0)

    for _ in range(max_iter):
        with np.errstate(divide="ignore", invalid="ignore"):

            term = (Sw**n) * (phi**m) / Rw + B * Qv * Sw
            Sw_new = (1.0 / Rt) / term

        Sw_new = np.clip(Sw_new, 0.0, 1.0)

        if np.allclose(Sw, Sw_new, atol=tol, equal_nan=True):
            break

        Sw = Sw_new

    # Clean invalid zones
    bad = (~np.isfinite(Rt)) | (~np.isfinite(phi)) | (phi <= 0)
    Sw[bad] = np.nan

    return Sw


