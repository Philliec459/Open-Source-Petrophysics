# petrocore/workflows/vsh.py

"""
Vsh (Shale Volume) calculations

All Vsh-related math lives here.
UI should only call these functions.
"""

import numpy as np


def compute_vsh_gr(gr, gr_clean, gr_shale):
    gr = np.asarray(gr, dtype=float)

    denom = (gr_shale - gr_clean)

    # Always initialize vsh first (this avoids your crash)
    vsh = np.full_like(gr, np.nan, dtype=float)

    if abs(denom) >= 1e-12:
        vsh = (gr - gr_clean) / denom

    # Preserve NaNs from input
    vsh[~np.isfinite(gr)] = np.nan

    return np.clip(vsh, 0.0, 1.0)




def hodges_lehman_estimator(values):
    """
    Hodges-Lehman estimator (robust median of pairwise means)
    """
    values = np.asarray(values)
    n = len(values)

    if n == 0:
        return np.nan
    if n == 1:
        return values[0]

    # Pairwise means
    means = []
    for i in range(n):
        for j in range(i, n):
            means.append((values[i] + values[j]) / 2.0)

    return np.median(means)


def smooth_vsh_hl(vsh, window=5):
    """
    Apply Hodges-Lehman smoothing to Vsh curve

    Parameters:
        vsh : array-like
        window : int (odd number recommended)

    Returns:
        np.ndarray
    """
    vsh = np.asarray(vsh)

    if window < 3:
        return vsh.copy()

    half = window // 2
    smoothed = np.full_like(vsh, np.nan)

    for i in range(len(vsh)):
        start = max(0, i - half)
        end = min(len(vsh), i + half + 1)


        window_vals = vsh[start:end]
        window_vals = window_vals[np.isfinite(window_vals)]

        if len(window_vals) > 0:
            smoothed[i] = hodges_lehman_estimator(window_vals)

    return smoothed


def compute_vsh_hl(gr, gr_clean, gr_shale, window=5):
    """
    Full pipeline:
        GR → linear Vsh → HL smoothing
    """
    vsh = compute_vsh_gr(gr, gr_clean, gr_shale)
    return smooth_vsh_hl(vsh, window=window)