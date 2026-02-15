import numpy as np
import pandas as pd

def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return np.nan
    aa = a[m] - np.nanmean(a[m])
    bb = b[m] - np.nanmean(b[m])
    da = np.nanstd(aa)
    db = np.nanstd(bb)
    if da == 0 or db == 0:
        return np.nan
    return float(np.nanmean((aa/da) * (bb/db)))

def windowed_bulk_shifts(df_base, df_mov, curve, win_ft=150.0, step_ft=150.0,
                         shift_min=-30.0, shift_max=30.0, shift_step=0.5):
    zb = df_base.index.astype(float).values
    xb = pd.to_numeric(df_base[curve], errors="coerce").values.astype(float)

    zm = df_mov.index.astype(float).values
    xm = pd.to_numeric(df_mov[curve], errors="coerce").values.astype(float)

    # overlap depth only
    zmin = max(np.nanmin(zb), np.nanmin(zm))
    zmax = min(np.nanmax(zb), np.nanmax(zm))

    shifts = np.arange(shift_min, shift_max + 1e-9, shift_step, dtype=float)

    rows = []
    z0 = zmin
    while z0 < zmax:
        z1 = min(z0 + win_ft, zmax)
        zc = 0.5 * (z0 + z1)

        mb = (zb >= z0) & (zb <= z1) & np.isfinite(xb)
        if mb.sum() < 20:
            rows.append((z0, z1, zc, np.nan, np.nan, int(mb.sum())))
            z0 += step_ft
            continue

        zb_w = zb[mb]
        xb_w = xb[mb]

        # sort moving for interp
        mm = np.isfinite(zm) & np.isfinite(xm)
        zm0 = zm[mm]
        xm0 = xm[mm]
        order = np.argsort(zm0)
        zm0 = zm0[order]
        xm0 = xm0[order]

        best_shift = np.nan
        best_corr = -np.inf
        best_n = 0

        for s in shifts:
            x_mov_on_base = np.interp(zb_w + s, zm0, xm0, left=np.nan, right=np.nan)
            m = np.isfinite(x_mov_on_base) & np.isfinite(xb_w)
            if m.sum() < 20:
                continue
            corr = np.corrcoef(xb_w[m], x_mov_on_base[m])[0, 1]
            if np.isfinite(corr) and corr > best_corr:
                best_corr = float(corr)
                best_shift = float(s)
                best_n = int(m.sum())

        rows.append((z0, z1, zc, best_shift, best_corr, best_n))
        z0 += step_ft

    return pd.DataFrame(rows, columns=["z0", "z1", "z_center", "shift_ft", "corr", "n"]),
