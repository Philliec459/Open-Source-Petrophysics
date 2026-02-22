import numpy as np
import pandas as pd
import lasio
import matplotlib.pyplot as plt

def to_num(df, c):
    if c not in df.columns:
        return None
    return pd.to_numeric(df[c], errors="coerce").to_numpy(float)

def interp_to_grid(depth, x, grid):
    m = np.isfinite(depth) & np.isfinite(x)
    if m.sum() < 50:
        return np.full_like(grid, np.nan, float)
    d = depth[m]; v = x[m]
    o = np.argsort(d)
    d = d[o]; v = v[o]
    return np.interp(grid, d, v, left=np.nan, right=np.nan)

def deriv(x):
    dx = np.full_like(x, np.nan, float)
    m = np.isfinite(x)
    idx = np.where(m)[0]
    if len(idx) < 5:
        return dx
    dx[idx[1:]] = x[idx[1:]] - x[idx[:-1]]
    return dx

def best_lag_abs_corr(a, b, max_lag):
    best_lag = 0
    best_score = -np.inf
    best_r = np.nan
    for lag in range(-max_lag, max_lag+1):
        if lag < 0:
            aa = a[-lag:]; bb = b[:len(aa)]
        elif lag > 0:
            bb = b[lag:]; aa = a[:len(bb)]
        else:
            aa = a; bb = b
        m = np.isfinite(aa) & np.isfinite(bb)
        if m.sum() < 200:
            continue
        r = np.corrcoef(aa[m], bb[m])[0, 1]
        if not np.isfinite(r):
            continue
        score = abs(r)
        if score > best_score:
            best_score = score
            best_lag = lag
            best_r = r
    return best_lag, best_score, best_r

def density_porosity(rhob, rho_ma=2.71, rho_f=1.10):
    rhob = np.asarray(rhob, float)
    denom = (rho_ma - rho_f)
    phi = (rho_ma - rhob) / denom
    return np.clip(phi, -0.15, 0.60)

def sonic_porosity(dt, dt_ma=47.6, dt_f=189.0):
    dt = np.asarray(dt, float)
    denom = (dt_f - dt_ma)
    phi = (dt - dt_ma) / denom
    return np.clip(phi, -0.15, 0.60)

def qc_run_alignment(df, ref_curve="GR_EDTC", other_series=None, step=0.5, max_lag_ft=10):
    """
    other_series: dict[label -> ndarray on original depth index]
    """
    depth = df.index.to_numpy(float)
    grid = np.arange(np.nanmin(depth), np.nanmax(depth)+step*0.5, step)

    ref = to_num(df, ref_curve)
    if ref is None:
        raise ValueError(f"Missing reference curve {ref_curve}")

    ref_g = interp_to_grid(depth, ref, grid)
    ref_d = deriv(ref_g)

    out = []
    max_lag = int(round(max_lag_ft/step))

    for label, x in other_series.items():
        if x is None:
            continue
        xg = interp_to_grid(depth, x, grid)
        xd = deriv(xg)
        lag_s, score, r = best_lag_abs_corr(ref_d, xd, max_lag)
        out.append((label, lag_s*step, r, score))
    return pd.DataFrame(out, columns=["series", "best_lag_ft_vs_GR_EDTC", "corr_at_best", "abs_corr_score"])

def plot_tracks(df, out_png="qc_tracks.png", dmin=None, dmax=None):
    depth = df.index.to_numpy(float)
    if dmin is None: dmin = float(np.nanmin(depth))
    if dmax is None: dmax = float(np.nanmax(depth))

    # Build porosity equivalents
    npor = to_num(df, "NPOR")
    rhoz = to_num(df, "RHOZ")
    dtco = to_num(df, "DTCO")

    phi_d = density_porosity(rhoz) if rhoz is not None else None
    phi_s = sonic_porosity(dtco) if dtco is not None else None

    def plot_track(ax, series_dict, title, xlab=None, invert_x=False, xscale=None):
        for name, x in series_dict.items():
            if x is None:
                continue
            ax.plot(x, depth, label=name)
        ax.set_ylim(dmax, 10000)  # inverted depth
        if invert_x:
            ax.invert_xaxis()
        if xscale:
            ax.set_xscale(xscale)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
        if xlab:
            ax.set_xlabel(xlab)

    fig, axs = plt.subplots(1, 4, figsize=(18, 38), sharey=True)

    # Track 1: Gamma
    plot_track(
        axs[0],
        {
            "GR_EDTC": to_num(df, "GR_EDTC"),
            "HCGR":    to_num(df, "HCGR"),
            "HSGR":    to_num(df, "HSGR"),
        },
        "Track 1: Gamma",
        xlab="GAPI",
    )

    # Track 2: Resistivity
    plot_track(
        axs[1],
        {"AF90": to_num(df, "AF90"), "AT90": to_num(df, "AT90")},
        "Track 2: Resistivity",
        xlab="ohm-m",
        xscale="log",
    )

    # Track 3: Porosity equivalents (apples-to-apples)
    plot_track(
        axs[2],
        {"NPOR": npor, "PHI_DEN": phi_d, "PHI_Sonic": phi_s},
        "Track 3: Porosity equivalents",
        xlab="v/v",
        invert_x=False,
    )

    # Track 4: NMR vs conventional porosity
    plot_track(
        axs[3],
        {
            "PHIT_NMR": to_num(df, "PHIT_NMR"),
            #"PHIE_NMR": to_num(df, "PHIE_NMR"),
            "NPOR": npor,
            "PHI_DEN": phi_d,
        },
        "Track 4: NMR vs conventional",
        xlab="v/v",
    )

    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)
    print("Wrote:", out_png)

def main():
    #las = lasio.read("./Merged_Well_Log_Bakken_Bakken_renamed.las")
    las = lasio.read("./Merged_Well_Log_JB_11-6TFH_29062 JB 11-6TFH_renamed.las")
    df = las.df().copy()
    df.index.name = "DEPT"

    # Choose a zoom window if you want (optional)
    # dmin, dmax = 10000, 10500
    dmin, dmax = None, None

    plot_tracks(df, out_png="qc_merge_tracks.png", dmin=dmin, dmax=dmax)

    # Build series to compare to GR_EDTC (use derivatives for bed-edge timing)
    npor = to_num(df, "NPOR")
    rhoz = to_num(df, "RHOZ")
    dtco = to_num(df, "DTCO")

    other = {
        # gamma companions
        "HCGR": to_num(df, "HCGR"),
        "HSGR": to_num(df, "HSGR"),

        # resistivity
        "AF90": to_num(df, "AF90"),
        "AT90": to_num(df, "AT90"),

        # porosity equivalents (better than raw RHOB/DT)
        "NPOR": npor,
        "PHI_DEN": density_porosity(rhoz) if rhoz is not None else None,
        "PHI_Sonic": sonic_porosity(dtco) if dtco is not None else None,

        # NMR
        "PHIT_NMR": to_num(df, "PHIT_NMR"),
        #"PHIE_NMR": to_num(df, "PHIE_NMR"),
    }

    rep = qc_run_alignment(df, ref_curve="GR_EDTC", other_series=other, step=0.5, max_lag_ft=10)
    rep = rep.sort_values("abs_corr_score", ascending=False)
    rep.to_csv("qc_merge_alignment_vs_GR_EDTC.csv", index=False)
    print("Wrote: qc_merge_alignment_vs_GR_EDTC.csv")
    print(rep.to_string(index=False))

if __name__ == "__main__":
    main()
