#!/usr/bin/env python3
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
import lasio
import matplotlib.pyplot as plt
from pathlib import Path


def robust_z(x: np.ndarray) -> np.ndarray:
    """Robust z-score using median/MAD. Keeps NaNs."""
    x = x.astype(float)
    m = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - m))
    if not np.isfinite(mad) or mad < 1e-12:
        # fallback to standard deviation
        s = np.nanstd(x)
        if not np.isfinite(s) or s < 1e-12:
            return x * np.nan
        return (x - m) / s
    return (x - m) / (1.4826 * mad)


def interpolate_to_grid(depth: np.ndarray, x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Linear interpolation to grid, keeping NaNs where data absent."""
    m = np.isfinite(depth) & np.isfinite(x)
    if m.sum() < 5:
        return np.full_like(grid, np.nan, dtype=float)

    d = depth[m]
    v = x[m]
    # ensure increasing for interp
    o = np.argsort(d)
    d = d[o]
    v = v[o]

    y = np.interp(grid, d, v, left=np.nan, right=np.nan)
    # np.interp can't produce NaNs in-between gaps; mask large gaps:
    # mark grid points farther than 2*step from nearest original sample as NaN
    # (simple gap handling)
    step = np.nanmedian(np.diff(d))
    if not np.isfinite(step) or step <= 0:
        return y
    # nearest distance via searchsorted
    idx = np.searchsorted(d, grid)
    idx0 = np.clip(idx - 1, 0, len(d) - 1)
    idx1 = np.clip(idx, 0, len(d) - 1)
    dist = np.minimum(np.abs(grid - d[idx0]), np.abs(grid - d[idx1]))
    y[dist > 2.5 * step] = np.nan
    return y


def best_lag_corr(a: np.ndarray, b: np.ndarray, max_lag_samples: int) -> tuple[int, float]:
    """
    Find lag (in samples) that maximizes correlation between a and b.
    Positive lag means b is shifted DOWN (to deeper) relative to a (b occurs later).
    """
    best_lag = 0
    best_r = -np.inf

    for lag in range(-max_lag_samples, max_lag_samples + 1):
        if lag < 0:
            aa = a[-lag:]
            bb = b[: len(aa)]
        elif lag > 0:
            bb = b[lag:]
            aa = a[: len(bb)]
        else:
            aa = a
            bb = b

        m = np.isfinite(aa) & np.isfinite(bb)
        if m.sum() < 50:
            continue

        x = aa[m]
        y = bb[m]
        # correlation
        r = np.corrcoef(x, y)[0, 1]
        if np.isfinite(r) and r > best_r:
            best_r = r
            best_lag = lag

    return best_lag, float(best_r if np.isfinite(best_r) else np.nan)


def qc_alignment(
    las_path: Path,
    out_dir: Path,
    families: dict[str, list[str]],
    grid_step: float | None,
    max_lag_ft: float,
    make_plots: bool,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    las = lasio.read(str(las_path))
    df = las.df().copy()
    df.index.name = "DEPT"

    depth = df.index.to_numpy(dtype=float)
    if depth.size < 10:
        raise RuntimeError("LAS has too few depth samples.")

    # Depth sanity
    diffs = np.diff(depth)
    step_med = float(np.nanmedian(diffs)) if diffs.size else np.nan
    monotonic = bool(np.all(diffs > 0))
    has_dupes = bool(np.any(diffs == 0))

    # Define grid
    if grid_step is None:
        grid_step = step_med if np.isfinite(step_med) and step_med > 0 else 0.5

    d0 = float(np.nanmin(depth))
    d1 = float(np.nanmax(depth))
    grid = np.arange(d0, d1 + grid_step * 0.5, grid_step)

    report_rows = []
    worst_examples = []

    for fam, cols in families.items():
        present = [c for c in cols if c in df.columns]
        if len(present) < 2:
            continue

        # interpolate + robust normalize each curve
        series = {}
        for c in present:
            y = interpolate_to_grid(depth, pd.to_numeric(df[c], errors="coerce").to_numpy(), grid)
            y = robust_z(y)
            series[c] = y

        max_lag_samples = int(round(max_lag_ft / grid_step))

        # pairwise lags
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                c1, c2 = present[i], present[j]
                lag_s, r = best_lag_corr(series[c1], series[c2], max_lag_samples)
                lag_ft = lag_s * grid_step

                report_rows.append(
                    {
                        "family": fam,
                        "curve_a": c1,
                        "curve_b": c2,
                        "grid_step_ft": grid_step,
                        "best_lag_samples": lag_s,
                        "best_lag_ft": lag_ft,
                        "corr_at_best_lag": r,
                    }
                )

                worst_examples.append((abs(lag_ft), fam, c1, c2, lag_ft, r))

    #report = pd.DataFrame(report_rows).sort_values(["family", "best_lag_ft"], key=lambda s: np.abs(s))

    report = pd.DataFrame(report_rows)
    if not report.empty:
        report["abs_lag_ft"] = report["best_lag_ft"].abs()
        report = report.sort_values(["family", "abs_lag_ft", "corr_at_best_lag"], ascending=[True, True, False])

    
    report_path = out_dir / "qc_alignment_report.csv"
    report.to_csv(report_path, index=False)

    # Save depth sanity
    sanity = {
        "monotonic_increasing_depth": monotonic,
        "has_duplicate_depths": has_dupes,
        "median_step_ft": step_med,
        "grid_step_ft_used": grid_step,
        "depth_min": float(np.nanmin(depth)),
        "depth_max": float(np.nanmax(depth)),
    }
    (out_dir / "qc_depth_sanity.txt").write_text("\n".join(f"{k}: {v}" for k, v in sanity.items()) + "\n")

    # Optional plots of the worst offenders
    if make_plots and worst_examples:
        worst_examples.sort(reverse=True)
        for k, (abs_lag, fam, c1, c2, lag_ft, r) in enumerate(worst_examples[:6], start=1):
            y1 = pd.to_numeric(df[c1], errors="coerce").to_numpy(dtype=float)
            y2 = pd.to_numeric(df[c2], errors="coerce").to_numpy(dtype=float)

            # choose a representative window with lots of valid data
            m = np.isfinite(y1) & np.isfinite(y2)
            if m.sum() < 200:
                continue
            # window centered on median valid depth
            d_valid = depth[m]
            center = float(np.nanmedian(d_valid))
            w = 50.0  # ft
            sel = (depth >= center - w) & (depth <= center + w)

            fig, ax = plt.subplots(1, 1, figsize=(8, 10))
            ax.plot(y1[sel], depth[sel], label=c1)
            ax.plot(y2[sel], depth[sel], label=c2)
            ax.invert_yaxis()
            ax.set_title(f"{fam}: {c1} vs {c2} | best lag ~ {lag_ft:+.2f} ft | r={r:.3f}")
            ax.set_xlabel("Value")
            ax.set_ylabel("Depth (ft)")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / f"qc_worst_{k}_{fam}_{c1}_vs_{c2}.png", dpi=160)
            plt.close(fig)

    print(f"Wrote: {report_path}")
    print(f"Wrote: {out_dir / 'qc_depth_sanity.txt'}")
    if make_plots:
        print(f"Wrote plots into: {out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("las", type=str, help="Path to merged LAS file")
    p.add_argument("--out", type=str, default="qc_out", help="Output directory")
    p.add_argument("--step", type=float, default=None, help="Resample grid step (ft). Default uses median LAS step.")
    p.add_argument("--max_lag_ft", type=float, default=10.0, help="Max lag search (ft)")
    p.add_argument("--plots", action="store_true", help="Write a few plots for worst offenders")
    args = p.parse_args()

    families = {
        "gamma": ["GR_EDTC ","GR", "CGR","HCGR", "SGR","HSGR"],
        "porosity_sonic": ["NPOR","NPHI", "TNPH", "RHOB", "RHOZ","PHIT_NMR", "DT", "DTC", "DTCO"],
        #"resistivity": ["RT", "ILD", "LLD", "ILM", "LLS", "AT90", "AF90", "AT60", "AF60"],
    }

    qc_alignment(
        las_path=Path(args.las),
        out_dir=Path(args.out),
        families=families,
        grid_step=args.step,
        max_lag_ft=args.max_lag_ft,
        make_plots=args.plots,
    )


if __name__ == "__main__":
    main()


#python qc_merged_las_alignment.py path/to/Merged_....las --out qc_merge --max_lag_ft 10 --plots

