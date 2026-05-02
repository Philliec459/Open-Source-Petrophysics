

# petrocore/workflow/xplot_helpers.py



from pathlib import Path
import numpy as np
import pandas as pd

from .xplot_config import CHART_DEFS, XPLOT_PRESETS, Z_AXIS_DEFAULTS, Z_AXIS_FALLBACK



CHART_DIR = Path(__file__).resolve().parent / "charts"



def get_preset(name):
    if not name:
        raise KeyError("Empty preset name")

    # exact key match first
    if name in XPLOT_PRESETS:
        preset = dict(XPLOT_PRESETS[name])
    else:
        # common GUI/display-name aliases
        aliases = {
            "Neutron-Density": "neutron_density",
            "Neutron-Sonic": "neutron_sonic",
            "Sonic-Neutron": "neutron_sonic",
            "Sonic-Density": "sonic_density",
            "PEF-Density": "pef_density",
            "PEF-Bulk Density": "pef_density",
            "UMAA-RHOMAA": "umaa_rhomaa",
            "POTA-THOR": "pota-thor",
            "POTA-PEF": "pota-pef",
            "Pickett Plot": "pickett_plot",
            "Buckles Plot": "buckles_plot",
            "Steiber Plot": "steiber_plot",
            "Generic XY": "generic_xy",
            "Generic Regression XY": "generic_regression_xy",
            "Histogram": "histogram",
            "Log-Log": "log_log",
            "Semilog": "semilog",
        }

        key = aliases.get(name, name)

        if key in XPLOT_PRESETS:
            preset = dict(XPLOT_PRESETS[key])
        else:
            # try matching against preset titles
            preset = None
            for preset_key, p in XPLOT_PRESETS.items():
                if p.get("title") == name or p.get("name") == name:
                    preset = dict(p)
                    break

            if preset is None:
                print("[get_preset] unknown preset:", name)
                print("[get_preset] available keys:", list(XPLOT_PRESETS.keys()))
                raise KeyError(name)

    # ---- normalize optional fields ----
    preset.setdefault("title", preset.get("name", name))
    preset.setdefault("phi_col", None)
    preset.setdefault("extra_cols", [])
    preset.setdefault("points", [])
    preset.setdefault("log_x", False)
    preset.setdefault("log_y", False)
    preset.setdefault("reverse_x", False)
    preset.setdefault("reverse_y", False)
    preset.setdefault("x_label", preset.get("x_col", "X"))
    preset.setdefault("y_label", preset.get("y_col", "Y"))
    preset.setdefault("xlim", None)
    preset.setdefault("ylim", None)

    return preset


def get_chart_def(chart_number: int) -> dict:
    return CHART_DEFS[chart_number]


def load_chart_df(chart_number: int) -> pd.DataFrame:
    chart_def = get_chart_def(chart_number)
    return pd.read_excel(CHART_DIR / chart_def["file"], index_col=False)


def first_curve_in_df(df: pd.DataFrame, candidates: list[str]):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def get_z_config(z_curve: str) -> dict:
    return Z_AXIS_DEFAULTS.get(z_curve, Z_AXIS_FALLBACK)


def compute_z_range(series: pd.Series, cfg: dict):
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return None, None

    if cfg["scale"] == "log":
        s = s[s > 0]
        if s.empty:
            return None, None

    if cfg["mode"] == "fixed":
        return cfg["zmin"], cfg["zmax"]

    if cfg["mode"] == "percentile":
        return float(np.percentile(s, cfg["pmin"])), float(np.percentile(s, cfg["pmax"]))

    return float(s.min()), float(s.max())


