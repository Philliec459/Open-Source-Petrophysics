





# apps/merge_gui/ui_panels/plots_panel.py
from __future__ import annotations

print(">>> LOADING plots_panel.py from:", __file__)

from pathlib import Path
import numpy as np
import pandas as pd
import pyqtgraph as pg
import matplotlib.pyplot as plt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PySide6.QtCore import Qt, QSettings, QTimer, Signal, QEvent
from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QSizePolicy,
    QMessageBox,
    QFileDialog,
    QVBoxLayout,
    QHBoxLayout,
    QTabWidget,
    QFormLayout,
    QPushButton,
    QInputDialog,
    QSplitter,
    QDoubleSpinBox,
)

# =============================================================================
# Parameter specs
# =============================================================================

SW_PARAM_SPEC = {
    "sw.m_cem": dict(default=1.90, mn=1.00, mx=3.00, step=0.01, dec=2, label="m (cementation)"),
    "sw.n_sat": dict(default=2.00, mn=1.00, mx=3.00, step=0.01, dec=2, label="n (saturation)"),
    "sw.rw": dict(default=0.023, mn=0.001, mx=0.20, step=0.001, dec=4, label="Rw"),
    "sw.mslope": dict(default=1.00, mn=0.01, mx=4.00, step=0.01, dec=2, label="M* slope"),
    "sw.B": dict(default=12.20, mn=0.00, mx=50.0, step=0.1, dec=2, label="B (Bdacy)"),
    "sw.cbw_intercept": dict(default=0.10, mn=0.00, mx=0.50, step=0.01, dec=3, label="CBW Intercept"),
    "sw.T_F": dict(default=150.0, mn=60.0, mx=300.0, step=1.0, dec=1, label="Temp (F)"),
    "sw.den_fl": dict(default=1.10, mn=0.80, mx=1.20, step=0.01, dec=2, label="Fluid density"),
}

VSH_PARAM_SPEC = {
    "neut_matrix": dict(default=-0.04, mn=-0.15, mx=1.20, step=0.01, dec=3, label="N_mat"),
    "neut_shale": dict(default=0.32, mn=-0.15, mx=1.20, step=0.01, dec=3, label="N_sh"),
    "den_matrix": dict(default=2.65),
    "den_shale": dict(default=2.65),
    "neut_fl": dict(default=1.00),
    "den_fl": dict(default=1.10),
    "dt_matrix": dict(default=55.5, mn=40.0, mx=120.0, step=0.5, dec=1, label="DT_mat"),
    "dt_shale": dict(default=90.0, mn=40.0, mx=140.0, step=0.5, dec=1, label="DT_sh"),
    "T_f": dict(default=150.0),
    "dt_fl": dict(default=189.0),
}


# =============================================================================
# Utilities
# =============================================================================

def _first_present(cols, candidates):
    s = set(cols)
    for c in candidates:
        if c in s:
            return c
    return None


def first_present(cols, candidates):
    return _first_present(cols, candidates)


def get_col(df: pd.DataFrame, col: str | None):
    if df is None or col is None or col not in df.columns:
        return None
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def pget(params: dict, key: str):
    if isinstance(params, dict) and key in params and params[key] is not None:
        return params[key]
    return VSH_PARAM_SPEC[key]["default"]


def make_dspin(minv, maxv, step, decimals, value, *, keyboard_tracking=False):
    sb = QDoubleSpinBox()
    sb.setRange(minv, maxv)
    sb.setSingleStep(step)
    sb.setDecimals(decimals)
    sb.setKeyboardTracking(keyboard_tracking)
    sb.setAccelerated(True)
    try:
        sb.setValue(float(value))
    except Exception:
        sb.setValue(float(minv))
    return sb


def vsh_from_gr(gr, gr_clean=25.0, gr_shale=160.0, clip=True):
    gr = np.asarray(gr, dtype=float)
    denom = (gr_shale - gr_clean)
    if abs(denom) < 1e-12:
        v = np.full_like(gr, np.nan, dtype=float)
    else:
        v = (gr - gr_clean) / denom
    if clip:
        v = np.clip(v, 0.0, 1.0)
    v[~np.isfinite(gr)] = np.nan
    return v


def vsh_from_gamma(df: pd.DataFrame, params: dict):
    gamma_col = params.get("cgr_curve")
    if not gamma_col or gamma_col not in df.columns:
        gamma_col = params.get("gr_curve")
    if not gamma_col or gamma_col not in df.columns:
        gamma_col = _first_present(df.columns, ["HCGR", "CGR", "ECGR", "GR_EDTC", "HSGR", "GR", "SGR", "HGR"])
    if not gamma_col or gamma_col not in df.columns:
        return None, None

    gr_clean = float(params.get("gr_clean", 25.0))
    gr_shale = float(params.get("gr_shale", 160.0))
    g = pd.to_numeric(df[gamma_col], errors="coerce").to_numpy(float)
    return vsh_from_gr(g, gr_clean=gr_clean, gr_shale=gr_shale, clip=True), gamma_col


def vsh_from_triangle(x, y, A, B, C, clip=True):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    out = np.full_like(x, np.nan, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if not np.any(m):
        return out

    Ax, Ay = A
    Bx, By = B
    Cx, Cy = C

    denom = (By - Cy) * (Ax - Cx) + (Cx - Bx) * (Ay - Cy)
    if not np.isfinite(denom) or abs(denom) < 1e-12:
        return out

    lamB = ((Cy - Ay) * (x[m] - Cx) + (Ax - Cx) * (y[m] - Cy)) / denom
    if clip:
        lamB = np.clip(lamB, 0.0, 1.0)
    out[m] = lamB
    return out


def vsh_from_nd_triangle(df: pd.DataFrame, params: dict):
    tnph = params.get("tnph_curve")
    rhob = params.get("rhob_curve")

    if (not tnph) or (tnph not in df.columns):
        tnph = _first_present(df.columns, ["TNPH", "NPHI", "CNL", "NPOR"])
    if (not rhob) or (rhob not in df.columns):
        rhob = _first_present(df.columns, ["RHOZ", "RHOB"])
    if (tnph is None) or (rhob is None):
        return None

    x = pd.to_numeric(df[tnph], errors="coerce").to_numpy(float)
    y = pd.to_numeric(df[rhob], errors="coerce").to_numpy(float)

    A = (float(pget(params, "neut_matrix")), float(pget(params, "den_matrix")))
    B = (float(pget(params, "neut_shale")), float(pget(params, "den_shale")))
    C = (float(pget(params, "neut_fl")), float(pget(params, "den_fl")))
    return vsh_from_triangle(x, y, A, B, C, clip=True)


def vsh_from_dt_triangle(df: pd.DataFrame, params: dict):
    rhob = params.get("rhob_curve")
    if (not rhob) or (rhob not in df.columns):
        rhob = _first_present(df.columns, ["RHOZ", "RHOB"])

    dtco = params.get("dtco_curve")
    if (not dtco) or (dtco not in df.columns):
        dtco = _first_present(df.columns, ["DTCO", "DTC", "AC"])

    if (rhob is None) or (dtco is None):
        return None

    x = pd.to_numeric(df[dtco], errors="coerce").to_numpy(float)
    y = pd.to_numeric(df[rhob], errors="coerce").to_numpy(float)

    A = (float(pget(params, "dt_matrix")), float(pget(params, "den_matrix")))
    B = (float(pget(params, "dt_shale")), float(pget(params, "den_shale")))
    C = (float(pget(params, "dt_fl")), float(pget(params, "den_fl")))
    return vsh_from_triangle(x, y, A, B, C, clip=True)


def compute_sal_kppm_from_rw75(Rw75):
    Rw75 = np.asarray(Rw75, dtype=float)
    x = np.clip(Rw75 - 0.0123, 1e-6, None)
    return (10 ** ((3.562 - np.log10(x)) / 0.955)) / 1000.0


def compute_bdacy(T_F, Rw):
    Rw = np.asarray(Rw, dtype=float)
    TC = (float(T_F) - 32.0) / 1.8
    TC_safe = max(TC, 1e-6)
    Rw_safe = np.clip(Rw, 1e-6, None)
    term1 = (1.0 - 0.83 * np.exp(-np.exp(-2.38 + (42.17 / TC_safe)) / Rw_safe))
    term2 = (-3.16 + 1.59 * np.log(TC_safe)) ** 2
    return term1 * term2


def write_constants_to_file(file_path, *, m_cem, n_sat, Rw, mslope, Bdacy):
    try:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f:
            f.write("# Pickett / Saturation Parameters\n")
            f.write("# -------------------------------\n")
            f.write(f"m_cem  : {m_cem:.3f}\n")
            f.write(f"n_sat  : {n_sat:.3f}\n")
            f.write(f"Rw     : {Rw:.5f}\n")
            f.write(f"mslope : {mslope:.3f}\n")
            f.write(f"Bdacy  : {Bdacy:.4f}\n")
        print(f"[PARAMS] Constants written to {file_path}")
    except Exception as e:
        print(f"[PARAMS] Error writing constants: {e}")



def compute_sw_products(
    df: pd.DataFrame,
    rt_col: str,
    phit_col: str,
    qv_col: str | None,
    vsh_col: str | None,
    m_cem: float,
    n_sat: float,
    rw: float,
    mslope: float,
    B: float,
):
    rt = pd.to_numeric(df[rt_col], errors="coerce").to_numpy(float)
    phit = pd.to_numeric(df[phit_col], errors="coerce").to_numpy(float)

    if qv_col and (qv_col in df.columns):
        qv = pd.to_numeric(df[qv_col], errors="coerce").to_numpy(float)
    else:
        qv = np.zeros_like(phit)

    if vsh_col and (vsh_col in df.columns):
        vsh = pd.to_numeric(df[vsh_col], errors="coerce").to_numpy(float)
    else:
        vsh = np.full_like(phit, np.nan)

    n = len(df)
    BVW = np.full(n, np.nan, float)
    SWT = np.full(n, np.nan, float)
    BVO = np.full(n, np.nan, float)
    MSTAR_APP = np.full(n, np.nan, float)
    MSTAR = np.full(n, np.nan, float)

    m = np.isfinite(rt) & np.isfinite(phit) & (rt > 0) & (phit > 0)

    # Archie-style BVW
    BVW_m = phit[m] * ((1.0 / (phit[m] ** m_cem)) * (rw / rt[m])) ** (1.0 / n_sat)
    BVW_m = np.minimum(BVW_m, phit[m])

    SWT_m = np.where(phit[m] > 0, BVW_m / phit[m], np.nan)
    BVO_m = phit[m] * (1.0 - SWT_m)

    # Waxman-Smits-adjusted apparent M*
    denom = rt[m] * (1.0 + rw * B * qv[m])
    ok = (
        np.isfinite(denom) &
        (denom > 0) &
        np.isfinite(phit[m]) &
        (phit[m] > 0) &
        (phit[m] != 1.0)
    )

    ms_app = np.full(np.sum(m), np.nan, float)
    ms_app[ok] = np.log10(rw / denom[ok]) / np.log10(phit[m][ok])

    BVW[m] = BVW_m
    SWT[m] = SWT_m
    BVO[m] = BVO_m
    MSTAR_APP[m] = ms_app

    if np.any(np.isfinite(vsh)):
        MSTAR = vsh * mslope + m_cem

    return dict(
        BVW=BVW,
        SWT=SWT,
        BVO=BVO,
        MSTAR_APP=MSTAR_APP,
        MSTAR=MSTAR,
    )






# =============================================================================
# Parameter store
# =============================================================================

class ParamStore:
    def __init__(self, controller, org="petro_suite6", app="merge_gui"):
        self.controller = controller
        self.settings = QSettings(org, app)

    def get(self, key, default=None):
        state = getattr(self.controller, "state", None)
        params = getattr(state, "params", None) if state is not None else None
        if not isinstance(params, dict):
            if state is not None:
                state.params = {}
                params = state.params
            else:
                params = {}
        if key in params:
            return params[key]
        return self.settings.value(key, default)

    def set(self, key, value):
        state = getattr(self.controller, "state", None)
        params = getattr(state, "params", None) if state is not None else None
        if not isinstance(params, dict):
            if state is not None:
                state.params = {}
                params = state.params
            else:
                params = {}
        params[key] = value
        self.settings.setValue(key, value)


# =============================================================================
# Professional float slider classes
# =============================================================================

class FloatSlider(pg.Qt.QtWidgets.QSlider):
    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        self._fine_div = 10

    def wheelEvent(self, e):
        delta = e.angleDelta().y()
        if delta == 0:
            delta = e.pixelDelta().y()
        if delta == 0:
            return

        step = self.singleStep()
        if e.modifiers() & Qt.ShiftModifier:
            step = max(1, step // self._fine_div)

        direction = 1 if delta > 0 else -1
        self.setValue(self.value() + direction * step)
        e.accept()


class FloatSliderRow(QWidget):
    valueChangedFloat = Signal(float)

    def __init__(self, min_val, max_val, step, decimals, value, parent=None, label_width=70):
        super().__init__(parent)

        self.decimals = int(decimals)
        self.scale = 10 ** self.decimals

        self.min_i = int(round(min_val * self.scale))
        self.max_i = int(round(max_val * self.scale))
        self.step_i = max(1, int(round(step * self.scale)))

        self.slider = FloatSlider(Qt.Horizontal)
        self.slider.setMinimum(self.min_i)
        self.slider.setMaximum(self.max_i)
        self.slider.setSingleStep(self.step_i)
        self.slider.setPageStep(max(1, (self.max_i - self.min_i) // 50))
        self.slider.setMinimumWidth(120)
        self.slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.label = QLabel("")
        self.label.setFixedWidth(label_width)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.slider, stretch=1)
        lay.addWidget(self.label, stretch=0)

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.slider.installEventFilter(self)
        self.label.installEventFilter(self)

        self.slider.valueChanged.connect(self._sync)
        self.slider.valueChanged.connect(lambda _: self.valueChangedFloat.emit(self.value()))

        self.set_value(value)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.MouseButtonDblClick:
            self._prompt_value()
            return True
        return super().eventFilter(obj, event)

    def _sync(self, v_i: int):
        self.label.setText(f"{v_i / self.scale:.{self.decimals}f}")

    def value(self) -> float:
        return self.slider.value() / self.scale

    def set_value(self, v: float):
        v_i = int(round(v * self.scale))
        v_i = max(self.min_i, min(self.max_i, v_i))
        self.slider.setValue(v_i)
        self._sync(v_i)

    def _prompt_value(self):
        v0 = self.value()
        v, ok = QInputDialog.getDouble(
            self,
            "Set value",
            "Value:",
            v0,
            self.min_i / self.scale,
            self.max_i / self.scale,
            self.decimals,
        )
        if ok:
            self.set_value(v)
            self.valueChangedFloat.emit(self.value())


# =============================================================================
# PlotsPanel
# =============================================================================

class PlotsPanel(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.param_store = ParamStore(controller)
        self.settings = QSettings("CrestedButtePetro", "PetroSuite6")

        self.tabs = QTabWidget()

        # Main pyqtgraph depth tracks
        self.tracks = pg.GraphicsLayoutWidget()
        self.track_items = {}
        self.tabs.addTab(self.tracks, "Depth Tracks")

        # Matplotlib canvases
        self.fig_nd, self.ax_nd = plt.subplots(1, 1, figsize=(6, 6))
        self.canvas_nd = FigureCanvas(self.fig_nd)

        self.fig_dt, self.ax_dt = plt.subplots(1, 1, figsize=(6, 6))
        self.canvas_dt = FigureCanvas(self.fig_dt)

        self.fig_gr, self.ax_gr = plt.subplots(1, 1, figsize=(6, 3))
        self.canvas_gr = FigureCanvas(self.fig_gr)

        self.fig_sw = plt.figure(figsize=(11, 7))
        axs = self.fig_sw.subplot_mosaic(
            [
                ["left", "middle"],
                ["right", "cbw"],
            ]
        )
        self.ax_sw_left = axs["left"]
        self.ax_sw_mid = axs["middle"]
        self.ax_sw_right = axs["right"]
        self.ax_sw_cbw = axs["cbw"]
        self.canvas_sw = FigureCanvas(self.fig_sw)

        self.vsh_depth_plot = None
        self.final_tracks = None
        self.final_track_items = {}

        self._seed_vsh_defaults_once()
        self._seed_sw_defaults_once()

        self._build_tracks()

        self.tabs.addTab(self._build_crossplots_tab(), "Crossplots + Histogram")
        self.tabs.addTab(self._build_sw_tab(), "Sw / Pickett / M* / CBW")
        self.tabs.addTab(self._build_final_depth_tab(), "Final Depth Plot")

        layout = QVBoxLayout(self)
        layout.addWidget(self.tabs)

        self._sw_recompute_timer = QTimer(self)
        self._sw_recompute_timer.setSingleShot(True)
        self._sw_recompute_timer.timeout.connect(self._on_compute_sw_clicked)

        self._vsh_recompute_timer = QTimer(self)
        self._vsh_recompute_timer.setSingleShot(True)
        self._vsh_recompute_timer.timeout.connect(self._live_vsh_update)

    # -------------------------------------------------------------------------
    # Public
    # -------------------------------------------------------------------------
    def update_all(self, state):
        self.update_depth_plot(state)
        self.update_nd_crossplot(state)
        self.update_gr_hist(state)
        self.update_dt_rhob_crossplot(state)
        self.update_sw_tab(state)
        try:
            self._refresh_vsh_depth_plot()
        except Exception:
            pass
        try:
            self._refresh_final_depth_plot()
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------
    def _state(self):
        return getattr(self.controller, "state", None)

    def _get_df_view_or_full(self):
        state = self._state()
        if state is None:
            return None
        df = getattr(state, "analysis_df_view", None)
        if df is None or df.empty:
            df = getattr(state, "analysis_df", None)
        return df

    def _safe_refresh(self):
        if hasattr(self.controller, "update_plots"):
            try:
                self.controller.update_plots()
                return
            except Exception:
                pass
        if hasattr(self.controller, "refresh_plots"):
            try:
                self.controller.refresh_plots()
            except Exception:
                pass

    def _safe_rebuild_view(self):
        if hasattr(self.controller, "rebuild_view"):
            try:
                self.controller.rebuild_view()
            except Exception:
                pass

    def _get_sw(self, key: str) -> float:
        spec = SW_PARAM_SPEC[key]
        state = self._state()
        params = getattr(state, "params", {}) if state else {}
        if not isinstance(params, dict):
            params = {}
        v = params.get(key, None)
        if v is None:
            v = self.param_store.get(key, spec["default"])
        try:
            return float(v)
        except Exception:
            return float(spec["default"])

    def _set_param_and_refresh(self, key, value):
        self.param_store.set(key, value)

        state = self._state()
        if state is not None:
            params = getattr(state, "params", None)
            if not isinstance(params, dict):
                state.params = {}
                params = state.params
            params[key] = value

        if key.startswith("sw."):
            self._sw_recompute_timer.start(150)

        if key in ("neut_matrix", "neut_shale", "dt_matrix", "dt_shale", "gr_clean", "gr_shale"):
            self._vsh_recompute_timer.start(150)

        self._safe_refresh()

    def _live_vsh_update(self):
        try:
            self._recompute_vsh_only()
            self._refresh_vsh_depth_plot()
        except Exception:
            pass

    def _seed_vsh_defaults_once(self):
        state = self._state()
        if state is None:
            return
        params = getattr(state, "params", None)
        if not isinstance(params, dict):
            state.params = {}
            params = state.params

        for key, spec in VSH_PARAM_SPEC.items():
            default_val = spec.get("default")
            if key not in params or params[key] is None:
                params[key] = default_val
                try:
                    self.param_store.set(key, default_val)
                except Exception:
                    pass

    def _seed_sw_defaults_once(self):
        state = self._state()
        if state is None:
            return
        params = getattr(state, "params", None)
        if not isinstance(params, dict):
            state.params = {}
            params = state.params

        for k, spec in SW_PARAM_SPEC.items():
            if k not in params or params[k] is None:
                params[k] = spec["default"]
                try:
                    self.param_store.set(k, spec["default"])
                except Exception:
                    pass

    def _depth_array(self, df):
        for c in ["DEPTH", "DEPT", "MD", "TVD", "TVDSS"]:
            if c in df.columns:
                return pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        return pd.to_numeric(df.index, errors="coerce").to_numpy(dtype=float)

    def _get_curve_name(self, df, params, *keys, fallbacks=None):
        for k in keys:
            v = params.get(k)
            if v and v in df.columns:
                return v
        if fallbacks:
            return first_present(df.columns, fallbacks)
        return None

    # -------------------------------------------------------------------------
    # Export
    # -------------------------------------------------------------------------
    def _on_export_df_clicked(self):
        state = self._state()
        if state is None:
            QMessageBox.warning(self, "Export DF", "State object not found.")
            return

        df = getattr(state, "analysis_df_view", None)
        if df is None or df.empty:
            df = getattr(state, "analysis_df", None)

        if df is None or df.empty:
            QMessageBox.warning(self, "Export DF", "No dataframe available to export.")
            return

        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Export DataFrame",
            "qt_export.csv",
            "CSV Files (*.csv);;Excel Files (*.xlsx)"
        )
        if not file_name:
            return

        try:
            out = df.copy()
            depth_name = out.index.name if out.index.name else "DEPT"
            out.insert(0, depth_name, pd.to_numeric(out.index, errors="coerce").to_numpy(dtype=float))

            if file_name.endswith(".xlsx"):
                out.to_excel(file_name, index=False)
            else:
                out.to_csv(file_name, index=False)

            QMessageBox.information(self, "Export DF", f"Saved:\n{file_name}")
        except Exception as e:
            QMessageBox.critical(self, "Export DF", str(e))

    # -------------------------------------------------------------------------
    # Computation buttons
    # -------------------------------------------------------------------------
    def _recompute_swb_only(self):
        state = self._state()
        if state is None:
            return
        df = getattr(state, "analysis_df", None)
        if df is None or df.empty:
            return

        cbw_col = next((c for c in ["CBW", "CBWapp"] if c in df.columns), None)
        phit_col = next((c for c in ["PHIT", "PHIT_NMR", "TCMR", "MPHI", "NMR_PHIT"] if c in df.columns), None)
        if not (cbw_col and phit_col):
            return

        cbw = pd.to_numeric(df[cbw_col], errors="coerce").to_numpy(dtype=float)
        phit = pd.to_numeric(df[phit_col], errors="coerce").to_numpy(dtype=float)

        swb = np.full_like(phit, np.nan, dtype=float)
        m = np.isfinite(cbw) & np.isfinite(phit) & (phit > 0.01)
        swb[m] = cbw[m] / phit[m]
        swb = np.clip(swb, 0.0, 1.0)

        swb_series = pd.Series(swb)
        swb_med = swb_series.rolling(window=3, center=True, min_periods=1).median()
        swb_smooth = swb_med.rolling(window=3, center=True, min_periods=1).mean()
        df["SWB"] = swb_smooth.to_numpy(dtype=float)

    def _recompute_vsh_only(self):
        state = self._state()
        if state is None:
            return
        df = getattr(state, "analysis_df", None)
        if df is None or df.empty:
            return

        params = getattr(state, "params", {}) or {}
        if not isinstance(params, dict):
            params = {}

        vsh_gr, _ = vsh_from_gamma(df, params)
        if vsh_gr is not None:
            df["VSH_GR"] = pd.to_numeric(vsh_gr, errors="coerce")

        vsh_nd = vsh_from_nd_triangle(df, params)
        if vsh_nd is not None:
            df["VSH_ND"] = pd.to_numeric(vsh_nd, errors="coerce")

        vsh_dt = vsh_from_dt_triangle(df, params)
        if vsh_dt is not None:
            df["VSH_DT"] = pd.to_numeric(vsh_dt, errors="coerce")

        base_cols = [c for c in ["VSH_GR", "VSH_ND", "VSH_DT"] if c in df.columns]
        good_cols = []
        for c in base_cols:
            arr = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
            if np.isfinite(arr).any():
                good_cols.append(c)

        if len(good_cols) == 0:
            hl = np.full(len(df), np.nan, dtype=float)
        elif len(good_cols) == 1:
            hl = pd.to_numeric(df[good_cols[0]], errors="coerce").to_numpy(dtype=float)
        elif len(good_cols) == 2:
            a = pd.to_numeric(df[good_cols[0]], errors="coerce").to_numpy(dtype=float)
            b = pd.to_numeric(df[good_cols[1]], errors="coerce").to_numpy(dtype=float)
            hl = 0.5 * (a + b)
        else:
            a, b, c = (pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float) for col in good_cols[:3])
            P = np.stack([0.5 * (a + b), 0.5 * (a + c), 0.5 * (b + c)], axis=0)
            hl = np.nanmedian(P, axis=0)

        hl = np.clip(pd.to_numeric(hl, errors="coerce"), 0.0, 1.0)
        if "VSH_GR" in df.columns:
            hl = pd.Series(hl, index=df.index).fillna(pd.to_numeric(df["VSH_GR"], errors="coerce")).to_numpy(dtype=float)

        df["VSH_HL"] = hl
        state.analysis_df = df

        dfV = getattr(state, "analysis_df_view", None)
        if dfV is not None and not dfV.empty:
            for c in [c for c in df.columns if "VSH" in c.upper()]:
                dfV[c] = pd.to_numeric(df[c], errors="coerce").reindex(dfV.index)
            state.analysis_df_view = dfV

        self._recompute_swb_only()

    def _on_compute_vsh_clicked(self):
        self._recompute_vsh_only()
        self._safe_rebuild_view()
        self._safe_refresh()
        try:
            self._refresh_vsh_depth_plot()
        except Exception:
            pass






    def _on_auto_fit_vsh_clicked(self):
        import numpy as np
        import pandas as pd
    
        try:
            from scipy.optimize import minimize
        except Exception as e:
            print(f"[VSH AUTO] scipy not available: {e}")
            return
    
        state = self._state()
        if state is None:
            return
    
        df = getattr(state, "analysis_df", None)
        if df is None or df.empty:
            return
    
        params = getattr(state, "params", {}) or {}
        if not isinstance(params, dict):
            params = {}
    
        # Make sure SWB exists if possible
        self._recompute_swb_only()
    
        # Refresh df in case _recompute_swb_only modified it
        df = getattr(state, "analysis_df", None)
        if df is None or df.empty:
            return
    
        swb = pd.to_numeric(df["SWB"], errors="coerce").to_numpy(float) if "SWB" in df.columns else None
    
        # -----------------------------
        # Initial values / references
        # -----------------------------
        x0 = np.array([
            float(params.get("gr_clean", 10.0)),
            float(params.get("gr_shale", 250.0)),
            float(params.get("neut_shale", 0.4)),
            float(params.get("dt_shale", 85.0)),
        ], dtype=float)
    
        # Reference values for regularization
        x_ref = x0.copy()
    
        # Characteristic scales for parameter movement
        x_scale = np.array([
            10.0,   # gr_clean
            30.0,   # gr_shale
            0.06,   # neut_shale
            10.0,   # dt_shale
        ], dtype=float)
    
        # Tighter, more geological bounds
        bounds = [
            (5.0, 50.0),    # gr_clean
            (100.0, 250.0),  # gr_shale
            (0.30, 0.45),    # neut_shale
            (75.0, 110.0),   # dt_shale
        ]
    
        def nan_mse_masked(a, b, mask):
            a = np.asarray(a, dtype=float)
            b = np.asarray(b, dtype=float)
            mask = np.asarray(mask, dtype=bool)
            m = np.isfinite(a) & np.isfinite(b) & mask
            if not np.any(m):
                return 0.0
            return float(np.nanmean((a[m] - b[m]) ** 2))
    
        def build_hl(vsh_gr, vsh_nd, vsh_dt):
            curves = []
            if vsh_gr is not None:
                curves.append(np.asarray(vsh_gr, dtype=float))
            if vsh_nd is not None:
                curves.append(np.asarray(vsh_nd, dtype=float))
            if vsh_dt is not None:
                curves.append(np.asarray(vsh_dt, dtype=float))
    
            if len(curves) == 0:
                return None
            if len(curves) == 1:
                hl = curves[0]
            elif len(curves) == 2:
                hl = 0.5 * (curves[0] + curves[1])
            else:
                P = np.stack([
                    0.5 * (curves[0] + curves[1]),
                    0.5 * (curves[0] + curves[2]),
                    0.5 * (curves[1] + curves[2]),
                ], axis=0)
                hl = np.nanmedian(P, axis=0)
    
            return np.clip(hl, 0.0, 1.0)
    
        def loss_fun(x):
            gr_clean, gr_shale, neut_shale, dt_shale = x
    
            p = dict(params)
            p["gr_clean"] = float(gr_clean)
            p["gr_shale"] = float(gr_shale)
            p["neut_shale"] = float(neut_shale)
            p["dt_shale"] = float(dt_shale)
    
            vsh_gr, _ = vsh_from_gamma(df, p)
            vsh_nd = vsh_from_nd_triangle(df, p)
            vsh_dt = vsh_from_dt_triangle(df, p)
    
            if (vsh_gr is None) and (vsh_nd is None) and (vsh_dt is None):
                return 1e9
    
            hl = build_hl(vsh_gr, vsh_nd, vsh_dt)
            if hl is None:
                return 1e9
 
    
 
            # -----------------------------------
            # Shale-only mask
            # -----------------------------------
            shale_mask = np.ones(len(df), dtype=bool)
            
            swb_min_fit = 0.12
            swb_max_fit = 0.50
            
            if swb is not None:
                shale_mask = np.isfinite(swb) & (swb > swb_min_fit) & (swb < swb_max_fit)
            
            # If too few samples, fall back to general finite area
            if np.sum(shale_mask) < 20:
                print("[VSH AUTO] Too few points in SWB fit mask; falling back to finite HL samples")
                shale_mask = np.isfinite(hl)    
             
                
            loss = 0.0
    
            # -----------------------------------
            # Internal consistency between Vsh tools
            # -----------------------------------
            if vsh_gr is not None and vsh_nd is not None:
                loss += 1.0 * nan_mse_masked(vsh_gr, vsh_nd, shale_mask)
            if vsh_gr is not None and vsh_dt is not None:
                loss += 1.0 * nan_mse_masked(vsh_gr, vsh_dt, shale_mask)
            if vsh_nd is not None and vsh_dt is not None:
                loss += 1.0 * nan_mse_masked(vsh_nd, vsh_dt, shale_mask)
    
            # -----------------------------------
            # Soft tie of HL to SWB
            # -----------------------------------
            if swb is not None:
                loss += 0.35 * nan_mse_masked(hl, swb, shale_mask)
    
            # -----------------------------------
            # Regularization toward reasonable initial values
            # -----------------------------------
            reg = np.sum(((x - x_ref) / x_scale) ** 2)
            loss += 0.50 * reg
    
            # -----------------------------------
            # Geological sanity penalties
            # -----------------------------------
            # Keep shale GR clearly above clean GR
            if gr_shale <= gr_clean + 40.0:
                loss += 25.0
    
            # Mild penalty if HL mean in shale interval is way off SWB mean
            if swb is not None:
                m = np.isfinite(hl) & np.isfinite(swb) & shale_mask
                if np.any(m):
                    hl_mean = float(np.nanmean(hl[m]))
                    swb_mean = float(np.nanmean(swb[m]))
                    loss += 0.5 * (hl_mean - swb_mean) ** 2
    
            return float(loss)
    
        result = minimize(
            loss_fun,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 200},
        )
    
        if not result.success:
            print(f"[VSH AUTO] Optimization did not fully converge: {result.message}")
    
        gr_clean, gr_shale, neut_shale, dt_shale = result.x
    
        # Save optimized parameters
        self._set_param_and_refresh("gr_clean", float(gr_clean))
        self._set_param_and_refresh("gr_shale", float(gr_shale))
        self._set_param_and_refresh("neut_shale", float(neut_shale))
        self._set_param_and_refresh("dt_shale", float(dt_shale))
    
        # Recompute with fitted params
        self._recompute_vsh_only()
        self._safe_rebuild_view()
        self._safe_refresh()
        try:
            self._refresh_vsh_depth_plot()
        except Exception:
            pass
    
        # Optional diagnostics
        pbest = dict(params)
        pbest["gr_clean"] = float(gr_clean)
        pbest["gr_shale"] = float(gr_shale)
        pbest["neut_shale"] = float(neut_shale)
        pbest["dt_shale"] = float(dt_shale)
    
        vsh_gr, _ = vsh_from_gamma(df, pbest)
        vsh_nd = vsh_from_nd_triangle(df, pbest)
        vsh_dt = vsh_from_dt_triangle(df, pbest)
        hl = build_hl(vsh_gr, vsh_nd, vsh_dt)
    
        def safe_mean(x):
            if x is None:
                return np.nan
            x = np.asarray(x, dtype=float)
            return float(np.nanmean(x)) if np.isfinite(x).any() else np.nan
    
        print(
            "[VSH AUTO] Best fit -> "
            f"gr_clean={gr_clean:.2f}, "
            f"gr_shale={gr_shale:.2f}, "
            f"neut_shale={neut_shale:.3f}, "
            f"dt_shale={dt_shale:.2f}"
        )
        print(
            "[VSH AUTO] Means -> "
            f"VSH_GR={safe_mean(vsh_gr):.3f}, "
            f"VSH_ND={safe_mean(vsh_nd):.3f}, "
            f"VSH_DT={safe_mean(vsh_dt):.3f}, "
            f"VSH_HL={safe_mean(hl):.3f}, "
            f"SWB={safe_mean(swb):.3f}"
        )
    


  
    def _on_compute_sw_clicked(self):
        w = self.focusWidget()
        if w is not None:
            try:
                w.clearFocus()
            except Exception:
                pass
    
        state = self._state()
        if state is None:
            return
    
        df = getattr(state, "analysis_df", None)
        if df is None or df.empty:
            return
    
        params = getattr(state, "params", {}) or {}
        if not isinstance(params, dict):
            params = {}
    
        # --- Resolve Rt
        rt_col = params.get("rt_curve")
        if (not rt_col) or (rt_col not in df.columns):
            rt_col = _first_present(df.columns, ["AT90", "AF90", "AO90", "ILD", "RT"])
    
        # --- Resolve PHIT
        phit_col = params.get("phit_curve")
        if (not phit_col) or (phit_col not in df.columns):
            phit_col = "PHIT" if "PHIT" in df.columns else _first_present(
                df.columns, ["PHIT_CHART", "PHIT_NMR", "TCMR", "MPHIS", "MPHI"]
            )
    
        if (rt_col is None) or (phit_col is None):
            print("[SW] Need Rt + PHIT to compute SW/BVW/M*")
            return
    
        # --- Optional curves
        qv_col = "Qv" if "Qv" in df.columns else ("QV" if "QV" in df.columns else None)
        vsh_col = "VSH_HL" if "VSH_HL" in df.columns else None
    
        # --- Scalar parameters
        m_cem  = self._get_sw("sw.m_cem")
        n_sat  = self._get_sw("sw.n_sat")
        rw     = self._get_sw("sw.rw")
        mslope = self._get_sw("sw.mslope")
        T_F    = self._get_sw("sw.T_F")
    
        # --- Compute scalar B from scalar temperature and scalar Rw
        B = float(compute_bdacy(T_F, rw))
    
        # save it so the rest of the app sees the current value
        params["sw.B"] = B
        state.params["sw.B"] = B
        self.param_store.set("sw.B", B)
    
        print(f"[SW] Using B = {B:.4f} from T_F={T_F:.1f} F and Rw={rw:.5f}")
    
        out = compute_sw_products(
            df=df,
            rt_col=rt_col,
            phit_col=phit_col,
            qv_col=qv_col,
            vsh_col=vsh_col,
            m_cem=m_cem,
            n_sat=n_sat,
            rw=rw,
            mslope=mslope,
            B=B,
        )
    
        df["PHIT_USED"] = pd.to_numeric(df[phit_col], errors="coerce")
        df["BVW"] = out["BVW"]
        df["SWT"] = out["SWT"]
        df["BVO"] = out["BVO"]
        df["MSTAR_APP"] = out["MSTAR_APP"]
        df["MSTAR_FIT"] = out["MSTAR"]
    
        # optional: save scalar B as a constant column for inspection/export
        df["Bdacy"] = B
    
        state.analysis_df = df
        self._safe_rebuild_view()
        self._safe_refresh()
    



    def _on_compute_ws_clicked(self):
        w = self.focusWidget()
        if w is not None:
            try:
                w.clearFocus()
            except Exception:
                pass
    
        state = self._state()
        if state is None:
            return
    
        df = getattr(state, "analysis_df", None)
        if df is None or df.empty:
            return
    
        params = getattr(state, "params", {}) or {}
        if not isinstance(params, dict):
            params = {}
    
        # --- Resolve Rt
        rt_col = params.get("rt_curve")
        if (not rt_col) or (rt_col not in df.columns):
            rt_col = _first_present(df.columns, ["AT90", "AF90", "AO90", "ILD", "RT"])
        if rt_col is None:
            print("[WS] No Rt curve found")
            return
    
        # --- Resolve PHIT
        phit_col = params.get("phit_curve")
        if (not phit_col) or (phit_col not in df.columns):
            phit_col = "PHIT" if "PHIT" in df.columns else _first_present(
                df.columns, ["PHIT_CHART", "PHIT_NMR", "TCMR", "MPHIS", "MPHI"]
            )
        if phit_col is None:
            print("[WS] No PHIT curve found")
            return
    
        if phit_col != "PHIT":
            df["PHIT"] = pd.to_numeric(df[phit_col], errors="coerce")
            phit_col = "PHIT"
    
        # --- Need VSH_HL
        vsh_col = "VSH_HL" if "VSH_HL" in df.columns else None
        if vsh_col is None:
            print("[WS] Need VSH_HL first")
            return
    
        # --- Scalar parameters
        m_cem = self._get_sw("sw.m_cem")
        n_sat = self._get_sw("sw.n_sat")
        rw = self._get_sw("sw.rw")
        mslope = self._get_sw("sw.mslope")
        cbw_intercept = self._get_sw("sw.cbw_intercept")
        T_F = self._get_sw("sw.T_F")
        den_fl = self._get_sw("sw.den_fl")
    
        # --- Scalar Rw75, SAL, and B
        rw = max(float(rw), 1e-6)
        Rw75 = ((T_F + 6.77) * rw) / (75.0 + 6.77)
        SAL = float(compute_sal_kppm_from_rw75(Rw75))
        B = float(compute_bdacy(T_F, rw))
    
        # store these as constant columns for display/export
        df["Rw75"] = Rw75
        df["SAL"] = SAL
        df["Bdacy"] = B
    
        # store scalar B in params
        params["sw.B"] = B
        state.params["sw.B"] = B
        self.param_store.set("sw.B", B)
    
        print(f"[WS] Using scalar values: Rw={rw:.5f}, Rw75={Rw75:.5f}, SAL={SAL:.5f}, B={B:.5f}")
    
        # --- Build CBWapp from Vsh
        Vsh = pd.to_numeric(df[vsh_col], errors="coerce").to_numpy(float)
        Vsh = np.clip(Vsh, 0.0, 1.0)
        df["CBWapp"] = np.clip(Vsh * cbw_intercept, 0.0, 1.0)
    
        # --- PHIT and Swb
        PHIT = pd.to_numeric(df[phit_col], errors="coerce").to_numpy(dtype=float)
        PHIT = np.clip(PHIT, 0.0, None)
        PHIT_safe = np.clip(PHIT, 1e-6, None)
    
        cbw = pd.to_numeric(df["CBWapp"], errors="coerce").to_numpy(dtype=float)
        cbw = np.clip(cbw, 0.0, None)
    
        Swb = np.clip(cbw / PHIT_safe, 0.0, 1.0)
        df["Swb"] = Swb
        df["SWB"] = Swb
    
        # --- Qv from scalar SAL
        denom = (0.6425 / np.sqrt(max(den_fl * SAL, 1e-12)) + 0.22)
        df["Qv"] = np.clip(Swb / denom, 0.0, 5.0)
        qv_col = "Qv"
    
        # --- Save constants file
        write_constants_to_file(
            "./apps/merge_gui/data/Pickett.txt",
            m_cem=m_cem,
            n_sat=n_sat,
            Rw=rw,
            mslope=mslope,
            Bdacy=B,
        )
    
        # --- Arrays for solver
        Rt = pd.to_numeric(df[rt_col], errors="coerce").to_numpy(float)
        Qv = pd.to_numeric(df[qv_col], errors="coerce").to_numpy(float)
    
        MSTAR = Vsh * mslope + m_cem
        df["MSTAR"] = MSTAR
    
        try:
            from petrocore.workflow.waxman_smits import waxman_smits_sw_iterative
    
            Sw_cp = waxman_smits_sw_iterative(
                rt=Rt,
                phit=PHIT,
                qv=Qv,
                rw=rw,          # scalar
                m=MSTAR,        # array
                n=n_sat,
                B=B,            # scalar
                max_iter=60,
                tol=1e-6,
                sw0=None,
            )
            Sw_cp = np.clip(np.asarray(Sw_cp, dtype=float), 1e-4, 1.0)
    
        except Exception as e:
            print(f"[WS] iterative solver unavailable, using fallback: {e}")
            good = np.isfinite(Rt) & np.isfinite(PHIT) & (Rt > 0) & (PHIT > 0)
            Sw_cp = np.full(len(df), np.nan, dtype=float)
            Sw_cp[good] = ((rw / Rt[good]) * (1.0 / np.clip(PHIT[good] ** MSTAR[good], 1e-12, None))) ** (1.0 / n_sat)
            Sw_cp = np.clip(Sw_cp, 1e-4, 1.0)
    
        # --- Outputs
        df["SW_CP"] = Sw_cp
        df["BVWT_CP"] = PHIT * Sw_cp
        df["BVWe_CP"] = np.clip(PHIT * Sw_cp - cbw, 0.0, None)
    
        PHIE = np.clip(PHIT - cbw, 0.0, None)
        PHIE = np.minimum(PHIE, PHIT)
        df["PHIE"] = PHIE
        df["PHIT"] = PHIT
    
        state.analysis_df = df
        self._safe_rebuild_view()
        self._safe_refresh()
    
        try:
            self._refresh_final_depth_plot()
        except Exception:
            pass
    
    
     
    







    def _on_final_zone_plot_clicked(self):
        state = self._state()
        if state is None:
            return

        df = getattr(state, "analysis_df", None)
        if df is None or df.empty:
            print("[ZonePlot] No analysis_df")
            return

        units = getattr(state, "units_map", {}) or {}
        top = getattr(state, "depth_top", None)
        base = getattr(state, "depth_base", None)

        if top is None or base is None:
            print("[ZonePlot] No depth window set")
            return

        try:
            from petrocore.viz.zone_template_plot import launch_zone_plot

            title = getattr(state, "well_name", "Zone Plot")
            launch_zone_plot(
                df,
                units,
                top,
                base,
                title=title,
                depth_col="DEPT",
                show=True,
            )
        except Exception as e:
            print(f"[ZonePlot] failed: {e}")




    # -------------------------------------------------------------------------
    # Tab builders
    # -------------------------------------------------------------------------
    def _build_crossplots_tab(self):
        w = QWidget()
        lay = QHBoxLayout(w)

        plots = QWidget()
        vlay = QVBoxLayout(plots)
        vlay.setContentsMargins(0, 0, 0, 0)
        vlay.setSpacing(6)
        vlay.addWidget(self.canvas_nd, stretch=1)
        vlay.addWidget(self.canvas_dt, stretch=1)
        vlay.addWidget(self.canvas_gr, stretch=1)

        if self.vsh_depth_plot is None:
            self.vsh_depth_plot = pg.PlotWidget()
            self.vsh_depth_plot.invertY(True)
            self.vsh_depth_plot.showGrid(x=True, y=True, alpha=0.5)
            self.vsh_depth_plot.setLabel("bottom", "Vsh")
            self.vsh_depth_plot.setLabel("left", "Depth")
            try:
                self.vsh_depth_plot.addLegend(offset=(10, 10))
            except Exception:
                pass

        split = QSplitter(Qt.Horizontal)
        split.addWidget(plots)
        split.addWidget(self.vsh_depth_plot)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)

        lay.addWidget(split, stretch=5)

        ctrl = QWidget()
        form = QFormLayout(ctrl)
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        lay.addWidget(ctrl, stretch=1)

        btn_vsh = QPushButton("2) Compute Vsh HL")
        btn_vsh.clicked.connect(self._on_compute_vsh_clicked)
        form.addRow(btn_vsh)

        for key in ("neut_matrix", "neut_shale", "dt_matrix", "dt_shale"):
            spec = VSH_PARAM_SPEC[key]
            val = float(self.param_store.get(key, spec["default"]))
            row = FloatSliderRow(spec["mn"], spec["mx"], spec["step"], spec["dec"], val)
            row.valueChangedFloat.connect(lambda v, kk=key: self._set_param_and_refresh(kk, float(v)))
            form.addRow(spec["label"], row)

        gr_clean = float(self.param_store.get("gr_clean", 25))
        gr_shale = float(self.param_store.get("gr_shale", 160))

        row_clean = FloatSliderRow(0, 300, 1, 0, gr_clean, label_width=40)
        row_shale = FloatSliderRow(0, 300, 1, 0, gr_shale, label_width=40)

        row_clean.valueChangedFloat.connect(lambda v: self._set_param_and_refresh("gr_clean", float(v)))
        row_shale.valueChangedFloat.connect(lambda v: self._set_param_and_refresh("gr_shale", float(v)))

        form.addRow("GR cl", row_clean)
        form.addRow("GR sh", row_shale)



        btn_auto_vsh = QPushButton("Assisted Fit Vsh Endpoints")
        btn_auto_vsh.clicked.connect(self._on_auto_fit_vsh_clicked)
        form.addRow(btn_auto_vsh)



        QTimer.singleShot(0, self._refresh_vsh_depth_plot)
        return w


    def _build_sw_tab(self):
        m_cem = self._get_sw("sw.m_cem")
        n_sat = self._get_sw("sw.n_sat")
        rw = self._get_sw("sw.rw")
        mslope = self._get_sw("sw.mslope")
        cbw_int = self._get_sw("sw.cbw_intercept")
        T_F = self._get_sw("sw.T_F")
    
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.addWidget(self.canvas_sw, stretch=4)
    
        ctrl = QWidget()
        form = QFormLayout(ctrl)
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        lay.addWidget(ctrl, stretch=1)
    
        note = QLabel(
            "<b>Parameter Controls</b><br>"
            "• Drag slider for coarse adjustment<br>"
            "• Two-finger scroll for fine adjustment<br>"
            "• SHIFT + scroll for ultra-fine adjustment<br>"
            "• Double-click slider or value to enter exact number"
        )
        note.setStyleSheet("""
            QLabel {
                color: red;
                font-size: 11px;
                background-color: #f8f8f8;
                border: 1px solid #dddddd;
                border-radius: 4px;
                padding: 8px;
            }
        """)
        note.setWordWrap(True)
        form.addRow(note)
    
        row_m = FloatSliderRow(1.00, 3.00, 0.01, 2, m_cem)
        row_m.valueChangedFloat.connect(lambda v: self._set_param_and_refresh("sw.m_cem", float(v)))
        form.addRow("m (cementation)", row_m)
    
        row_n = FloatSliderRow(1.00, 3.00, 0.01, 2, n_sat)
        row_n.valueChangedFloat.connect(lambda v: self._set_param_and_refresh("sw.n_sat", float(v)))
        form.addRow("n (saturation)", row_n)
    
        row_rw = FloatSliderRow(0.001, 0.20, 0.001, 4, rw)
        row_rw.valueChangedFloat.connect(lambda v: self._set_param_and_refresh("sw.rw", float(v)))
        form.addRow("Rw", row_rw)
    
        row_tf = FloatSliderRow(60.0, 300.0, 1.0, 1, T_F)
        row_tf.valueChangedFloat.connect(lambda v: self._set_param_and_refresh("sw.T_F", float(v)))
        form.addRow("Temp (F)", row_tf)
    
        row_ms = FloatSliderRow(0.01, 4.00, 0.01, 2, mslope)
        row_ms.valueChangedFloat.connect(lambda v: self._set_param_and_refresh("sw.mslope", float(v)))
        form.addRow("M* slope", row_ms)
    
        row_cbw = FloatSliderRow(0.0, 0.5, 0.01, 3, cbw_int)
        row_cbw.valueChangedFloat.connect(lambda v: self._set_param_and_refresh("sw.cbw_intercept", float(v)))
        form.addRow("CBW Intercept", row_cbw)
    
        btn_sw = QPushButton("3) Compute General Sw / BVW / M*")
        btn_sw.clicked.connect(self._on_compute_sw_clicked)
        btn_sw.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        form.addRow(btn_sw)
    
        btn_ws = QPushButton("4) Compute Waxman–Smits Sw")
        btn_ws.clicked.connect(self._on_compute_ws_clicked)
        btn_ws.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        form.addRow(btn_ws)
    
        btn_export = QPushButton("Export DF")
        btn_export.clicked.connect(self._on_export_df_clicked)
        btn_export.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        form.addRow(btn_export)
    
        btn_zone = QPushButton("Final Zone Plot")
        btn_zone.clicked.connect(self._on_final_zone_plot_clicked)
        btn_zone.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        form.addRow(btn_zone)
    
        return w
    



    def _build_final_depth_tab(self):
        w = QWidget()
        lay = QHBoxLayout(w)
        self.final_tracks = pg.GraphicsLayoutWidget()
        lay.addWidget(self.final_tracks, stretch=5)
        self._build_final_tracks()
        QTimer.singleShot(0, self._refresh_final_depth_plot)
        return w

    # -------------------------------------------------------------------------
    # VSH side depth plot
    # -------------------------------------------------------------------------
    def _refresh_vsh_depth_plot(self):
        p = self.vsh_depth_plot
        if p is None:
            return

        df = self._get_df_view_or_full()
        if df is None or df.empty:
            p.clear()
            return

        depth = self._depth_array(df)
        good = np.isfinite(depth)
        if not good.any():
            p.clear()
            p.setTitle("VSH: depth not numeric")
            return

        d = depth[good].astype(float, copy=True)

        vsh_cols = [c for c in df.columns if "VSH" in c.upper()]
        if "VSH_HL" in vsh_cols:
            vsh_cols = [c for c in vsh_cols if c != "VSH_HL"] + ["VSH_HL"]

        p.clear()
        p.showGrid(x=True, y=True, alpha=0.35)

        pen_map = {
            "VSH_GR": pg.mkPen("g", width=2),
            "VSH_ND": pg.mkPen("w", width=2),
            "VSH_DT": pg.mkPen("b", width=2),
            "VSH_HL": pg.mkPen("r", width=6),
        }

        any_plotted = False

        swb_col = next((c for c in df.columns if c.strip().upper() == "SWB"), None)
        if swb_col is not None:
            swb_all = pd.to_numeric(df[swb_col], errors="coerce").to_numpy(dtype=float)
            swb = swb_all[good]
            m = np.isfinite(swb) & np.isfinite(d)
            if m.any():
                p.plot(
                    np.ascontiguousarray(swb[m], dtype=np.float64),
                    np.ascontiguousarray(d[m], dtype=np.float64),
                    name=swb_col,
                    pen=pg.mkPen("y", width=2),
                )
                any_plotted = True

        for col in vsh_cols:
            x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)[good]
            m = np.isfinite(x) & np.isfinite(d)
            if m.any():
                p.plot(x[m], d[m], name=col, pen=pen_map.get(col, pg.mkPen(width=2)))
                any_plotted = True

        if any_plotted:
            p.setXRange(0.0, 1.0, padding=0.02)
            p.setYRange(float(np.nanmin(d)), float(np.nanmax(d)), padding=0.0)
            p.invertY(True)

    # -------------------------------------------------------------------------
    # Final tracks
    # -------------------------------------------------------------------------
    def _build_final_tracks(self):
        self.final_tracks.clear()

        gr = self.final_tracks.addPlot(row=0, col=0, title="GR/CGR")
        por = self.final_tracks.addPlot(row=0, col=1, title="Porosity")
        rt = self.final_tracks.addPlot(row=0, col=2, title="Rt")
        nmr = self.final_tracks.addPlot(row=0, col=3, title="NMR")
        bvw = self.final_tracks.addPlot(row=0, col=4, title="BVW / WS")

        por.setYLink(gr)
        rt.setYLink(gr)
        nmr.setYLink(gr)
        bvw.setYLink(gr)

        for p in (gr, por, rt, nmr, bvw):
            p.invertY(True)
            p.showGrid(x=True, y=True, alpha=0.6)

        self.final_track_items = {"gr": gr, "por": por, "rt": rt, "nmr": nmr, "bvw": bvw}

    def _plot_track_curve(self, items, track_key, depth, x, pen, xlim=None, invert_x=False, logx=False, name=None):
        if x is None:
            return
        m = np.isfinite(x) & np.isfinite(depth)
        if logx:
            m &= (x > 0)
        if not np.any(m):
            return
        pi = items[track_key]
        pi.setLogMode(x=bool(logx), y=False)
        pi.getViewBox().invertX(invert_x)
        pi.plot(x[m], depth[m], pen=pen, name=name)
        if xlim is not None:
            pi.setXRange(xlim[0], xlim[1], padding=0.0)

    def _refresh_final_depth_plot(self):
        state = self._state()
        if state is None:
            return

        items = self.final_track_items
        if not isinstance(items, dict) or not items:
            return
        for pi in items.values():
            pi.clear()

        df = self._get_df_view_or_full()
        if df is None or df.empty:
            return

        params = getattr(state, "params", {}) or {}
        if not isinstance(params, dict):
            params = {}

        depth = self._depth_array(df)

        def arr(name):
            return get_col(df, name)

        gr_curve = self._get_curve_name(df, params, "gr_curve", fallbacks=["HSGR", "GR", "SGR", "GR_EDTC", "HGR", "EGR"])
        cgr_curve = self._get_curve_name(df, params, "cgr_curve", fallbacks=["HCGR", "CGR", "ECGR", "GR_EDTC"])
        tnph_curve = self._get_curve_name(df, params, "tnph_curve", fallbacks=["TNPH", "NPOR", "NPHI", "CNL"])
        rhob_curve = self._get_curve_name(df, params, "rhob_curve", fallbacks=["RHOB", "RHOZ"])
        rt_curve = self._get_curve_name(df, params, "rt_curve", fallbacks=["AT90", "AF90", "AO90", "ILD", "RT"])
        phit_nmr_curve = self._get_curve_name(df, params, "tcmr_curve", fallbacks=["PHIT_NMR", "TCMR", "MPHS"])
        phie_nmr_curve = self._get_curve_name(df, params, "cmrp_curve", fallbacks=["PHIE_NMR", "CMRP_3MS", "CMRP3MS", "CMRP", "MPHI"])
        bvie_curve = self._get_curve_name(df, params, "bvie_curve", fallbacks=["BVIE", "BVI_E", "BVI", "MBVI"])

        # GR/CGR
        pi_gr = items["gr"]
        x_gr = arr(gr_curve)
        x_cgr = arr(cgr_curve)
        if (x_gr is not None) and (x_cgr is not None):
            m = np.isfinite(depth) & np.isfinite(x_gr) & np.isfinite(x_cgr)
            if np.any(m):
                d = depth[m].astype(float)
                grv = np.clip(x_gr[m].astype(float), 0.0, 200.0)
                cgrv = np.clip(x_cgr[m].astype(float), 0.0, 200.0)
                c_gr = pi_gr.plot(grv, d, pen=pg.mkPen("m", width=2), name="GR")
                c_cgr = pi_gr.plot(cgrv, d, pen=pg.mkPen("g", width=2), name="CGR")
                zerogr = np.zeros_like(grv)
                c_zero = pi_gr.plot(zerogr, d, pen=pg.mkPen("w", width=1), name="0")
                pi_gr.addItem(pg.FillBetweenItem(c_gr, c_cgr, brush=pg.mkBrush("magenta")))
                pi_gr.addItem(pg.FillBetweenItem(c_zero, c_cgr, brush=pg.mkBrush("g")))
                pi_gr.setXRange(0, 200, padding=0.0)
        else:
            self._plot_track_curve(items, "gr", depth, x_gr, pg.mkPen("m", width=2), xlim=(0, 200), name="GR")
            self._plot_track_curve(items, "gr", depth, x_cgr, pg.mkPen("g", width=2), xlim=(0, 200), name="CGR")

        vsh = get_col(df, "VSH_HL")
        if vsh is not None:
            self._plot_track_curve(items, "gr", depth, vsh * 200.0, pg.mkPen("brown", width=6), xlim=(0, 200), name="VSH_HL*200")
        if getattr(pi_gr, "legend", None) is None:
            pi_gr.addLegend(offset=(10, 10))

        # Porosity
        self._plot_track_curve(items, "por", depth, arr(tnph_curve), pg.mkPen("g", width=1), xlim=(-0.15, 0.45), invert_x=True, name=tnph_curve)
        self._plot_track_curve(items, "por", depth, arr(phit_nmr_curve), pg.mkPen("k", width=1), xlim=(-0.15, 0.45), invert_x=True, name=phit_nmr_curve)
        self._plot_track_curve(items, "por", depth, arr(phie_nmr_curve), pg.mkPen("b", width=1), xlim=(-0.15, 0.45), invert_x=True, name=phie_nmr_curve)

        x_phit_chart = get_col(df, "PHIT_CHART")
        if x_phit_chart is not None:
            self._plot_track_curve(items, "por", depth, x_phit_chart, pg.mkPen("y", width=2), xlim=(-0.15, 0.45), invert_x=True, name="PHIT_CHART")

        rhob_x = arr(rhob_curve)
        if rhob_x is not None:
            rho_ma = 2.71
            rho_fl = 1.10
            phid = np.clip((rho_ma - rhob_x) / (rho_ma - rho_fl), -0.15, 0.45)
            self._plot_track_curve(items, "por", depth, phid, pg.mkPen("r", width=2), xlim=(-0.15, 0.45), invert_x=True, name="PHID")

        if getattr(items["por"], "legend", None) is None:
            items["por"].addLegend(offset=(10, 10))

        # Rt
        self._plot_track_curve(items, "rt", depth, arr(rt_curve), pg.mkPen("lightgray", width=2, style=Qt.DashLine), xlim=(0.2, 20), logx=False, name=rt_curve)
        if getattr(items["rt"], "legend", None) is None:
            items["rt"].addLegend(offset=(10, 10))

        # NMR fills
        pi_nmr = items["nmr"]
        x_phit = arr(phit_nmr_curve)
        x_phie = arr(phie_nmr_curve)
        x_bvie = arr(bvie_curve)
        if (x_phit is not None) and (x_phie is not None) and (x_bvie is not None):
            m = np.isfinite(depth) & np.isfinite(x_phit) & np.isfinite(x_phie) & np.isfinite(x_bvie)
            if np.any(m):
                d = depth[m].astype(float)
                phit = np.clip(x_phit[m].astype(float), 0.0, 0.30)
                phie = np.minimum(np.clip(x_phie[m].astype(float), 0.0, 0.30), phit)
                bvie = np.minimum(np.clip(x_bvie[m].astype(float), 0.0, 0.30), phie)

                c_phit = pi_nmr.plot(phit, d, pen=pg.mkPen("gray", width=1), name="PHIT_NMR")
                c_phie = pi_nmr.plot(phie, d, pen=pg.mkPen("y", width=1), name="PHIE_NMR")
                c_bvie = pi_nmr.plot(bvie, d, pen=pg.mkPen("b", width=1), name="BVIE")
                zero = np.zeros_like(bvie)
                c_zero = pi_nmr.plot(zero, d, pen=pg.mkPen("w", width=1), name="0")

                pi_nmr.addItem(pg.FillBetweenItem(c_phit, c_phie, brush=pg.mkBrush("gray")))
                pi_nmr.addItem(pg.FillBetweenItem(c_phie, c_bvie, brush=pg.mkBrush("y")))
                pi_nmr.addItem(pg.FillBetweenItem(c_bvie, c_zero, brush=pg.mkBrush("b")))
                pi_nmr.getViewBox().invertX(True)
                pi_nmr.setXRange(0.0, 0.30, padding=0.0)

        if getattr(pi_nmr, "legend", None) is None:
            pi_nmr.addLegend(offset=(10, 10))

        # BVW / WS
        pi_bvw = items["bvw"]
        x_phit = get_col(df, "PHIT")
        x_phie = get_col(df, "PHIE")
        x_bvwe = get_col(df, "BVWe_CP")
        if (x_phit is not None) and (x_phie is not None) and (x_bvwe is not None):
            m = np.isfinite(depth) & np.isfinite(x_phit) & np.isfinite(x_phie) & np.isfinite(x_bvwe)
            if np.any(m):
                d = depth[m].astype(float)
                phit = np.clip(x_phit[m].astype(float), 0.0, 0.30)
                phie = np.minimum(np.clip(x_phie[m].astype(float), 0.0, 0.30), phit)
                bvwe = np.minimum(np.clip(x_bvwe[m].astype(float), 0.0, 0.30), phie)

                c_phit = pi_bvw.plot(phit, d, pen=pg.mkPen("yellow", width=1), name="PHIT")
                c_phie = pi_bvw.plot(phie, d, pen=pg.mkPen("green", width=1), name="PHIE")
                c_bvwe = pi_bvw.plot(bvwe, d, pen=pg.mkPen("dodgerblue", width=1), name="BVWe_CP")
                zero = np.zeros_like(bvwe)
                c_zero = pi_bvw.plot(zero, d, pen=pg.mkPen("w", width=1), name="0")

                pi_bvw.addItem(pg.FillBetweenItem(c_phit, c_phie, brush=pg.mkBrush("gray")))
                pi_bvw.addItem(pg.FillBetweenItem(c_phie, c_bvwe, brush=pg.mkBrush("green")))
                pi_bvw.addItem(pg.FillBetweenItem(c_bvwe, c_zero, brush=pg.mkBrush("dodgerblue")))
                pi_bvw.getViewBox().invertX(True)
                pi_bvw.setXRange(0.0, 0.30, padding=0.0)
        '''
        swcp = get_col(df, "SW_CP")
        if swcp is not None:
            m = np.isfinite(swcp) & np.isfinite(depth)
            if m.any():
                d = depth[m].astype(float)
                sw_scaled = np.clip(swcp[m].astype(float), 0.0, 1.0) * 0.30
                pi_bvw.plot(sw_scaled, d, pen=pg.mkPen("c", width=3), name="SW_CP (x0.30)")'''

        if getattr(pi_bvw, "legend", None) is None:
            pi_bvw.addLegend(offset=(10, 10))

        # Tops
        tops_df = getattr(state, "tops_df", None)
        if tops_df is not None and len(tops_df) > 0:
            depth_col = None
            for c in ("Depth", "DEPT", "dept", "TopDepth", "Top_Depth", "MD", "TVD"):
                if c in tops_df.columns:
                    depth_col = c
                    break

            name_col = None
            for c in ("Top", "Name", "Horizon", "Marker", "Pick", "Formation"):
                if c in tops_df.columns:
                    name_col = c
                    break

            if depth_col is not None:
                ylo = float(np.nanmin(depth))
                yhi = float(np.nanmax(depth))

                for _, row in tops_df.iterrows():
                    dv = row.get(depth_col, None)
                    if dv is None or pd.isna(dv):
                        continue
                    try:
                        y = float(dv)
                    except Exception:
                        continue
                    if not (ylo <= y <= yhi):
                        continue

                    label = None
                    if name_col is not None:
                        vv = row.get(name_col, None)
                        if vv is not None and not pd.isna(vv):
                            label = str(vv)

                    for pi in items.values():
                        pi.addLine(y=y, pen=pg.mkPen("r", width=2))

                    if label:
                        gr_pi = items["gr"]
                        x_left = gr_pi.viewRange()[0][0]
                        t = pg.TextItem(label, anchor=(0, 1))
                        t.setPos(x_left, y)
                        gr_pi.addItem(t)

        ymin = float(np.nanmin(depth))
        ymax = float(np.nanmax(depth))
        for pi in items.values():
            pi.enableAutoRange(axis=pg.ViewBox.YAxis, enable=False)
            pi.setYRange(ymin, ymax, padding=0.0)

    # -------------------------------------------------------------------------
    # Main depth tracks
    # -------------------------------------------------------------------------
    def _build_tracks(self):
        self.tracks.clear()

        gr = self.tracks.addPlot(row=0, col=0, title="GR/CGR")
        por = self.tracks.addPlot(row=0, col=1, title="Porosity")
        rt = self.tracks.addPlot(row=0, col=2, title="Rt")
        nmr = self.tracks.addPlot(row=0, col=3, title="NMR")

        por.setYLink(gr)
        rt.setYLink(gr)
        nmr.setYLink(gr)

        for p in (gr, por, rt, nmr):
            p.invertY(True)
            p.showGrid(x=True, y=True, alpha=0.6)

        self.track_items = {"gr": gr, "por": por, "rt": rt, "nmr": nmr}

    def update_depth_plot(self, state):
        for pi in self.track_items.values():
            pi.clear()

        df = getattr(state, "analysis_df_view", None)
        if df is None or df.empty:
            df = getattr(state, "analysis_df", None)
        if df is None or df.empty:
            return

        params = getattr(state, "params", {}) or {}
        if not isinstance(params, dict):
            params = {}

        depth = self._depth_array(df)

        def arr(name):
            return get_col(df, name)

        gr_curve = self._get_curve_name(df, params, "gr_curve", fallbacks=["HSGR", "GR", "SGR", "GR_EDTC", "HGR", "EGR"])
        cgr_curve = self._get_curve_name(df, params, "cgr_curve", fallbacks=["HCGR", "CGR", "ECGR", "GR_EDTC"])
        tnph_curve = self._get_curve_name(df, params, "tnph_curve", fallbacks=["TNPH", "NPOR", "NPHI", "CNL"])
        rhob_curve = self._get_curve_name(df, params, "rhob_curve", fallbacks=["RHOB", "RHOZ"])
        rt_curve = self._get_curve_name(df, params, "rt_curve", fallbacks=["AT90", "AF90", "AO90", "ILD", "RT"])
        phit_nmr_curve = self._get_curve_name(df, params, "tcmr_curve", fallbacks=["PHIT_NMR", "TCMR", "MPHS"])
        phie_nmr_curve = self._get_curve_name(df, params, "cmrp_curve", fallbacks=["PHIE_NMR", "CMRP_3MS", "CMRP3MS", "CMRP", "MPHI"])
        bvie_curve = self._get_curve_name(df, params, "bvie_curve", fallbacks=["BVIE", "BVI_E", "BVI", "MBVI"])

        # GR/CGR fills
        pi_gr = self.track_items["gr"]
        x_gr = arr(gr_curve)
        x_cgr = arr(cgr_curve)

        if (x_gr is not None) and (x_cgr is not None):
            m = np.isfinite(depth) & np.isfinite(x_gr) & np.isfinite(x_cgr)
            if np.any(m):
                d = depth[m].astype(float)
                grv = np.clip(x_gr[m].astype(float), 0.0, 200.0)
                cgrv = np.clip(x_cgr[m].astype(float), 0.0, 200.0)

                c_gr = pi_gr.plot(grv, d, pen=pg.mkPen("m", width=2), name="GR")
                c_cgr = pi_gr.plot(cgrv, d, pen=pg.mkPen("g", width=2), name="CGR")
                zerogr = np.zeros_like(grv)
                c_zero = pi_gr.plot(zerogr, d, pen=pg.mkPen("w", width=1), name="0")

                pi_gr.addItem(pg.FillBetweenItem(c_gr, c_cgr, brush=pg.mkBrush("magenta")))
                pi_gr.addItem(pg.FillBetweenItem(c_zero, c_cgr, brush=pg.mkBrush("g")))
                pi_gr.setXRange(0, 200, padding=0.0)
        else:
            self._plot_track_curve(self.track_items, "gr", depth, x_gr, pg.mkPen("m", width=2), xlim=(0, 200), name="GR")
            self._plot_track_curve(self.track_items, "gr", depth, x_cgr, pg.mkPen("g", width=2), xlim=(0, 200), name="CGR")

        vsh = get_col(df, "VSH_HL")
        if vsh is not None:
            self._plot_track_curve(self.track_items, "gr", depth, vsh * 200.0, pg.mkPen("brown", width=6), xlim=(0, 200), name="VSH_HL*200")

        if getattr(pi_gr, "legend", None) is None:
            pi_gr.addLegend(offset=(10, 10))

        # Porosity overlays
        self._plot_track_curve(self.track_items, "por", depth, arr(tnph_curve), pg.mkPen("g", width=1), xlim=(-0.15, 0.45), invert_x=True)
        self._plot_track_curve(self.track_items, "por", depth, arr(phit_nmr_curve), pg.mkPen("k", width=1), xlim=(-0.15, 0.45), invert_x=True)
        self._plot_track_curve(self.track_items, "por", depth, arr(phie_nmr_curve), pg.mkPen("b", width=1), xlim=(-0.15, 0.45), invert_x=True)

        x_phit_chart = get_col(df, "PHIT_CHART")
        if x_phit_chart is not None:
            self._plot_track_curve(self.track_items, "por", depth, x_phit_chart, pg.mkPen("y", width=2), xlim=(-0.15, 0.45), invert_x=True)

        rhob_x = arr(rhob_curve)
        if rhob_x is not None:
            rho_ma = 2.71
            rho_fl = 1.10
            phid = np.clip((rho_ma - rhob_x) / (rho_ma - rho_fl), -0.15, 0.45)
            self._plot_track_curve(self.track_items, "por", depth, phid, pg.mkPen("r", width=2), xlim=(-0.15, 0.45), invert_x=True)

        # Rt
        self._plot_track_curve(self.track_items, "rt", depth, arr(rt_curve), pg.mkPen("lightgray", width=2, style=Qt.DashLine), xlim=(0.2, 20), logx=False)

        # NMR fills
        pi_nmr = self.track_items["nmr"]
        x_phit = arr(phit_nmr_curve)
        x_phie = arr(phie_nmr_curve)
        x_bvie = arr(bvie_curve)

        if (x_phit is not None) and (x_phie is not None) and (x_bvie is not None):
            m = np.isfinite(depth) & np.isfinite(x_phit) & np.isfinite(x_phie) & np.isfinite(x_bvie)
            if np.any(m):
                d = depth[m].astype(float)
                phit = np.clip(x_phit[m].astype(float), 0.0, 0.30)
                phie = np.minimum(np.clip(x_phie[m].astype(float), 0.0, 0.30), phit)
                bvie = np.minimum(np.clip(x_bvie[m].astype(float), 0.0, 0.30), phie)

                c_phit = pi_nmr.plot(phit, d, pen=pg.mkPen("gray", width=1), name="PHIT_NMR")
                c_phie = pi_nmr.plot(phie, d, pen=pg.mkPen("y", width=1), name="PHIE_NMR")
                c_bvie = pi_nmr.plot(bvie, d, pen=pg.mkPen("b", width=1), name="BVIE")
                zero = np.zeros_like(bvie)
                c_zero = pi_nmr.plot(zero, d, pen=pg.mkPen("w", width=1), name="0")

                pi_nmr.addItem(pg.FillBetweenItem(c_phit, c_phie, brush=pg.mkBrush("gray")))
                pi_nmr.addItem(pg.FillBetweenItem(c_phie, c_bvie, brush=pg.mkBrush("y")))
                pi_nmr.addItem(pg.FillBetweenItem(c_bvie, c_zero, brush=pg.mkBrush("b")))
                pi_nmr.getViewBox().invertX(True)
                pi_nmr.setXRange(0.0, 0.30, padding=0.0)

        # Tops
        tops_df = getattr(state, "tops_df", None)
        if tops_df is not None and len(tops_df) > 0:
            depth_col = None
            for c in ("Depth", "DEPT", "dept", "TopDepth", "Top_Depth", "MD", "TVD"):
                if c in tops_df.columns:
                    depth_col = c
                    break

            name_col = None
            for c in ("Top", "Name", "Horizon", "Marker", "Pick", "Formation"):
                if c in tops_df.columns:
                    name_col = c
                    break

            if depth_col is not None:
                ylo = float(np.nanmin(depth))
                yhi = float(np.nanmax(depth))

                for _, row in tops_df.iterrows():
                    dval = row.get(depth_col, None)
                    if dval is None or pd.isna(dval):
                        continue
                    try:
                        y = float(dval)
                    except Exception:
                        continue
                    if not (ylo <= y <= yhi):
                        continue

                    label = None
                    if name_col is not None:
                        v = row.get(name_col, None)
                        if v is not None and not pd.isna(v):
                            label = str(v)

                    for pi in self.track_items.values():
                        pi.addLine(y=y, pen=pg.mkPen("r", width=2))

                    if label:
                        gr_pi = self.track_items["gr"]
                        x_left = gr_pi.viewRange()[0][0]
                        t = pg.TextItem(label, anchor=(0, 1))
                        t.setPos(x_left, y)
                        gr_pi.addItem(t)

        ymin = float(np.nanmin(depth))
        ymax = float(np.nanmax(depth))
        for pi in self.track_items.values():
            pi.enableAutoRange(axis=pg.ViewBox.YAxis, enable=False)
            pi.setYRange(ymin, ymax, padding=0.0)

    # -------------------------------------------------------------------------
    # Matplotlib updates
    # -------------------------------------------------------------------------
    def update_nd_crossplot(self, state):
        ax = self.ax_nd
        ax.clear()

        df = getattr(state, "analysis_df_view", None)
        if df is None or df.empty:
            df = getattr(state, "analysis_df", None)
        if df is None or df.empty:
            self.canvas_nd.draw_idle()
            return

        params = getattr(state, "params", {}) or {}
        if not isinstance(params, dict):
            params = {}

        tnph = params.get("tnph_curve")
        rhob = params.get("rhob_curve")
        if (not tnph) or (tnph not in df.columns):
            tnph = _first_present(df.columns, ["TNPH", "NPHI", "CNL", "NPOR"])
        if (not rhob) or (rhob not in df.columns):
            rhob = _first_present(df.columns, ["RHOZ", "RHOB"])

        if tnph and rhob and (tnph in df.columns) and (rhob in df.columns):
            x = pd.to_numeric(df[tnph], errors="coerce").to_numpy(float)
            y = pd.to_numeric(df[rhob], errors="coerce").to_numpy(float)
            m = np.isfinite(x) & np.isfinite(y)
            ax.scatter(x[m], y[m], s=6, color="red", alpha=0.6, label="Well data")
            ax.set_xlabel("NPOR (V/V)")
            ax.set_ylabel("RHOB (g/cc)")
            ax.invert_yaxis()
            ax.grid(True, alpha=0.25)

        try:
            file_path = "./apps/merge_gui/data/cnl_chart_1pt1.xlsx"
            spine = pd.read_excel(file_path, index_col=False)
            if spine is not None and not spine.empty and ("Neutron" in spine.columns) and ("RHOB" in spine.columns):
                cx = pd.to_numeric(spine["Neutron"], errors="coerce").to_numpy(float)
                cy = pd.to_numeric(spine["RHOB"], errors="coerce").to_numpy(float)
                cm = np.isfinite(cx) & np.isfinite(cy)
                ax.scatter(cx[cm], cy[cm], s=8, alpha=0.7, c="k", label="SLB chartbook")
        except Exception:
            pass

        neut_shale = float(params.get("neut_shale", VSH_PARAM_SPEC["neut_shale"]["default"]))
        den_shale = float(params.get("den_shale", VSH_PARAM_SPEC["den_shale"]["default"]))
        neut_matrix = float(params.get("neut_matrix", VSH_PARAM_SPEC["neut_matrix"]["default"]))
        den_matrix = float(params.get("den_matrix", VSH_PARAM_SPEC["den_matrix"]["default"]))
        neut_fl = float(params.get("neut_fl", VSH_PARAM_SPEC["neut_fl"]["default"]))
        den_fl = float(params.get("den_fl", VSH_PARAM_SPEC["den_fl"]["default"]))

        A = (neut_matrix, den_matrix)
        B = (neut_shale, den_shale)
        C = (neut_fl, den_fl)

        ax.plot([A[0], B[0], C[0], A[0]], [A[1], B[1], C[1], A[1]], lw=2, color="m", alpha=0.9, label="Shale triangle")
        ax.text(A[0], A[1], " Mat", color="blue", fontsize=9, va="top")
        ax.text(B[0], B[1], f" Sh:{neut_shale:g}", color="blue", fontsize=9, va="top")
        ax.text(C[0], C[1], " Fl", color="blue", fontsize=9, va="center")
        ax.legend(loc="best", fontsize=9)
        self.canvas_nd.draw_idle()

    def update_gr_hist(self, state):
        ax = self.ax_gr
        ax.clear()

        df = getattr(state, "analysis_df_view", None)
        if df is None or df.empty:
            df = getattr(state, "analysis_df", None)
        if df is None or df.empty:
            self.canvas_gr.draw_idle()
            return

        params = getattr(state, "params", {}) or {}
        if not isinstance(params, dict):
            params = {}

        gr = params.get("gr_curve")
        if (not gr) or (gr not in df.columns):
            gr = _first_present(df.columns, ["GR", "HSGR", "SGR", "GR_EDTC", "HCGR", "CGR", "ECGR"])

        if not gr or gr not in df.columns:
            ax.text(0.5, 0.5, "No GR curve found", ha="center", va="center")
            self.canvas_gr.draw_idle()
            return

        gr_clean = float(params.get("gr_clean", 25.0))
        gr_shale = float(params.get("gr_shale", 160.0))

        g = pd.to_numeric(df[gr], errors="coerce").to_numpy(float)
        g = g[np.isfinite(g)]
        if g.size:
            ax.hist(g, bins=60, alpha=0.85, color="green")
            ax.axvline(gr_clean, lw=2, linestyle="--", color="blue", label=f"GR clean = {gr_clean:g}")
            ax.axvline(gr_shale, lw=2, linestyle="--", color="brown", label=f"GR shale = {gr_shale:g}")
            ax.axvspan(gr_clean, gr_shale, color="yellow", alpha=0.1)
            ax.set_xlabel(gr)
            ax.set_ylabel("Count")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best", fontsize=9)

        self.canvas_gr.draw_idle()

    def update_dt_rhob_crossplot(self, state):
        ax = self.ax_dt
        ax.clear()

        df = getattr(state, "analysis_df_view", None)
        if df is None or df.empty:
            df = getattr(state, "analysis_df", None)
        if df is None or df.empty:
            self.canvas_dt.draw_idle()
            return

        params = getattr(state, "params", {}) or {}
        if not isinstance(params, dict):
            params = {}

        rhob = params.get("rhob_curve")
        if (not rhob) or (rhob not in df.columns):
            rhob = _first_present(df.columns, ["RHOZ", "RHOB"])

        dtco = params.get("dtco_curve")
        if (not dtco) or (dtco not in df.columns):
            dtco = _first_present(df.columns, ["DTCO", "DTC", "AC"])

        if not rhob or not dtco or (rhob not in df.columns) or (dtco not in df.columns):
            ax.text(0.5, 0.5, "Need RHOB and DTCO", ha="center", va="center")
            self.canvas_dt.draw_idle()
            return

        x = pd.to_numeric(df[dtco], errors="coerce").to_numpy(float)
        y = pd.to_numeric(df[rhob], errors="coerce").to_numpy(float)
        m = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[m], y[m], s=6, color="dodgerblue", alpha=1, label="Well data")

        ax.set_xlabel("DTCO (µs/ft)")
        ax.set_ylabel("RHOB (g/cc)")
        ax.set_xlim(40, 189)
        ax.set_ylim(3.0, 1.0)
        ax.grid(True, alpha=0.25)

        den_matrix = float(params.get("den_matrix", VSH_PARAM_SPEC["den_matrix"]["default"]))
        den_shale = float(params.get("den_shale", VSH_PARAM_SPEC["den_shale"]["default"]))
        den_fl = float(params.get("den_fl", VSH_PARAM_SPEC["den_fl"]["default"]))
        dt_matrix = float(params.get("dt_matrix", VSH_PARAM_SPEC["dt_matrix"]["default"]))
        dt_shale = float(params.get("dt_shale", VSH_PARAM_SPEC["dt_shale"]["default"]))
        dt_fl = float(params.get("dt_fl", VSH_PARAM_SPEC["dt_fl"]["default"]))

        A = (dt_matrix, den_matrix)
        B = (dt_shale, den_shale)
        C = (dt_fl, den_fl)

        ax.plot([A[0], B[0], C[0], A[0]], [A[1], B[1], C[1], A[1]], lw=2, alpha=0.9, color="m", label="Shale triangle")
        ax.text(A[0], A[1], " Mat", color="blue", fontsize=9, va="top")
        ax.text(B[0], B[1],  f"Sh:{dt_shale:g}", color="blue", fontsize=9, va="top")
        ax.text(C[0], C[1], " Fl", color="blue", fontsize=9, va="center")
        ax.legend(loc="best", fontsize=9)
        self.canvas_dt.draw_idle()














    def update_sw_tab(self, state):
        axL = self.ax_sw_left
        axM = self.ax_sw_mid
        axR = self.ax_sw_right
        axC = self.ax_sw_cbw

        axL.clear()
        axM.clear()
        axR.clear()
        axC.clear()

        df_view = getattr(state, "analysis_df_view", None)
        df_full = getattr(state, "analysis_df", None)
        df = df_view if (df_view is not None and not df_view.empty) else df_full
        if df is None or df.empty:
            self.canvas_sw.draw_idle()
            return

        params = getattr(state, "params", {}) or {}
        if not isinstance(params, dict):
            params = {}

        rt_col = params.get("rt_curve")
        if (not rt_col) or (rt_col not in df.columns):
            rt_col = _first_present(df.columns, ["AT90", "AF90", "AO90", "ILD", "RT"])

        phit_col = params.get("phit_curve")
        if (not phit_col) or (phit_col not in df.columns):
            phit_col = "PHIT" if "PHIT" in df.columns else _first_present(df.columns, ["PHIT_CHART", "PHIT_NMR", "TCMR", "MPHS", "MPHI"])

        if (rt_col is None) or (phit_col is None):
            axL.text(0.5, 0.5, "Need PHIT and Rt", ha="center", va="center")
            self.canvas_sw.draw_idle()
            return

        y = self._depth_array(df)
        phit = pd.to_numeric(df[phit_col], errors="coerce").to_numpy(float)
        rt = pd.to_numeric(df[rt_col], errors="coerce").to_numpy(float)

        bvw = pd.to_numeric(df["BVW"], errors="coerce").to_numpy(float) if "BVW" in df.columns else None
        bvo = pd.to_numeric(df["BVO"], errors="coerce").to_numpy(float) if "BVO" in df.columns else None
        ms_app = pd.to_numeric(df["MSTAR_APP"], errors="coerce").to_numpy(float) if "MSTAR_APP" in df.columns else None
        vsh = pd.to_numeric(df["VSH_HL"], errors="coerce").to_numpy(float) if "VSH_HL" in df.columns else None

        # Left
        axL.set_title("Bulk Volume Plot")
        axL.plot(phit, y, "-r", lw=1, label="PHIT")
        if bvw is not None:
            axL.plot(bvw, y, "-k", lw=1, label="BVW")
            axL.fill_betweenx(y, phit, bvw, where=np.isfinite(phit) & np.isfinite(bvw), color="green", alpha=0.35, label="BVO")
            axL.fill_betweenx(y, bvw, 0, where=np.isfinite(bvw), color="cyan", alpha=0.35, label="BVW fill")

        axL.set_xlim(0.5, 0.0)
        if np.isfinite(y).any():
            axL.set_ylim(np.nanmax(y), np.nanmin(y))
        axL.set_xlabel("BVO/BVW")
        axL.set_ylabel("Depth")
        axL.grid(True, alpha=0.25)
        axL.legend(loc="best", fontsize=9)

        # Right
        axR.set_title("Vsh_HL vs. Mstar_Apparent")
        if (vsh is None) or (ms_app is None):
            axR.text(0.5, 0.5, "Need VSH_HL and MSTAR_APP", ha="center", va="center")
        else:
            m = np.isfinite(vsh) & np.isfinite(ms_app)
            axR.plot(vsh[m], ms_app[m], "r.", ms=3)
            m_cem = self._get_sw("sw.m_cem")
            mslope = self._get_sw("sw.mslope")
            xline = np.linspace(0.0, 1.0, 200)
            axR.plot(xline, xline * mslope + m_cem, "k-", lw=2)

        axR.set_xlim(0.0, 1.0)
        axR.set_ylim(0.0, 7.0)
        axR.set_xlabel("Vsh_HL [v/v]")
        axR.set_ylabel("Mstar Apparent")
        axR.grid(True, alpha=0.25)

        # Middle Pickett
        axM.set_title("Pickett Plot")
        m = np.isfinite(rt) & (rt > 0) & np.isfinite(phit) & (phit > 0)
        axM.loglog(rt[m], phit[m], "r.", ms=3)

        axM.set_xlim(0.01, 1000)
        axM.set_ylim(0.01, 1.0)
        axM.set_xlabel(f"{rt_col} [ohm-m]")
        axM.set_ylabel("PHIT [v/v]")
        axM.grid(True, which="both", alpha=0.25)

        m_cem = self._get_sw("sw.m_cem")
        n_sat = self._get_sw("sw.n_sat")
        rw = self._get_sw("sw.rw")

        sw_plot = (1.0, 0.8, 0.6, 0.4, 0.2)
        phit_plot = np.array([0.01, 1.0])
        for sw in sw_plot:
            rt_line = (rw / (sw ** n_sat)) * (1.0 / (phit_plot ** m_cem))
            axM.loglog(rt_line, phit_plot, lw=2, label=f"Sw {int(sw * 100)}%")
        axM.legend(loc="best", fontsize=8)

        # CBW vs VSH
        axC.set_title("Vsh_HL vs. CBW", color="blue")
        cbw_col = _first_present(df.columns, ["CBW", "CBWapp", "CBWa", "CBW_NMR", "CBWAPP"])
        if (vsh is None) or (cbw_col is None):
            axC.text(0.5, 0.5, "Need VSH_HL + CBW", ha="center", va="center")
        else:
            cbw = pd.to_numeric(df[cbw_col], errors="coerce").to_numpy(float)
            mm = np.isfinite(vsh) & np.isfinite(cbw)
            axC.plot(vsh[mm], cbw[mm], "r.", ms=3)

            cbw_int = self._get_sw("sw.cbw_intercept")
            x = np.linspace(0.0, 1.0, 200)
            axC.plot(x, x * cbw_int, "k-", lw=2)

            axC.set_xlim(0.0, 1.0)
            axC.set_ylim(0.0, 0.5)
            axC.set_xlabel("Vsh_HL [v/v]", color="blue")
            axC.set_ylabel(f"{cbw_col} [v/v]", color="blue")
            axC.grid(True, alpha=0.25)

        self.fig_sw.tight_layout()
        self.canvas_sw.draw_idle()