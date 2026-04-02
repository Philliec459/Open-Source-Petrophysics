# apps/merge_gui/ui_panels/plots_panel_vsh.py
from __future__ import annotations

print(">>> LOADING plots_panel_vsh.py from:", __file__)

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

from petrocore.services.curve_family_service import classify_curve_families
from petrocore.workflow.phit_chartbook import compute_phit_chartbook


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
    "gr_clean": dict(default=10.0,  mn=0.0, mx=300.0, step=0.5, dec=1, label="GR_clean"),
    "gr_shale": dict(default=210.0, mn=0.0, mx=300.0, step=0.5, dec=1, label="GR_shale"),



}

RHOB_CANDS = ["RHOB", "RHOZ", "DEN", "DENS", "RHO8"]
NPHI_CANDS = ["TNPH", "NPHI", "CNL", "NPOR", "NEUT"]
DT_CANDS = ["DTCO", "DTC", "DT", "AC"]
GR_CANDS = ["GR", "SGR", "CGR", "HCGR", "ECGR", "GRC", "GR_EDTC", "HSGR", "HGR"]
DEPT_CANDS = ["DEPT", "DEPTH", "Depth", "depth", "MD"]


# =============================================================================
# Utilities
# =============================================================================

def _first_present(cols, candidates):
    s = set(cols)
    for c in candidates:
        if c in s:
            return c
    return None


def _find_curve_case_insensitive(cols, candidates):
    """
    First try exact match, then case-insensitive exact match,
    then substring match.
    """
    if cols is None:
        return None

    cols_list = list(cols)
    cols_upper = {c.upper(): c for c in cols_list}

    # exact
    for cand in candidates:
        if cand in cols_list:
            return cand

    # case-insensitive exact
    for cand in candidates:
        if cand.upper() in cols_upper:
            return cols_upper[cand.upper()]

    # substring fallback
    for cand in candidates:
        cu = cand.upper()
        for col in cols_list:
            if cu in col.upper():
                return col

    return None


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


def vsh_from_gr(gr, gr_clean=10.0, gr_shale=210.0, clip=True):
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
        gamma_col = _first_present(
            df.columns,
            ["HCGR", "CGR", "ECGR", "GR_EDTC", "HSGR", "GR", "SGR", "HGR"]
        )
    if not gamma_col or gamma_col not in df.columns:
        return None, None

    gr_clean = float(params.get("gr_clean", 10.0))
    gr_shale = float(params.get("gr_shale", 210.0))
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
        tnph = _find_curve_case_insensitive(df.columns, NPHI_CANDS)
    if (not rhob) or (rhob not in df.columns):
        rhob = _find_curve_case_insensitive(df.columns, RHOB_CANDS)
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
        rhob = _find_curve_case_insensitive(df.columns, RHOB_CANDS)

    dtco = params.get("dtco_curve")
    if (not dtco) or (dtco not in df.columns):
        dtco = _find_curve_case_insensitive(df.columns, DT_CANDS)

    if (rhob is None) or (dtco is None):
        return None

    x = pd.to_numeric(df[dtco], errors="coerce").to_numpy(float)
    y = pd.to_numeric(df[rhob], errors="coerce").to_numpy(float)

    A = (float(pget(params, "dt_matrix")), float(pget(params, "den_matrix")))
    B = (float(pget(params, "dt_shale")), float(pget(params, "den_shale")))
    C = (float(pget(params, "dt_fl")), float(pget(params, "den_fl")))
    return vsh_from_triangle(x, y, A, B, C, clip=True)


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

class PlotsPanelVsh(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.param_store = ParamStore(controller)
        self.settings = QSettings("CrestedButtePetro", "PetroSuite6")

        self.tabs = QTabWidget()

        # Main pyqtgraph depth tracks
        self.tracks = pg.GraphicsLayoutWidget()
        self.track_items = {}

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
        self._build_tracks()

        self.tabs.addTab(self._build_crossplots_tab(), "Crossplots + Histogram")
        # self.tabs.addTab(self._build_sw_tab(), "Sw / Pickett / M* / CBW")
        # self.tabs.addTab(self._build_final_depth_tab(), "Final Depth Plot")

        layout = QVBoxLayout(self)
        layout.addWidget(self.tabs)

        self._vsh_recompute_timer = QTimer(self)
        self._vsh_recompute_timer.setSingleShot(True)
        self._vsh_recompute_timer.timeout.connect(self._live_vsh_update)

        self._sw_recompute_timer = QTimer(self)
        self._sw_recompute_timer.setSingleShot(True)
        self._sw_recompute_timer.timeout.connect(self._live_sw_update)

    # -------------------------------------------------------------------------
    # Public
    # -------------------------------------------------------------------------
    def update_all(self, state):
        self.update_nd_crossplot(state)
        self.update_gr_hist(state)
        self.update_dt_rhob_crossplot(state)
        try:
            self._refresh_vsh_depth_plot()
        except Exception:
            pass

    def _after_las_loaded(self):
        state = self._state()
        if state is None:
            return

        df = getattr(state, "analysis_df", None)
        if df is None or df.empty:
            return

        families = classify_curve_families(df)
        state.curve_families = families

        print("Curve families:", families)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------
    def _state(self):
        return getattr(self.controller, "state", None)

    def _get_df_view_or_full(self):
        state = self._state()
        if state is None:
            return None

        # ✅ Always trust controller
        df = getattr(state, "view_df", None)
        if df is not None and not df.empty:
            return df

        # fallback
        return getattr(state, "analysis_df", None)

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
            self.update_nd_crossplot(self._state())
            self.update_dt_rhob_crossplot(self._state())
            self.update_gr_hist(self._state())
        except Exception:
            pass

    def _live_sw_update(self):
        try:
            self._recompute_swb_only()
            self._safe_refresh()
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
            return _first_present(df.columns, fallbacks)
        return None

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
            a, b, c = (
                pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
                for col in good_cols[:3]
            )
            P = np.stack([0.5 * (a + b), 0.5 * (a + c), 0.5 * (b + c)], axis=0)
            hl = np.nanmedian(P, axis=0)

        hl = np.clip(pd.to_numeric(hl, errors="coerce"), 0.0, 1.0)
        if "VSH_GR" in df.columns:
            hl = pd.Series(hl, index=df.index).fillna(
                pd.to_numeric(df["VSH_GR"], errors="coerce")
            ).to_numpy(dtype=float)

        df["VSH_HL"] = hl
        state.analysis_df = df

        families = classify_curve_families(df)
        state.curve_families = families

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
            self.update_nd_crossplot(self._state())
            self.update_dt_rhob_crossplot(self._state())
            self.update_gr_hist(self._state())
        except Exception:
            pass
        #self.status_label.setText("VSH calculated successfully ✔")
        #self._update_notes()
        print()
        print()
        print("[VSH Message:] Vsh_HL Computed and plots refreshed")


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

        self._recompute_swb_only()

        df = getattr(state, "analysis_df", None)
        if df is None or df.empty:
            return

        swb = pd.to_numeric(df["SWB"], errors="coerce").to_numpy(float) if "SWB" in df.columns else None

        x0 = np.array([
            float(params.get("gr_clean", 10.0)),
            float(params.get("gr_shale", 210.0)),
            float(params.get("neut_shale", 0.4)),
            float(params.get("dt_shale", 90.0)),
        ], dtype=float)

        x_ref = x0.copy()

        x_scale = np.array([
            10.0,
            30.0,
            0.06,
            10.0,
        ], dtype=float)

        bounds = [
            (5.0, 50.0),
            (100.0, 250.0),
            (0.30, 0.45),
            (75.0, 110.0),
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

            shale_mask = np.ones(len(df), dtype=bool)

            swb_min_fit = 0.12
            swb_max_fit = 0.50

            if swb is not None:
                shale_mask = np.isfinite(swb) & (swb > swb_min_fit) & (swb < swb_max_fit)

            if np.sum(shale_mask) < 20:
                print("[VSH AUTO] Too few points in SWB fit mask; falling back to finite HL samples")
                shale_mask = np.isfinite(hl)

            loss = 0.0

            if vsh_gr is not None and vsh_nd is not None:
                loss += 1.0 * nan_mse_masked(vsh_gr, vsh_nd, shale_mask)
            if vsh_gr is not None and vsh_dt is not None:
                loss += 1.0 * nan_mse_masked(vsh_gr, vsh_dt, shale_mask)
            if vsh_nd is not None and vsh_dt is not None:
                loss += 1.0 * nan_mse_masked(vsh_nd, vsh_dt, shale_mask)

            if swb is not None:
                loss += 0.35 * nan_mse_masked(hl, swb, shale_mask)

            reg = np.sum(((x - x_ref) / x_scale) ** 2)
            loss += 0.50 * reg

            if gr_shale <= gr_clean + 40.0:
                loss += 25.0

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

        self._set_param_and_refresh("gr_clean", float(gr_clean))
        self._set_param_and_refresh("gr_shale", float(gr_shale))
        self._set_param_and_refresh("neut_shale", float(neut_shale))
        self._set_param_and_refresh("dt_shale", float(dt_shale))

        self._recompute_vsh_only()
        self._safe_rebuild_view()
        self._safe_refresh()
        try:
            self._refresh_vsh_depth_plot()
        except Exception:
            pass

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

        gr_clean = float(self.param_store.get("gr_clean", 10))
        gr_shale = float(self.param_store.get("gr_shale", 205))

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

        #vsh_cols = [c for c in df.columns if "VSH" in c.upper()]
        #if "VSH_HL" in vsh_cols:
        #    vsh_cols = [c for c in vsh_cols if c != "VSH_HL"] + ["VSH_HL"]

        allowed_vsh_cols = ["VSH_ND", "VSH_DT", "VSH_GR", "VSH_HL"]
        vsh_cols = [c for c in allowed_vsh_cols if c in df.columns]



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
                p.plot(
                    np.ascontiguousarray(x[m], dtype=np.float64),
                    np.ascontiguousarray(d[m], dtype=np.float64),
                    name=col,
                    pen=pen_map.get(col, pg.mkPen(width=2)),
                )
                any_plotted = True

        if any_plotted:
            p.setXRange(0.0, 1.0, padding=0.02)
            p.setYRange(float(np.nanmin(d)), float(np.nanmax(d)), padding=0.0)
            p.invertY(True)








    # -------------------------------------------------------------------------
    # Final tracks
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Matplotlib updates
    # -------------------------------------------------------------------------
    def update_nd_crossplot(self, state):
        ax = self.ax_nd
        ax.clear()

        df = self._get_df_view_or_full()
        if df is None or df.empty:
            self.canvas_nd.draw_idle()
            return

        params = getattr(state, "params", {}) or {}
        if not isinstance(params, dict):
            params = {}

        tnph = params.get("tnph_curve")
        rhob = params.get("rhob_curve")

        if (not tnph) or (tnph not in df.columns):
            tnph = _find_curve_case_insensitive(df.columns, NPHI_CANDS)
        if (not rhob) or (rhob not in df.columns):
            rhob = _find_curve_case_insensitive(df.columns, RHOB_CANDS)

        if tnph and rhob and (tnph in df.columns) and (rhob in df.columns):
            x = pd.to_numeric(df[tnph], errors="coerce").to_numpy(float)
            y = pd.to_numeric(df[rhob], errors="coerce").to_numpy(float)
            m = np.isfinite(x) & np.isfinite(y)
            ax.scatter(x[m], y[m], s=6, color="red", alpha=0.6, label="Well data")
            ax.set_xlabel(tnph)
            ax.set_ylabel(rhob)
            ax.invert_yaxis()
            ax.grid(True, alpha=0.25)
        else:
            ax.text(
                0.5, 0.5,
                f"Need neutron and density\nFound TNPH={tnph}, RHOB={rhob}",
                ha="center", va="center", transform=ax.transAxes
            )

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

        ax.plot(
            [A[0], B[0], C[0], A[0]],
            [A[1], B[1], C[1], A[1]],
            lw=2, color="m", alpha=0.9, label="Shale triangle"
        )
        ax.text(A[0], A[1], " Mat", color="blue", fontsize=9, va="top")
        ax.text(B[0], B[1], f" Sh:{neut_shale:g}", color="blue", fontsize=9, va="top")
        ax.text(C[0], C[1], " Fl", color="blue", fontsize=9, va="center")
        ax.legend(loc="best", fontsize=9)
        self.canvas_nd.draw_idle()

    def update_gr_hist(self, state):
        ax = self.ax_gr
        ax.clear()

        df = self._get_df_view_or_full()
        if df is None or df.empty:
            self.canvas_gr.draw_idle()
            return

        params = getattr(state, "params", {}) or {}
        if not isinstance(params, dict):
            params = {}

        gr = params.get("gr_curve")
        if (not gr) or (gr not in df.columns):
            gr = _find_curve_case_insensitive(df.columns, GR_CANDS)

        if not gr or gr not in df.columns:
            ax.text(0.5, 0.5, "No GR curve found", ha="center", va="center")
            self.canvas_gr.draw_idle()
            return

        gr_clean = float(params.get("gr_clean", 10.0))
        gr_shale = float(params.get("gr_shale", 205.0))

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

        df = self._get_df_view_or_full()
        if df is None or df.empty:
            self.canvas_dt.draw_idle()
            return

        params = getattr(state, "params", {}) or {}
        if not isinstance(params, dict):
            params = {}

        rhob = params.get("rhob_curve")
        if (not rhob) or (rhob not in df.columns):
            rhob = _find_curve_case_insensitive(df.columns, RHOB_CANDS)

        dtco = params.get("dtco_curve")
        if (not dtco) or (dtco not in df.columns):
            dtco = _find_curve_case_insensitive(df.columns, DT_CANDS)

        if not rhob or not dtco or (rhob not in df.columns) or (dtco not in df.columns):
            ax.text(
                0.5, 0.5,
                f"Need RHOB and DTCO\nFound RHOB={rhob}, DTCO={dtco}",
                ha="center", va="center", transform=ax.transAxes
            )
            self.canvas_dt.draw_idle()
            return

        x = pd.to_numeric(df[dtco], errors="coerce").to_numpy(float)
        y = pd.to_numeric(df[rhob], errors="coerce").to_numpy(float)
        m = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[m], y[m], s=6, color="dodgerblue", alpha=1, label="Well data")

        ax.set_xlabel(dtco)
        ax.set_ylabel(rhob)
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

        ax.plot(
            [A[0], B[0], C[0], A[0]],
            [A[1], B[1], C[1], A[1]],
            lw=2, alpha=0.9, color="m", label="Shale triangle"
        )
        ax.text(A[0], A[1], " Mat", color="blue", fontsize=9, va="top")
        ax.text(B[0], B[1], f"Sh:{dt_shale:g}", color="blue", fontsize=9, va="top")
        ax.text(C[0], C[1], " Fl", color="blue", fontsize=9, va="center")
        ax.legend(loc="best", fontsize=9)
        self.canvas_dt.draw_idle()