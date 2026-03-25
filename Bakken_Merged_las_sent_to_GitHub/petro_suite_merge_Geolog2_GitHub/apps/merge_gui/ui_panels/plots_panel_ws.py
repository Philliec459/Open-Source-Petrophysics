# apps/merge_gui/ui_panels/plots_panel_ws.py
from __future__ import annotations

print(">>> LOADING plots_panel_ws.py from:", __file__)

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
)

from petrocore.services.curve_family_service import classify_curve_families


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


def compute_sal_kppm_from_rw75(rw75):
    rw75 = np.asarray(rw75, dtype=float)
    x = np.clip(rw75 - 0.0123, 1e-6, None)
    return (10 ** ((3.562 - np.log10(x)) / 0.955)) / 1000.0


def compute_bdacy(T_F, rw):
    rw = np.asarray(rw, dtype=float)
    tc = (float(T_F) - 32.0) / 1.8
    tc_safe = max(tc, 1e-6)
    rw_safe = np.clip(rw, 1e-6, None)
    term1 = (1.0 - 0.83 * np.exp(-np.exp(-2.38 + (42.17 / tc_safe)) / rw_safe))
    term2 = (-3.16 + 1.59 * np.log(tc_safe)) ** 2
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

    if qv_col and qv_col in df.columns:
        qv = pd.to_numeric(df[qv_col], errors="coerce").to_numpy(float)
    else:
        qv = np.zeros_like(phit)

    if vsh_col and vsh_col in df.columns:
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

    BVW_m = phit[m] * ((1.0 / (phit[m] ** m_cem)) * (rw / rt[m])) ** (1.0 / n_sat)
    BVW_m = np.minimum(BVW_m, phit[m])

    SWT_m = np.where(phit[m] > 0, BVW_m / phit[m], np.nan)
    BVO_m = phit[m] * (1.0 - SWT_m)

    denom = rt[m] * (1.0 + rw * B * qv[m])
    ok = (
        np.isfinite(denom)
        & (denom > 0)
        & np.isfinite(phit[m])
        & (phit[m] > 0)
        & (phit[m] != 1.0)
    )

    ms_app = np.full(np.sum(m), np.nan, float)
    ms_app[ok] = np.log10(rw / denom[ok]) / np.log10(phit[m][ok])

    BVW[m] = BVW_m
    SWT[m] = SWT_m
    BVO[m] = BVO_m
    MSTAR_APP[m] = ms_app

    if np.any(np.isfinite(vsh)):
        MSTAR = vsh * mslope + m_cem

    return {
        "BVW": BVW,
        "SWT": SWT,
        "BVO": BVO,
        "MSTAR_APP": MSTAR_APP,
        "MSTAR": MSTAR,
    }


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
# Float slider UI
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
# PlotsPanelWS
# =============================================================================

class PlotsPanelWS(QWidget):
    """
    Waxman-Smits / Pickett / BVW workspace.

    ZoI behavior in this version:
    - plots use ZoI only
    - exports use ZoI only
    - calculations are performed on ZoI copy and merged back into full analysis_df
      by index, so the full well dataframe is not replaced by the ZoI slice
    """

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.param_store = ParamStore(controller)
        self.settings = QSettings("CrestedButtePetro", "PetroSuite6")

        self.tabs = QTabWidget()

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

        self._seed_sw_defaults_once()

        self.tabs.addTab(self._build_sw_tab(), "Sw / Pickett / M* / CBW")

        layout = QVBoxLayout(self)
        layout.addWidget(self.tabs)

        self._sw_recompute_timer = QTimer(self)
        self._sw_recompute_timer.setSingleShot(True)
        self._sw_recompute_timer.timeout.connect(self._on_compute_sw_clicked)

    # -------------------------------------------------------------------------
    # Public
    # -------------------------------------------------------------------------
    def update_all(self, state):
        self.update_sw_tab(state)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------
    def _state(self):
        return getattr(self.controller, "state", None)

    def _get_full_df(self):
        state = self._state()
        if state is None:
            return None
        return getattr(state, "analysis_df", None)

    def _get_df_view_or_full(self):
        state = self._state()
        if state is None:
            return None

        # 1) Prefer existing ZoI/view dataframe
        df = getattr(state, "analysis_df_view", None)
        if df is not None and not df.empty:
            return df

        # 2) Otherwise build ZoI from full well if tops/zone exists
        df = getattr(state, "analysis_df", None)
        if df is None or df.empty:
            return None

        df = df.copy()

        depth_col = None
        for c in ["DEPT", "DEPTH", "Depth", "depth", "MD"]:
            if c in df.columns:
                depth_col = c
                break

        if depth_col is None:
            return df

        top = getattr(state, "depth_top", None)
        base = getattr(state, "depth_base", None)

        if top is not None and base is not None:
            lo, hi = sorted([float(top), float(base)])
            d = pd.to_numeric(df[depth_col], errors="coerce")
            df = df[(d >= lo) & (d <= hi)].copy()

        return df

    def _merge_view_results_into_full(self, df_view: pd.DataFrame, columns: list[str]):
        """
        Push computed ZoI columns back into full analysis_df by index so we do not
        accidentally replace the full well with only the ZoI slice.
        """
        state = self._state()
        if state is None or df_view is None or df_view.empty:
            return

        df_full = getattr(state, "analysis_df", None)
        if df_full is None or df_full.empty:
            state.analysis_df = df_view.copy()
            return

        df_full = df_full.copy()

        for col in columns:
            if col not in df_full.columns:
                df_full[col] = np.nan
            if col in df_view.columns:
                df_full.loc[df_view.index, col] = df_view[col]

        state.analysis_df = df_full

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

        self._safe_refresh()

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

    def _after_las_loaded(self):
        state = self._state()
        if state is None:
            return

        df = getattr(state, "analysis_df", None)
        if df is None or df.empty:
            return

        try:
            families = classify_curve_families(df)
            state.curve_families = families
            print("Curve families:", families)
        except Exception as e:
            print(f"[PlotsPanelWS] classify_curve_families failed: {e}")

    # -------------------------------------------------------------------------
    # Export
    # -------------------------------------------------------------------------
    def _on_export_df_clicked(self):
        state = self._state()
        if state is None:
            QMessageBox.warning(self, "Export DF", "State object not found.")
            return

        df = self._get_df_view_or_full()
        if df is None or df.empty:
            QMessageBox.warning(self, "Export DF", "No dataframe available to export.")
            return

        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Export DataFrame",
            "qt_export.csv",
            "CSV Files (*.csv);;Excel Files (*.xlsx)",
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
    # Computation helpers
    # -------------------------------------------------------------------------
    def _recompute_swb_only(self):
        state = self._state()
        if state is None:
            return

        df_view = self._get_df_view_or_full()
        if df_view is None or df_view.empty:
            return

        df_view = df_view.copy()

        cbw_col = next((c for c in ["CBW", "CBWapp"] if c in df_view.columns), None)
        phit_col = next((c for c in ["PHIT", "PHIT_NMR", "TCMR", "MPHI", "NMR_PHIT"] if c in df_view.columns), None)
        if not (cbw_col and phit_col):
            return

        cbw = pd.to_numeric(df_view[cbw_col], errors="coerce").to_numpy(dtype=float)
        phit = pd.to_numeric(df_view[phit_col], errors="coerce").to_numpy(dtype=float)

        swb = np.full_like(phit, np.nan, dtype=float)
        m = np.isfinite(cbw) & np.isfinite(phit) & (phit > 0.01)
        swb[m] = cbw[m] / phit[m]
        swb = np.clip(swb, 0.0, 1.0)

        swb_series = pd.Series(swb, index=df_view.index)
        swb_med = swb_series.rolling(window=3, center=True, min_periods=1).median()
        swb_smooth = swb_med.rolling(window=3, center=True, min_periods=1).mean()
        df_view["SWB"] = swb_smooth.to_numpy(dtype=float)

        self._merge_view_results_into_full(df_view, ["SWB"])

    # -------------------------------------------------------------------------
    # Compute buttons
    # -------------------------------------------------------------------------
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

        df = self._get_df_view_or_full()
        if df is None or df.empty:
            self.canvas_sw.draw_idle()
            return

        df = df.copy()

        params = getattr(state, "params", {}) or {}
        if not isinstance(params, dict):
            params = {}

        rt_col = params.get("rt_curve")
        if (not rt_col) or (rt_col not in df.columns):
            rt_col = _first_present(df.columns, ["AT90", "AF90", "AO90", "ILD", "RT"])

        phit_col = params.get("phit_curve")
        if (not phit_col) or (phit_col not in df.columns):
            phit_col = "PHIT" if "PHIT" in df.columns else _first_present(
                df.columns, ["PHIT_CHART", "PHIT_NMR", "TCMR", "MPHIS", "MPHI"]
            )

        if (rt_col is None) or (phit_col is None):
            print("[SW] Need Rt + PHIT to compute SW/BVW/M*")
            return

        qv_col = "Qv" if "Qv" in df.columns else ("QV" if "QV" in df.columns else None)
        vsh_col = "VSH_HL" if "VSH_HL" in df.columns else None

        m_cem = self._get_sw("sw.m_cem")
        n_sat = self._get_sw("sw.n_sat")
        rw = self._get_sw("sw.rw")
        mslope = self._get_sw("sw.mslope")
        T_F = self._get_sw("sw.T_F")

        B = float(compute_bdacy(T_F, rw))

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
        df["Bdacy"] = B

        cols_to_merge = [
            "PHIT_USED",
            "BVW",
            "SWT",
            "BVO",
            "MSTAR_APP",
            "MSTAR_FIT",
            "Bdacy",
        ]
        self._merge_view_results_into_full(df, cols_to_merge)

        self._safe_rebuild_view()
        self._safe_refresh()
        self.update_sw_tab(state)

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

        df = self._get_df_view_or_full()
        if df is None or df.empty:
            self.canvas_sw.draw_idle()
            return

        df = df.copy()

        params = getattr(state, "params", {}) or {}
        if not isinstance(params, dict):
            params = {}

        rt_col = params.get("rt_curve")
        if (not rt_col) or (rt_col not in df.columns):
            rt_col = _first_present(df.columns, ["AT90", "AF90", "AO90", "ILD", "RT"])
        if rt_col is None:
            print("[WS] No Rt curve found")
            return

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

        vsh_col = "VSH_HL" if "VSH_HL" in df.columns else None
        if vsh_col is None:
            print("[WS] Need VSH_HL first")
            return

        m_cem = self._get_sw("sw.m_cem")
        n_sat = self._get_sw("sw.n_sat")
        rw = self._get_sw("sw.rw")
        mslope = self._get_sw("sw.mslope")
        cbw_intercept = self._get_sw("sw.cbw_intercept")
        T_F = self._get_sw("sw.T_F")
        den_fl = self._get_sw("sw.den_fl")

        rw = max(float(rw), 1e-6)
        Rw75 = ((T_F + 6.77) * rw) / (75.0 + 6.77)
        SAL = float(compute_sal_kppm_from_rw75(Rw75))
        B = float(compute_bdacy(T_F, rw))

        df["Rw75"] = Rw75
        df["SAL"] = SAL
        df["Bdacy"] = B

        params["sw.B"] = B
        state.params["sw.B"] = B
        self.param_store.set("sw.B", B)

        print(f"[WS] Using scalar values: Rw={rw:.5f}, Rw75={Rw75:.5f}, SAL={SAL:.5f}, B={B:.5f}")

        Vsh = pd.to_numeric(df[vsh_col], errors="coerce").to_numpy(float)
        Vsh = np.clip(Vsh, 0.0, 1.0)
        df["CBWapp"] = np.clip(Vsh * cbw_intercept, 0.0, 1.0)

        PHIT = pd.to_numeric(df[phit_col], errors="coerce").to_numpy(dtype=float)
        PHIT = np.clip(PHIT, 0.0, None)
        PHIT_safe = np.clip(PHIT, 1e-6, None)

        cbw = pd.to_numeric(df["CBWapp"], errors="coerce").to_numpy(dtype=float)
        cbw = np.clip(cbw, 0.0, None)

        Swb = np.clip(cbw / PHIT_safe, 0.0, 1.0)
        df["Swb"] = Swb
        df["SWB"] = Swb

        denom = (0.6425 / np.sqrt(max(den_fl * SAL, 1e-12)) + 0.22)
        df["Qv"] = np.clip(Swb / denom, 0.0, 5.0)
        qv_col = "Qv"

        write_constants_to_file(
            "./apps/merge_gui/data/Pickett.txt",
            m_cem=m_cem,
            n_sat=n_sat,
            Rw=rw,
            mslope=mslope,
            Bdacy=B,
        )

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
                rw=rw,
                m=MSTAR,
                n=n_sat,
                B=B,
                max_iter=60,
                tol=1e-6,
                sw0=None,
            )
            Sw_cp = np.clip(np.asarray(Sw_cp, dtype=float), 1e-4, 1.0)

        except Exception as e:
            print(f"[WS] iterative solver unavailable, using fallback: {e}")
            good = np.isfinite(Rt) & np.isfinite(PHIT) & (Rt > 0) & (PHIT > 0)
            Sw_cp = np.full(len(df), np.nan, dtype=float)
            Sw_cp[good] = (
                (rw / Rt[good]) * (1.0 / np.clip(PHIT[good] ** MSTAR[good], 1e-12, None))
            ) ** (1.0 / n_sat)
            Sw_cp = np.clip(Sw_cp, 1e-4, 1.0)

        df["SW_CP"] = Sw_cp
        df["BVWT_CP"] = PHIT * Sw_cp
        df["BVWe_CP"] = np.clip(PHIT * Sw_cp - cbw, 0.0, None)

        PHIE = np.clip(PHIT - cbw, 0.0, None)
        PHIE = np.minimum(PHIE, PHIT)
        df["PHIE"] = PHIE
        df["PHIT"] = PHIT

        cols_to_merge = [
            "PHIT",
            "Rw75",
            "SAL",
            "Bdacy",
            "CBWapp",
            "Swb",
            "SWB",
            "Qv",
            "MSTAR",
            "SW_CP",
            "BVWT_CP",
            "BVWe_CP",
            "PHIE",
        ]
        self._merge_view_results_into_full(df, cols_to_merge)

        self._safe_rebuild_view()
        self._safe_refresh()
        self.update_sw_tab(state)

    def _on_final_zone_plot_clicked(self):
        state = self._state()
        if state is None:
            return

        df = self._get_df_view_or_full()
        if df is None or df.empty:
            self.canvas_sw.draw_idle()
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
        note.setStyleSheet(
            """
            QLabel {
                color: red;
                font-size: 11px;
                background-color: #f8f8f8;
                border: 1px solid #dddddd;
                border-radius: 4px;
                padding: 8px;
            }
            """
        )
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

    # -------------------------------------------------------------------------
    # Plot update
    # -------------------------------------------------------------------------
    def update_sw_tab(self, state):
        axL = self.ax_sw_left
        axM = self.ax_sw_mid
        axR = self.ax_sw_right
        axC = self.ax_sw_cbw

        axL.clear()
        axM.clear()
        axR.clear()
        axC.clear()

        df = self._get_df_view_or_full()
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
            phit_col = "PHIT" if "PHIT" in df.columns else _first_present(
                df.columns, ["PHIT_CHART", "PHIT_NMR", "TCMR", "MPHS", "MPHI"]
            )

        if (rt_col is None) or (phit_col is None):
            axL.text(0.5, 0.5, "Need PHIT and Rt", ha="center", va="center")
            self.canvas_sw.draw_idle()
            return

        y = self._depth_array(df)
        phit = pd.to_numeric(df[phit_col], errors="coerce").to_numpy(float)
        rt = pd.to_numeric(df[rt_col], errors="coerce").to_numpy(float)

        bvw = pd.to_numeric(df["BVW"], errors="coerce").to_numpy(float) if "BVW" in df.columns else None
        ms_app = pd.to_numeric(df["MSTAR_APP"], errors="coerce").to_numpy(float) if "MSTAR_APP" in df.columns else None
        vsh = pd.to_numeric(df["VSH_HL"], errors="coerce").to_numpy(float) if "VSH_HL" in df.columns else None

        axL.set_title("Bulk Volume Plot")
        axL.plot(phit, y, "-r", lw=1, label="PHIT")
        if bvw is not None:
            axL.plot(bvw, y, "-k", lw=1, label="BVW")
            axL.fill_betweenx(
                y, phit, bvw,
                where=np.isfinite(phit) & np.isfinite(bvw),
                color="green", alpha=0.35, label="BVO",
            )
            axL.fill_betweenx(
                y, bvw, 0,
                where=np.isfinite(bvw),
                color="cyan", alpha=0.35, label="BVW fill",
            )

        axL.set_xlim(0.5, 0.0)
        if np.isfinite(y).any():
            axL.set_ylim(np.nanmax(y), np.nanmin(y))
        axL.set_xlabel("BVO/BVW")
        axL.set_ylabel("Depth")
        axL.grid(True, alpha=0.25)
        axL.legend(loc="best", fontsize=9)

        axR.set_title("Vsh_HL vs. Mstar Apparent")
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