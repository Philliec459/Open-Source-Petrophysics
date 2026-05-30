





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

# =============================================================================
# PlotsPanel
# =============================================================================

class PlotsPanelInitial(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
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

 
        self._build_tracks()

        #self.tabs.addTab(self._build_crossplots_tab(), "Crossplots + Histogram")
        #self.tabs.addTab(self._build_sw_tab(), "Sw / Pickett / M* / CBW")
        #self.tabs.addTab(self._build_final_depth_tab(), "Final Depth Plot")

        layout = QVBoxLayout(self)
        layout.addWidget(self.tabs)


    # -------------------------------------------------------------------------
    # Public
    # -------------------------------------------------------------------------
    def update_all(self, state):
        self.update_depth_plot(state)
        try:
            self._refresh_final_depth_plot()
        except Exception:
            pass





    def _after_las_loaded(self):

        from petrocore.services.curve_family_service import classify_curve_families

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


