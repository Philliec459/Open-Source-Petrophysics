from __future__ import annotations

print(">>> LOADING plots_panel.py from:", __file__)

import numpy as np
import pandas as pd
import pyqtgraph as pg

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QTabWidget

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas


class PlotsPanel(QWidget):
    """
    Tabs:
      1) Depth Tracks (PyQtGraph multi-track)
      2) N–D Crossplot (Matplotlib)
    """
    def __init__(self):
        super().__init__()

        self.tabs = QTabWidget()

        # -------------------------
        # Depth tracks (PyQtGraph)
        # -------------------------
        self.tracks = pg.GraphicsLayoutWidget()
        self.tabs.addTab(self.tracks, "Depth Tracks")

        self.track_items: dict[str, pg.PlotItem] = {}
        self._build_tracks()

        # -------------------------
        # N–D crossplot (Matplotlib)
        # -------------------------
        self.fig, self.ax = plt.subplots(figsize=(4, 4))
        self.canvas = FigureCanvas(self.fig)
        self.tabs.addTab(self.canvas, "N-D Crossplot")

        layout = QVBoxLayout(self)
        layout.addWidget(self.tabs)

    # -------------------------
    # Public API
    # -------------------------
    def update_all(self, state):
        self.update_depth_plot(state)
        self.update_nd_crossplot(state)

    # -------------------------
    # Tracks (BUILD ONLY)
    # -------------------------
    def _build_tracks(self):
        self.tracks.clear()

        gr  = self.tracks.addPlot(row=0, col=0, title="GR/CGR")
        por = self.tracks.addPlot(row=0, col=1, title="Porosity")

        rt  = self.tracks.addPlot(row=0, col=2, title="log10(Rt)")

        nmr = self.tracks.addPlot(row=0, col=3, title="NMR")


        por.setYLink(gr)
        rt.setYLink(gr)    # ✅ SAFE now because we are NOT using log axis
        nmr.setYLink(gr)



        
        for p in (gr, por, rt, nmr):
            p.invertY(True)
            p.showGrid(x=True, y=True, alpha=0.2)

        # Resistivity is log-x
        #rt.setLogMode(x=True, y=False)

        self.track_items = {"gr": gr, "por": por, "rt": rt, "nmr": nmr}

    # -------------------------
    # Plot update
    # -------------------------
    def update_depth_plot(self, state):
        # Clear tracks
        for pi in self.track_items.values():
            pi.clear()

        # Prefer depth-windowed view if present
        df = getattr(state, "analysis_df_view", None)
        if df is None or df.empty:
            df = getattr(state, "analysis_df", None)
        if df is None or df.empty:
            return



        # --- Resolve a curve by (1) state.params key, else (2) dataset families map, else (3) candidates ---
        p = getattr(state, "params", {}) or {}
        ds = getattr(state, "dataset", None)
        
        def resolve_curve(param_key: str, family_key: str, candidates: list[str]) -> str | None:
            # 1) curve picker selection
            v = p.get(param_key)
            if v and v in df.columns:
                return v
        
            # 2) dataset families map
            if ds is not None:
                fam_map = getattr(ds, "families_map", None)
                if isinstance(fam_map, dict) and family_key in fam_map:
                    for c in fam_map[family_key]:
                        if c in df.columns:
                            return c
        
            # 3) final fallback list
            for c in candidates:
                if c in df.columns:
                    return c
        
            return None

     
        # Debug tops presence (safe here)
        tops_df_dbg = getattr(state, "tops_df", None)
        print("[PlotsPanel] tops_df is None?", tops_df_dbg is None,
              "| n_tops =", 0 if tops_df_dbg is None else len(tops_df_dbg))

        params = getattr(state, "params", {}) or {}
        if not isinstance(params, dict):
            params = {}

        depth = df.index.to_numpy(dtype=float)

        def arr(name: str | None):
            if not name or name not in df.columns:
                return None
            return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)

        def first_present(cols, candidates):
            s = set(cols)
            for c in candidates:
                if c in s:
                    return c
            return None

        def get_curve(*keys, fallbacks=None):
            # 1) Try state.params
            for k in keys:
                v = params.get(k)
                if v:
                    return v
            # 2) Fallback to df columns
            if fallbacks:
                return first_present(df.columns, fallbacks)
            return None

        # -------------------------
        # Resolve curves (params -> fallback to df columns)
        # -------------------------
        gr_curve   = get_curve("gr_curve", "GR", fallbacks=["HSGR", "GR", "SGR", "HGR"])
        cgr_curve  = get_curve("cgr_curve", "CGR", fallbacks=["HCGR", "CGR"])

        tnph_curve = get_curve("tnph_curve", "TNPH", fallbacks=["TNPH", "NPHI", "CNL", "NPOR"])
        rhob_curve = get_curve("rhob_curve", "RHOB", fallbacks=["RHOZ", "RHOB"])

        # Resistivity candidates (your data has AT90/AF90)
      
        rt_curve = resolve_curve(
            param_key="rt_curve",
            family_key="RT",
            candidates=["AT90","AF90","AO90","ILD","RT","RES","RLA1","RDEP","RD","RXOZ","RXO8"]
        )
        print("[PlotsPanel] rt_curve resolved to:", rt_curve)

        
        # NMR porosity curves (often plotted on porosity track)
        phit_nmr_curve = get_curve("phit_nmr_curve", "phit_nmr", "PHIT_NMR", fallbacks=["PHIT_NMR", "TCMR", "MPHIS"])
        phie_nmr_curve = get_curve("phie_nmr_curve", "phie_nmr", "PHIE_NMR", fallbacks=["PHIE_NMR", "CMRP_3MS", "CMRP"])

        # NMR partition curves (NMR track)
        cbw_curve  = get_curve("cbw_curve", "cbw", "CBW", fallbacks=["CBW"])
        ffi_curve  = get_curve("ffi_curve", "ffi", "FFI", fallbacks=["FFI", "CMFF"])
        bvie_curve = get_curve("bvie_curve", "bvie", "BVIE", fallbacks=["BVIE", "BVI_E", "BVI"])

        # -------------------------
        # Plot helper
        # -------------------------
        def plot(track_key: str, x, pen, xlim=None, invert_x=False, logx=False, auto_x=False):
            if x is None:
                return

            m = np.isfinite(x) & np.isfinite(depth)

            # log-x requires x > 0
            if logx:
                m = m & (x > 0)

            if not np.any(m):
                return

            pi = self.track_items[track_key]
            pi.setLogMode(x=bool(logx), y=False)
            pi.getViewBox().invertX(invert_x)

            pi.plot(x[m], depth[m], pen=pen)

            if xlim is not None:
                pi.setXRange(xlim[0], xlim[1], padding=0.0)
            elif auto_x:
                pi.enableAutoRange(axis="x", enable=True)

        # -------------------------
        # Track 1: GR/CGR
        # -------------------------
        plot("gr", arr(gr_curve),  pg.mkPen("g", width=2), xlim=(0, 200))
        plot("gr", arr(cgr_curve), pg.mkPen("m", width=2), xlim=(0, 200))

        # -------------------------
        # Track 2: Porosity overlays
        # -------------------------
        plot("por", arr(tnph_curve),     pg.mkPen("g", width=1), xlim=(-0.15, 0.45), invert_x=True)
        plot("por", arr(phit_nmr_curve), pg.mkPen("k", width=1), xlim=(-0.15, 0.45), invert_x=True)
        plot("por", arr(phie_nmr_curve), pg.mkPen("b", width=1), xlim=(-0.15, 0.45), invert_x=True)

        # -------------------------
        # Track 3: Resistivity (log-x)
        # -------------------------
        #plot("rt", arr(rt_curve), pg.mkPen("k", width=2, style=Qt.DashLine), xlim=(0.2, 2000), logx=True)


     



        # -------------------------
        # Track 3: Resistivity (plot log10(Rt) on linear axis)
        # -------------------------
        rt_x = arr(rt_curve)
        
        if rt_x is None:
            print("[PlotsPanel] RT: curve missing in df:", rt_curve)
        else:
            m = np.isfinite(rt_x) & np.isfinite(depth) & (rt_x > 0)
        
            if not np.any(m):
                print("[PlotsPanel] RT: no finite positive values to plot",
                      "| finite =", int(np.isfinite(rt_x).sum()),
                      "| >0 =", int((rt_x > 0).sum()))
            else:
                rt_log = np.log10(rt_x[m])
                print("[PlotsPanel] RT log10 range:",
                      float(np.nanmin(rt_log)), float(np.nanmax(rt_log)),
                      "| N =", int(rt_log.size))
        
                pi = self.track_items["rt"]
                pi.setLogMode(x=False, y=False)      # make sure no log axis
                pi.getViewBox().invertX(False)
        
                pi.plot(rt_log, depth[m], pen=pg.mkPen("k", width=2))
                pi.setXRange(-1, 4, padding=0.0)     # 0.1..10,000 ohm-m
        

        
        # -------------------------
        # Track 4: NMR partition (line-only)
        # -------------------------
        plot("nmr", arr(cbw_curve),  pg.mkPen("k", width=1), xlim=(0.0, 0.30), invert_x=True)
        plot("nmr", arr(ffi_curve),  pg.mkPen("y", width=1), xlim=(0.0, 0.30), invert_x=True)
        plot("nmr", arr(bvie_curve), pg.mkPen("b", width=1), xlim=(0.0, 0.30), invert_x=True)

        # -------------------------
        # Tops overlay (expects state.tops_df with columns: Depth + Top)
        # -------------------------
        tops_df = getattr(state, "tops_df", None)
        if tops_df is not None and len(tops_df) > 0:
            # robust column detection
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
                for _, row in tops_df.iterrows():
                    d = row.get(depth_col, None)
                    if d is None or pd.isna(d):
                        continue
                    try:
                        y = float(d)
                    except Exception:
                        continue

                    label = None
                    if name_col is not None:
                        v = row.get(name_col, None)
                        if v is not None and not pd.isna(v):
                            label = str(v)

                    for pi in self.track_items.values():
                        pi.addLine(y=y, pen=pg.mkPen("r", width=2))

                    # label only on GR track to reduce clutter
                    if label:
                        gr_pi = self.track_items["gr"]
                        x_left = gr_pi.viewRange()[0][0]
                        t = pg.TextItem(label, anchor=(0, 1))
                        t.setPos(x_left, y)
                        gr_pi.addItem(t)



    # -------------------------
    # N–D crossplot
    # -------------------------
    def update_nd_crossplot(self, state):
        self.ax.clear()

        df = getattr(state, "analysis_df_view", None)
        if df is None or df.empty:
            df = getattr(state, "analysis_df", None)
        if df is None or df.empty:
            self.canvas.draw_idle()
            return

        params = getattr(state, "params", {}) or {}
        if not isinstance(params, dict):
            params = {}

        tnph = params.get("tnph_curve", None)
        rhob = params.get("rhob_curve", None)

        # fallback if params not set
        if not tnph or tnph not in df.columns:
            tnph = "TNPH" if "TNPH" in df.columns else ("NPHI" if "NPHI" in df.columns else None)
        if not rhob or rhob not in df.columns:
            rhob = "RHOZ" if "RHOZ" in df.columns else ("RHOB" if "RHOB" in df.columns else None)

        if tnph in df.columns and rhob in df.columns:
            x = pd.to_numeric(df[tnph], errors="coerce")
            y = pd.to_numeric(df[rhob], errors="coerce")
            m = x.notna() & y.notna()

            self.ax.scatter(x[m], y[m], s=6, alpha=0.6)
            self.ax.set_xlabel(tnph)
            self.ax.set_ylabel(rhob)
            self.ax.invert_yaxis()
            self.ax.grid(True, alpha=0.25)

        self.canvas.draw_idle()
