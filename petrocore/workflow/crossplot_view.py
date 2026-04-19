import numpy as np
import pandas as pd

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import Normalize, LogNorm

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton
)

from .xplot_helpers import (
    get_preset,
    get_chart_def,
    load_chart_df,
    get_z_config,
    compute_z_range,
)


class CrossplotView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.df = None
        self.current_preset = None

        self._build_ui()

    # --------------------------------------------------
    # UI
    # --------------------------------------------------
    def _build_ui(self):
        layout = QVBoxLayout(self)

        # --- Top controls ---
        top = QHBoxLayout()

        self.title_label = QLabel("No crossplot loaded")

        self.x_combo = QComboBox()
        self.y_combo = QComboBox()
        self.z_combo = QComboBox()

        self.draw_btn = QPushButton("Redraw")
        self.draw_btn.clicked.connect(self.redraw_plot)

        top.addWidget(QLabel("Plot:"))
        top.addWidget(self.title_label)
        top.addStretch(1)
        top.addWidget(QLabel("X"))
        top.addWidget(self.x_combo)
        top.addWidget(QLabel("Y"))
        top.addWidget(self.y_combo)
        top.addWidget(QLabel("Z"))
        top.addWidget(self.z_combo)
        top.addWidget(self.draw_btn)

        layout.addLayout(top)

        # --- Matplotlib canvas ---
        self.figure = Figure(figsize=(6, 6))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        layout.addWidget(self.canvas)

    # --------------------------------------------------
    # DATA
    # --------------------------------------------------
    def set_dataframe(self, df):
        self.df = df
        self._populate_combos()

    def _populate_combos(self):
        self.x_combo.clear()
        self.y_combo.clear()
        self.z_combo.clear()

        if self.df is None or self.df.empty:
            return

        numeric_cols = [
            c for c in self.df.columns
            if pd.api.types.is_numeric_dtype(self.df[c])
        ]

        self.x_combo.addItems(numeric_cols)
        self.y_combo.addItems(numeric_cols)
        self.z_combo.addItems([""] + numeric_cols)

    def _set_best_combo(self, combo, candidates):
        available = [combo.itemText(i) for i in range(combo.count())]

        for c in candidates:
            if c in available:
                combo.setCurrentText(c)
                return

    # --------------------------------------------------
    # PRESETS
    # --------------------------------------------------
    
    def plot_histogram(self, df, preset):
        import numpy as np
    
        x_col = preset.get("x_col")
        if not x_col or x_col not in df.columns:
            print(f"[Crossplot] histogram column not found: {x_col}")
            return
    
        data = pd.to_numeric(df[x_col], errors="coerce").dropna()
        if data.empty:
            print(f"[Crossplot] no valid data for histogram: {x_col}")
            return
    
        bins = preset.get("bins", 30)
    
        self.ax.clear()
        self.ax.hist(data, bins=bins, edgecolor="black")
    
        self.ax.set_xlabel(preset.get("x_label", x_col))
        self.ax.set_ylabel(preset.get("y_label", "Frequency"))
    
        if preset.get("xlim") is not None:
            self.ax.set_xlim(*preset["xlim"])
    
        if preset.get("ylim") is not None:
            self.ax.set_ylim(*preset["ylim"])
    
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw_idle()




    
    def load_preset(self, name, df):
        self.current_preset = name
        self.set_dataframe(df)
    
        preset = get_preset(name)
        self.current_preset_def = preset
    
        self.title_label.setText(preset.get("title", preset.get("name", name)))
    
        # histogram presets usually only need x
        if preset.get("plot_type") == "histogram":
            self._set_best_combo(self.x_combo, preset.get("x_curve_candidates", [preset.get("x_col", "")]))
            self._set_best_combo(self.y_combo, [])
            self._set_best_combo(self.z_combo, [])
        else:
            # set default curves
            self._set_best_combo(self.x_combo, preset.get("x_curve_candidates", []))
            self._set_best_combo(self.y_combo, preset.get("y_curve_candidates", []))
            self._set_best_combo(self.z_combo, preset.get("z_curve_candidates", []))
    
        self.redraw_plot()
        

    # --------------------------------------------------
    # PLOTTING
    # --------------------------------------------------
        
        
    def redraw_plot(self):
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
    
        if self.df is None or self.df.empty:
            self.ax.set_title("No data loaded")
            self.canvas.draw_idle()
            return
    
        # -------------------------------------------------
        # Current preset
        # -------------------------------------------------
        preset = None
        if self.current_preset:
            try:
                preset = get_preset(self.current_preset)
            except Exception as e:
                print("[Crossplot] preset lookup failed:", e)
                preset = None
    
        is_histogram = preset and preset.get("plot_type") == "histogram"
    
        # -------------------------------------------------
        # Histogram mode
        # -------------------------------------------------
        if is_histogram:
            x_name = self.x_combo.currentText().strip()
    
            if not x_name:
                self.ax.set_title("Select X curve for histogram")
                self.canvas.draw_idle()
                return
    
            if x_name not in self.df.columns:
                self.ax.set_title(f"Column not found: {x_name}")
                self.canvas.draw_idle()
                return
    
            x = pd.to_numeric(self.df[x_name], errors="coerce").dropna()
    
            if x.empty:
                self.ax.set_title("No valid data")
                self.canvas.draw_idle()
                return
    
            bins = preset.get("bins", 30)
    
            self.ax.hist(x, bins=bins, edgecolor="black")
            self.ax.set_xlabel(x_name)
            self.ax.set_ylabel("Frequency")
            self.ax.set_title(f"{x_name} Histogram")
            self.ax.grid(True, alpha=0.3)
    
            self.canvas.draw_idle()
            return
    
        # -------------------------------------------------
        # Normal X-Y crossplot
        # -------------------------------------------------
        x_name = self.x_combo.currentText().strip()
        y_name = self.y_combo.currentText().strip()
        z_name = self.z_combo.currentText().strip()
    
        if not x_name or not y_name:
            self.ax.set_title("Select X and Y curves")
            self.canvas.draw_idle()
            return
    
        if x_name not in self.df.columns or y_name not in self.df.columns:
            self.ax.set_title("Selected X or Y curve not found in dataframe")
            self.canvas.draw_idle()
            return
    
        x = pd.to_numeric(self.df[x_name], errors="coerce")
        y = pd.to_numeric(self.df[y_name], errors="coerce")
    
        mask = np.isfinite(x) & np.isfinite(y)
    
        chart_def = None
    
        # Only attempt chart_def lookup for non-histogram presets with chart_number
        if preset and not is_histogram and preset.get("chart_number") is not None:
            try:
                chart_def = get_chart_def(preset["chart_number"])
            except Exception as e:
                print("[Crossplot] chart definition lookup failed:", e)
                chart_def = None
    
        # -------------------------------------------------
        # Safe log filtering
        # -------------------------------------------------
        log_x = chart_def.get("log_x", False) if chart_def else False
        log_y = chart_def.get("log_y", False) if chart_def else False
    
        if log_x:
            mask &= (x > 0)
        if log_y:
            mask &= (y > 0)
    
        if z_name and z_name in self.df.columns:
            z = pd.to_numeric(self.df[z_name], errors="coerce")
            mask &= np.isfinite(z)
            z = z[mask]
        else:
            z = None
    
        x = x[mask]
        y = y[mask]
    
        if len(x) == 0:
            self.ax.set_title("No valid data")
            self.canvas.draw_idle()
            return
    
        # -------------------------------------------------
        # Scatter
        # -------------------------------------------------
        if z is not None:
            try:
                z_cfg = get_z_config(z_name)
                zmin, zmax = compute_z_range(z, z_cfg)
    
                if zmin is not None and zmax is not None:
                    norm = LogNorm(vmin=zmin, vmax=zmax) if z_cfg["scale"] == "log" else Normalize(vmin=zmin, vmax=zmax)
                    sc = self.ax.scatter(x, y, c=z, cmap="rainbow", norm=norm, s=16, alpha=0.8)
                    self.figure.colorbar(sc, ax=self.ax, label=z_name)
                else:
                    self.ax.scatter(x, y, s=16, alpha=0.8)
            except Exception as e:
                print("[Crossplot] z-color failed:", e)
                self.ax.scatter(x, y, s=16, alpha=0.8)
        else:
            self.ax.scatter(x, y, s=16, alpha=0.8)
    
        # -------------------------------------------------
        # Chart overlay
        # -------------------------------------------------
        if chart_def and preset and not is_histogram and preset.get("chart_number") is not None:
            try:
                chart_df = load_chart_df(preset["chart_number"])
    
                if chart_def["x_col"] in chart_df.columns and chart_def["y_col"] in chart_df.columns:
                    chart_x = pd.to_numeric(chart_df[chart_def["x_col"]], errors="coerce").copy()
                    chart_y = pd.to_numeric(chart_df[chart_def["y_col"]], errors="coerce").copy()
    
                    # Keep NaNs so blank rows break the line
                    if chart_def.get("log_x", False):
                        chart_x[chart_x <= 0] = np.nan
                    if chart_def.get("log_y", False):
                        chart_y[chart_y <= 0] = np.nan
    
                    self.ax.plot(
                        chart_x.to_numpy(),
                        chart_y.to_numpy(),
                        color="magenta",
                        linewidth=2.5
                    )
    
                # fixed mineral/reference points
                for label, xpt, ypt in chart_def.get("points", []):
                    if chart_def.get("log_x", False) and xpt <= 0:
                        continue
                    if chart_def.get("log_y", False) and ypt <= 0:
                        continue
    
                    self.ax.scatter(
                        [xpt], [ypt],
                        s=80,
                        marker="^",
                        facecolors="none",
                        edgecolors="magenta",
                        linewidths=1.2,
                        zorder=5,
                    )
                    self.ax.text(
                        xpt, ypt, f" {label}",
                        fontsize=9,
                        color="magenta",
                        va="center",
                        zorder=6,
                    )
    
                self.ax.set_xlabel(chart_def.get("x_label", x_name))
                self.ax.set_ylabel(chart_def.get("y_label", y_name))
    
                if chart_def.get("log_x", False):
                    self.ax.set_xscale("log")
                if chart_def.get("log_y", False):
                    self.ax.set_yscale("log")
    
                if chart_def.get("xlim") is not None:
                    self.ax.set_xlim(chart_def["xlim"])
                if chart_def.get("ylim") is not None:
                    self.ax.set_ylim(chart_def["ylim"])
    
                if chart_def.get("reverse_x", False):
                    self.ax.invert_xaxis()
                if chart_def.get("reverse_y", False):
                    self.ax.invert_yaxis()
    
            except Exception as e:
                print("[Crossplot] chart overlay failed:", e)
                self.ax.set_xlabel(x_name)
                self.ax.set_ylabel(y_name)
    
        else:
            self.ax.set_xlabel(x_name)
            self.ax.set_ylabel(y_name)
    
        self.ax.set_title(f"{x_name} vs {y_name}")
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw_idle()    
        
    
        
    
    
    

    

