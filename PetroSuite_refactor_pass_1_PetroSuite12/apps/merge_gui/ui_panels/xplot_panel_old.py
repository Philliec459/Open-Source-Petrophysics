import numpy as np
import pandas as pd

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QListWidget, QLabel, QComboBox, QPushButton
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class XPlotPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.df = None
        self.chart_registry = {}

        self._build_ui()

    # --------------------------------------------------
    # UI
    # --------------------------------------------------
    def _build_ui(self):
        layout = QHBoxLayout(self)

        # -----------------------------
        # LEFT: launcher list
        # -----------------------------
        self.launch_list = QListWidget()
        self.launch_list.addItems([
            "Neutron-Density",
            "Sonic-Neutron",
            "Sonic-Density",
            "PEF-Bulk Density",
            "UMAA-RHOMAA",
            "Histogram",
            "Generic XY"
        ])
        self.launch_list.itemClicked.connect(self._on_launch_clicked)

        layout.addWidget(self.launch_list, 1)

        # -----------------------------
        # RIGHT: controls + plot
        # -----------------------------
        right = QVBoxLayout()

        # Curve pickers
        row = QHBoxLayout()
        self.x_combo = QComboBox()
        self.y_combo = QComboBox()
        self.z_combo = QComboBox()

        row.addWidget(QLabel("X:"))
        row.addWidget(self.x_combo)
        row.addWidget(QLabel("Y:"))
        row.addWidget(self.y_combo)
        row.addWidget(QLabel("Z:"))
        row.addWidget(self.z_combo)

        right.addLayout(row)

        # redraw button
        self.btn_draw = QPushButton("Draw Plot")
        self.btn_draw.clicked.connect(self.redraw_plot)
        right.addWidget(self.btn_draw)

        # matplotlib canvas
        self.fig = Figure(figsize=(5, 5))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)

        right.addWidget(self.canvas, 5)

        layout.addLayout(right, 4)

    # --------------------------------------------------
    # DATA
    # --------------------------------------------------
    def set_dataframe(self, df: pd.DataFrame):
        self.df = df
        self._populate_curve_combos()

    def set_chart_registry(self, chart_registry: dict):
        """
        Example:
        {
            "neutron_density": (chart_x, chart_y),
            ...
        }
        """
        self.chart_registry = chart_registry

    def _populate_curve_combos(self):
        if self.df is None:
            return

        cols = [c for c in self.df.columns
                if pd.api.types.is_numeric_dtype(self.df[c])]

        self.x_combo.clear()
        self.y_combo.clear()
        self.z_combo.clear()

        self.x_combo.addItems(cols)
        self.y_combo.addItems(cols)
        self.z_combo.addItems([""] + cols)

    # --------------------------------------------------
    # LAUNCHERS
    # --------------------------------------------------
    def _on_launch_clicked(self, item):
        name = item.text()

        if name == "Neutron-Density":
            self._load_neutron_density()

        elif name == "Generic XY":
            self.redraw_plot()

    def _load_neutron_density(self):
        # auto-pick curves
        self._set_if_exists(self.x_combo, ["NPHI", "TNPH", "CNL"])
        self._set_if_exists(self.y_combo, ["RHOB", "DEN"])

        self._set_if_exists(self.z_combo, ["VCLAY", "VILL", "PHIT"])

        self.current_overlay = "neutron_density"

        self.redraw_plot()

    def _set_if_exists(self, combo, options):
        for opt in options:
            if opt in [combo.itemText(i) for i in range(combo.count())]:
                combo.setCurrentText(opt)
                return

    # --------------------------------------------------
    # PLOT ENGINE
    # --------------------------------------------------
    def redraw_plot(self):
        if self.df is None:
            return

        x_name = self.x_combo.currentText()
        y_name = self.y_combo.currentText()
        z_name = self.z_combo.currentText()

        if not x_name or not y_name:
            return

        x = self.df[x_name].values
        y = self.df[y_name].values

        z = None
        if z_name:
            z = self.df[z_name].values

        self.ax.clear()

        # overlay chart if exists
        if hasattr(self, "current_overlay"):
            if self.current_overlay in self.chart_registry:
                cx, cy = self.chart_registry[self.current_overlay]
                self.ax.plot(cx, cy, 'm-', linewidth=1.5)

        # scatter
        if z is not None:
            sc = self.ax.scatter(x, y, s=15, c=z, cmap="rainbow")
        else:
            sc = self.ax.scatter(x, y, s=15)

        self.ax.set_title(f"{x_name} vs {y_name}")
        self.ax.set_xlabel(x_name)
        self.ax.set_ylabel(y_name)
        self.ax.grid(True)

        self.canvas.draw()