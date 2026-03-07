# petrocore/workflows/qt_zone_selector.py
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QDoubleSpinBox, QPushButton, QCheckBox, QMessageBox
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from petrocore.workflow.zone import (
    df_depth_limits_any,
    curve_depth_limits_any,
    slice_zone_df,
    build_units_map_for_df,
)
from petrocore.workflow.zone_plot import plot_zone_template_from_df


class ZoneSelectorDockWidget(QWidget):
    """
    Qt version of the ZOI picker.

    Requires:
      - dataset.merged_df: DataFrame
      - dataset.combined_units_map: dict (optional)
    Writes:
      - dataset.analysis_depth_range
      - dataset.analysis_df
    """
    zone_changed = Signal(float, float)  # emits (top, base) after update

    def __init__(self, dataset, *, depth_col: str = "DEPT", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.dataset = dataset
        self.depth_col = depth_col

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.timeout.connect(self._redraw_plot)

        self._fig = Figure(figsize=(6, 8), dpi=100)
        self._canvas = FigureCanvas(self._fig)

        # -----------------------------
        # Controls
        # -----------------------------
        self.lbl_title = QLabel("ZOI (Zone of Interest)")
        self.lbl_title.setStyleSheet("font-weight: 700; font-size: 14px;")

        self.top_spin = QDoubleSpinBox()
        self.base_spin = QDoubleSpinBox()
        for sp in (self.top_spin, self.base_spin):
            sp.setDecimals(1)
            sp.setSingleStep(0.5)
            sp.setKeyboardTracking(False)  # update on enter/focus-out

        self.auto_from_phitnmr = QCheckBox("Auto-range from PHIT_NMR (if present)")
        self.auto_from_phitnmr.setChecked(True)

        self.btn_apply = QPushButton("Set ZOI")
        self.btn_save = QPushButton("Save Plot PNG")

        # Wire events
        self.top_spin.valueChanged.connect(self._on_depth_changed)
        self.base_spin.valueChanged.connect(self._on_depth_changed)
        self.auto_from_phitnmr.stateChanged.connect(self._auto_init_range)
        self.btn_apply.clicked.connect(self._apply_zone_to_dataset)
        self.btn_save.clicked.connect(self._save_plot)

        # Layout
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Top:"))
        row1.addWidget(self.top_spin, 1)
        row1.addWidget(QLabel("Base:"))
        row1.addWidget(self.base_spin, 1)

        row2 = QHBoxLayout()
        row2.addWidget(self.btn_apply)
        row2.addWidget(self.btn_save)

        v = QVBoxLayout(self)
        v.addWidget(self.lbl_title)
        v.addLayout(row1)
        v.addWidget(self.auto_from_phitnmr)
        v.addLayout(row2)
        v.addWidget(self._canvas, 1)

        # Init ranges + first plot
        self._init_spin_limits()
        self._auto_init_range()
        self._redraw_plot()

    # -----------------------------
    # Init helpers
    # -----------------------------
    def _init_spin_limits(self):
        df = self.dataset.merged_df
        if df is None or df.empty:
            raise ValueError("Dataset.merged_df is empty; cannot build ZOI selector.")

        dmin, dmax = df_depth_limits_any(df, depth_col=self.depth_col)

        # Keep spins valid even if depth is decreasing somewhere (we slice by values anyway)
        self.top_spin.setRange(dmin, dmax)
        self.base_spin.setRange(dmin, dmax)

    def _auto_init_range(self):
        df = self.dataset.merged_df
        if df is None or df.empty:
            return

        dmin, dmax = df_depth_limits_any(df, depth_col=self.depth_col)

        if self.auto_from_phitnmr.isChecked() and ("PHIT_NMR" in df.columns):
            try:
                zmin, zmax = curve_depth_limits_any(df, "PHIT_NMR", depth_col=self.depth_col)
            except Exception:
                zmin, zmax = dmin, dmax
        else:
            zmin, zmax = dmin, dmax

        # set without looping redraw too much
        self.top_spin.blockSignals(True)
        self.base_spin.blockSignals(True)
        self.top_spin.setValue(float(min(zmin, zmax)))
        self.base_spin.setValue(float(max(zmin, zmax)))
        self.top_spin.blockSignals(False)
        self.base_spin.blockSignals(False)

        self._queue_redraw()

    # -----------------------------
    # Callbacks
    # -----------------------------
    def _on_depth_changed(self, *_):
        # Enforce top <= base
        top = float(self.top_spin.value())
        base = float(self.base_spin.value())
        if top > base:
            # swap
            self.top_spin.blockSignals(True)
            self.base_spin.blockSignals(True)
            self.top_spin.setValue(base)
            self.base_spin.setValue(top)
            self.top_spin.blockSignals(False)
            self.base_spin.blockSignals(False)
        self._queue_redraw()

    def _queue_redraw(self):
        self._debounce.start(150)

    # -----------------------------
    # Main actions
    # -----------------------------
    def _apply_zone_to_dataset(self):
        df = self.dataset.merged_df
        if df is None or df.empty:
            QMessageBox.warning(self, "ZOI", "merged_df is empty.")
            return

        top = float(self.top_spin.value())
        base = float(self.base_spin.value())
        top, base = (top, base) if top <= base else (base, top)

        z = slice_zone_df(df, top, base, depth_col=self.depth_col)

        self.dataset.analysis_depth_range = (top, base)
        self.dataset.analysis_df = z

        self.zone_changed.emit(top, base)

        QMessageBox.information(
            self, "ZOI Set",
            f"analysis_depth_range = ({top:.1f}, {base:.1f})\nanalysis_df shape = {z.shape}"
        )

    def _redraw_plot(self):
        df = self.dataset.merged_df
        if df is None or df.empty:
            return

        top = float(self.top_spin.value())
        base = float(self.base_spin.value())
        top, base = (top, base) if top <= base else (base, top)

        base_units = getattr(self.dataset, "combined_units_map", {}) or {}
        units = build_units_map_for_df(df, base_units=base_units)

        # Make/replace figure cleanly
        self._fig.clf()

        try:
            # Use your existing plotter (matplotlib Figure returned)
            fig2 = plot_zone_template_from_df(
                df, units, top, base,
                title="Zone Plot — merged_df",
                depth_col=self.depth_col,
                tops_depths=(globals().get("tops_depths", None)),
                tops=(globals().get("tops", None)),
            )

            # Copy artists by drawing fig2 onto our canvas:
            # easiest reliable approach: just replace canvas figure
            self._fig = fig2
            self._canvas.figure = self._fig
            self._canvas.draw()

        except Exception as e:
            # fallback message
            ax = self._fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Plot failed:\n{e}", ha="center", va="center")
            ax.set_axis_off()
            self._canvas.draw()

    def _save_plot(self):
        df = self.dataset.merged_df
        if df is None or df.empty:
            QMessageBox.warning(self, "Save Plot", "merged_df is empty.")
            return

        top = float(self.top_spin.value())
        base = float(self.base_spin.value())
        top, base = (top, base) if top <= base else (base, top)

        plots_dir = "plots"
        os.makedirs(plots_dir, exist_ok=True)
        fname = f"ZonePlot_{int(top)}-{int(base)}ft.png"
        outfile = os.path.join(plots_dir, fname)

        try:
            self._canvas.figure.savefig(outfile, dpi=300, bbox_inches="tight", facecolor="white")
            QMessageBox.information(self, "Saved", f"Saved:\n{outfile}")
        except Exception as e:
            QMessageBox.warning(self, "Save failed", str(e))
