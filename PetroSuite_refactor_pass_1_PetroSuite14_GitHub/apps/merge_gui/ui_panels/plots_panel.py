from __future__ import annotations

print(">>> LOADING plots_panel.py from:", __file__)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QTextEdit,
    QMessageBox,
)


class PlotsPanel(QWidget):
    """
    Merge-oriented plots panel.

    Purpose
    -------
    - View loaded LAS runs
    - Choose a curve to compare across runs
    - Plot curve vs depth for QC
    - Avoid dependencies on old petrophysical workflow modules
    """

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.fig = plt.Figure(figsize=(10, 10))
        self.canvas = FigureCanvas(self.fig)
        self._build_ui()

    # ---------------------------------------------------------------------
    # Convenience
    # ---------------------------------------------------------------------

    def _state(self):
        return self.controller.get_state()

    # ---------------------------------------------------------------------
    # UI
    # ---------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)

        title = QLabel("Depth Plots / Merge QC")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        subtitle = QLabel(
            "Select a curve and compare it across loaded LAS runs to inspect depth alignment and overlap."
        )
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        controls = QHBoxLayout()

        self.curve_combo = QComboBox()
        self.curve_combo.setMinimumWidth(240)

        self.btn_refresh_curves = QPushButton("Refresh Curves")
        self.btn_refresh_curves.clicked.connect(self._populate_curve_list)

        self.btn_plot = QPushButton("Plot Selected Curve")
        self.btn_plot.clicked.connect(self._plot_selected_curve)

        self.btn_plot_default = QPushButton("Plot Default QC Curve")
        self.btn_plot_default.clicked.connect(self._plot_default_curve)

        controls.addWidget(QLabel("Curve:"))
        controls.addWidget(self.curve_combo)
        controls.addWidget(self.btn_refresh_curves)
        controls.addWidget(self.btn_plot)
        controls.addWidget(self.btn_plot_default)
        controls.addStretch()

        layout.addLayout(controls)
        layout.addWidget(self.canvas, stretch=1)

        layout.addWidget(QLabel("Status / Details:"))
        self.info_box = QTextEdit()
        self.info_box.setReadOnly(True)
        self.info_box.setMinimumHeight(120)
        layout.addWidget(self.info_box)

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    def _append_info(self, text: str):
        if text:
            self.info_box.append(text)

    def _get_runs(self):
        state = self._state()
        runs = getattr(state, "las_runs", [])
        return runs if isinstance(runs, list) else []

    def _get_all_curve_names(self):
        runs = self._get_runs()
        names = []
        seen = set()

        for run in runs:
            df = run.get("df")
            if isinstance(df, pd.DataFrame):
                for col in df.columns:
                    if col not in seen:
                        seen.add(col)
                        names.append(col)

        return names

    def _find_depth_col(self, df: pd.DataFrame) -> str | None:
        candidates = ["DEPT", "DEPTH", "MD"]
        upper_map = {str(c).upper(): c for c in df.columns}

        for c in candidates:
            if c in upper_map:
                return upper_map[c]

        # fallback: if index is numeric-ish, we can use that later
        return None

    def _find_default_qc_curve(self) -> str | None:
        """
        Pick a sensible default curve for QC.
        Priority: GR-family, then resistivity, then density, then first non-depth.
        """
        curve_names = self._get_all_curve_names()
        upper = {c.upper(): c for c in curve_names}

        preferred = [
            "GR_EDTC", "SGR", "GR", "HSGR", "HCGR", "CGR",
            "RT", "ILD", "LLD", "AT90", "AF90",
            "RHOB", "RHOZ", "NPHI", "TNPH",
        ]

        for p in preferred:
            if p in upper:
                return upper[p]

        for c in curve_names:
            if str(c).upper() not in {"DEPT", "DEPTH", "MD"}:
                return c

        return None

    # ---------------------------------------------------------------------
    # Curve list population
    # ---------------------------------------------------------------------

    def _populate_curve_list(self):
        current = self.curve_combo.currentText()
        self.curve_combo.clear()

        curve_names = self._get_all_curve_names()
        self.curve_combo.addItems(curve_names)

        if current and current in curve_names:
            self.curve_combo.setCurrentText(current)

        self._append_info(f"Found {len(curve_names)} unique curves across loaded runs.")

    # ---------------------------------------------------------------------
    # Plotting
    # ---------------------------------------------------------------------

    def _plot_selected_curve(self):
        curve = self.curve_combo.currentText().strip()
        if not curve:
            QMessageBox.information(self, "No Curve", "Please select a curve to plot.")
            return
        self._plot_curve(curve)

    def _plot_default_curve(self):
        curve = self._find_default_qc_curve()
        if not curve:
            QMessageBox.information(self, "No Curve", "No suitable default curve was found.")
            return

        if self.curve_combo.findText(curve) >= 0:
            self.curve_combo.setCurrentText(curve)

        self._plot_curve(curve)

    def _plot_curve(self, curve_name: str):
        runs = self._get_runs()

        self.fig.clear()
        ax = self.fig.add_subplot(111)

        if not runs:
            ax.text(0.5, 0.5, "No LAS runs loaded", ha="center", va="center")
            ax.set_axis_off()
            self.canvas.draw_idle()
            return

        plotted = 0
        messages = [f"Plotting curve: {curve_name}"]

        for i, run in enumerate(runs, start=1):
            df = run.get("df")
            file_name = run.get("file_name", f"Run {i}")

            if not isinstance(df, pd.DataFrame) or df.empty:
                messages.append(f"{file_name}: skipped (no dataframe)")
                continue

            depth_col = self._find_depth_col(df)

            if depth_col is not None:
                depth = pd.to_numeric(df[depth_col], errors="coerce").to_numpy()
            else:
                try:
                    depth = pd.to_numeric(pd.Index(df.index), errors="coerce").to_numpy()
                except Exception:
                    messages.append(f"{file_name}: skipped (no valid depth)")
                    continue

            if curve_name not in df.columns:
                messages.append(f"{file_name}: curve not present")
                continue

            x = pd.to_numeric(df[curve_name], errors="coerce").to_numpy()

            mask = np.isfinite(depth) & np.isfinite(x)
            if mask.sum() < 2:
                messages.append(f"{file_name}: insufficient valid data")
                continue

            ax.plot(x[mask], depth[mask], linewidth=0.9, label=file_name)
            plotted += 1

            messages.append(
                f"{file_name}: plotted {mask.sum():,} samples"
            )

        if plotted == 0:
            ax.text(0.5, 0.5, f"No valid data found for curve '{curve_name}'",
                    ha="center", va="center")
            ax.set_axis_off()
        else:
            ax.set_title(f"{curve_name} vs Depth")
            ax.set_xlabel(curve_name)
            ax.set_ylabel("Depth")
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=8)

        self.canvas.draw_idle()
        self.info_box.clear()
        self._append_info("\n".join(messages))

    # ---------------------------------------------------------------------
    # Refresh hook
    # ---------------------------------------------------------------------

    def refresh(self):
        self._populate_curve_list()

        # Draw something sensible if nothing has been drawn yet
        if self.fig.axes == []:
            self._plot_default_curve()