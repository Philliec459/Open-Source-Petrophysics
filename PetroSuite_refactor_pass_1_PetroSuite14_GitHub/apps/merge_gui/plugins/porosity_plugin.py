from __future__ import annotations

from typing import Any
import pandas as pd

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QLabel, QPushButton,
    QComboBox, QDoubleSpinBox, QMessageBox,
)

from .base_plugin import BasePlugin
from petrocore.config.curve_aliases import best_curve
from petrocore.workflow.porosity_engine import compute_porosity, WorkflowResult as PluginResult


class PorosityEngine:
    """Thin plugin wrapper around petrocore. Qt stays in apps; math stays in petrocore."""

    def compute(self, df: pd.DataFrame, curve_map: dict[str, str], params: dict[str, Any]) -> PluginResult:
        return compute_porosity(df, curve_map, params)


class PorosityWidget(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.engine = PorosityEngine()
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        title = QLabel("Porosity Calculation")
        title.setStyleSheet("font-size: 12pt; font-weight: bold;")
        layout.addWidget(title)

        form = QFormLayout()

        self.method_combo = QComboBox()
        self.method_combo.addItems(["density_neutron", "density_only", "neutron_only"])
        form.addRow("Method:", self.method_combo)

        self.rhob_combo = QComboBox()
        self.nphi_combo = QComboBox()

        df = self.controller.get_analysis_df()
        cols = list(df.columns) if df is not None else []
        self.rhob_combo.addItems(cols)
        self.nphi_combo.addItems(cols)

        rhob_pick = best_curve(cols, "RHOB")
        nphi_pick = best_curve(cols, "TNPH")

        if rhob_pick is not None:
            idx = self.rhob_combo.findText(rhob_pick)
            if idx >= 0:
                self.rhob_combo.setCurrentIndex(idx)
        if nphi_pick is not None:
            idx = self.nphi_combo.findText(nphi_pick)
            if idx >= 0:
                self.nphi_combo.setCurrentIndex(idx)

        form.addRow("Density (RHOB):", self.rhob_combo)
        form.addRow("Neutron (NPHI/TNPH):", self.nphi_combo)

        self.matrix_density = QDoubleSpinBox()
        self.matrix_density.setRange(1.0, 4.0)
        self.matrix_density.setDecimals(3)
        self.matrix_density.setSingleStep(0.01)
        self.matrix_density.setValue(2.71)

        self.fluid_density = QDoubleSpinBox()
        self.fluid_density.setRange(0.5, 2.0)
        self.fluid_density.setDecimals(3)
        self.fluid_density.setSingleStep(0.01)
        self.fluid_density.setValue(1.10)

        form.addRow("Matrix Density:", self.matrix_density)
        form.addRow("Fluid Density:", self.fluid_density)
        layout.addLayout(form)

        run_btn = QPushButton("Compute Porosity")
        run_btn.clicked.connect(self._run)
        layout.addWidget(run_btn)
        layout.addStretch()

    def _run(self):
        df = self.controller.get_analysis_df()
        if df is None or df.empty:
            QMessageBox.warning(self, "Error", "No data loaded.")
            return

        curve_map = {
            "RHOB": self.rhob_combo.currentText().strip(),
            "NPHI": self.nphi_combo.currentText().strip(),
        }
        params = {
            "method": self.method_combo.currentText(),
            "matrix_density": self.matrix_density.value(),
            "fluid_density": self.fluid_density.value(),
        }

        try:
            result = self.engine.compute(df, curve_map, params)
            self.controller.update_analysis_df(result.df)
            QMessageBox.information(self, "Success", "\n".join(result.messages))
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))


class PorosityPlugin(BasePlugin):
    plugin_id = "porosity"
    display_name = "Porosity"
    tooltip = "Compute porosity from density and neutron logs"

    def launch(self, controller):
        widget = PorosityWidget(controller)
        controller.set_center_widget(widget, self.display_name)
