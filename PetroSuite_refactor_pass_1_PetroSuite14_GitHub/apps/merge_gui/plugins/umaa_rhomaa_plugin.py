from __future__ import annotations

from typing import Any
import pandas as pd

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QLabel, QPushButton,
    QComboBox, QDoubleSpinBox, QMessageBox,
)

from .base_plugin import BasePlugin
from petrocore.config.curve_aliases import best_curve
from petrocore.workflow.umaa_rhomaa import compute_umaa_rhomaa
from petrocore.workflow.porosity_engine import WorkflowResult as PluginResult


class UmaaEngine:
    """Thin plugin wrapper around petrocore. Qt stays in apps; math stays in petrocore."""

    def compute(self, df: pd.DataFrame, curve_map: dict[str, str], params: dict[str, Any]) -> PluginResult:
        return compute_umaa_rhomaa(df, curve_map, params)


class UmaaWidget(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.engine = UmaaEngine()
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        title = QLabel("UMAA-RHOMAA Calculation")
        title.setStyleSheet("font-size: 12pt; font-weight: bold;")
        layout.addWidget(title)

        form = QFormLayout()
        self.rhob_combo = QComboBox()
        self.pef_combo = QComboBox()
        self.phit_combo = QComboBox()

        df = self.controller.get_analysis_df()
        cols = list(df.columns) if df is not None else []
        self.rhob_combo.addItems(cols)
        self.pef_combo.addItems(cols)
        self.phit_combo.addItems(cols)

        for combo, pick in [
            (self.rhob_combo, best_curve(cols, "RHOB")),
            (self.pef_combo, best_curve(cols, "PEF")),
            (self.phit_combo, best_curve(cols, "PHIT")),
        ]:
            if pick is not None:
                idx = combo.findText(pick)
                if idx >= 0:
                    combo.setCurrentIndex(idx)

        form.addRow("Density (RHOB):", self.rhob_combo)
        form.addRow("PEF:", self.pef_combo)
        form.addRow("PHIT:", self.phit_combo)

        self.fluid_density = QDoubleSpinBox()
        self.fluid_density.setRange(0.5, 2.0)
        self.fluid_density.setDecimals(3)
        self.fluid_density.setSingleStep(0.01)
        self.fluid_density.setValue(1.10)
        form.addRow("Fluid Density:", self.fluid_density)

        layout.addLayout(form)
        run_btn = QPushButton("Compute UMAA-RHOMAA")
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
            "PEF": self.pef_combo.currentText().strip(),
            "PHIT": self.phit_combo.currentText().strip(),
        }
        params = {"fluid_density": self.fluid_density.value()}

        try:
            result = self.engine.compute(df, curve_map, params)
            self.controller.update_analysis_df(result.df)
            QMessageBox.information(self, "Success", "\n".join(result.messages))
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))


class UmaaPlugin(BasePlugin):
    plugin_id = "umaa-rhomaa"
    display_name = "UMAA-RHOMAA"
    tooltip = "Compute UMAA/RHOMAA from density, PEF, and PHIT logs"

    def launch(self, controller):
        widget = UmaaWidget(controller)
        controller.set_center_widget(widget, self.display_name)
