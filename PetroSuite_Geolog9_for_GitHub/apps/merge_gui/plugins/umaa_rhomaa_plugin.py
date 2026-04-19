from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np
import pandas as pd

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QFormLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QDoubleSpinBox,
    QMessageBox,
)

from .base_plugin import BasePlugin


# ============================================================
# Result container
# ============================================================

@dataclass
class PluginResult:
    df: pd.DataFrame
    outputs: list[str]
    messages: list[str] = field(default_factory=list)


# ============================================================
# Helpers
# ============================================================

RHOB_CANDS = ["RHOZ", "RHOB", "DEN", "DENS", "ZDEN"]
PEF_CANDS = ["PEF", "PEFZ", "PE"]
PHIT_CANDS = ["PHIT"]


def _find_curve_case_insensitive(cols, candidates):
    if cols is None:
        return None

    cols_list = list(cols)
    cols_upper = {str(c).upper(): c for c in cols_list}

    # exact
    for cand in candidates:
        if cand in cols_list:
            return cand

    # case-insensitive exact
    for cand in candidates:
        cu = cand.upper()
        if cu in cols_upper:
            return cols_upper[cu]

    # substring fallback
    for cand in candidates:
        cu = cand.upper()
        for col in cols_list:
            if cu in str(col).upper():
                return col

    return None


# ============================================================
# Core compute engine
# ============================================================

class UmaaEngine:
    def compute(self, df: pd.DataFrame, curve_map: dict[str, str], params: dict[str, Any]) -> PluginResult:
        out = df.copy()

        #method = params.get("method", "standard")
        #matrix_density = float(params.get("matrix_density", 2.71))
        fluid_density = float(params.get("fluid_density", 1.10))

        rhob_curve = curve_map.get("RHOB", "")
        pef_curve = curve_map.get("PEF", "")
        phit_curve = curve_map.get("PHIT", "")


        if not rhob_curve or rhob_curve not in out.columns:
            raise ValueError("No density curve selected.")
        if not pef_curve or pef_curve not in out.columns:
            raise ValueError("No pef curve selected.")
        if not phit_curve or phit_curve not in out.columns:
            raise ValueError("No phit curve selected.")

        rhob = pd.to_numeric(out[rhob_curve], errors="coerce")
        pef = pd.to_numeric(out[pef_curve], errors="coerce")
        phit = pd.to_numeric(out[phit_curve], errors="coerce")


        out["UMAA"] = rhob * pef
        out["UMAA"] = out["UMAA"].clip(lower=0.0, upper=30)

        out["RHOMAA"] =  (rhob - phit * fluid_density) / (1 - phit)
        out["RHOMAA"] = out["RHOMAA"].clip(lower=1, upper=3.2)

        return PluginResult(
            out,
            ["UMAA", "RHOMAA"],
            ["umaa and rhomaa computed."]
        )


# ============================================================
# UI widget
# ============================================================

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

        #self.method_combo = QComboBox()
        #self.method_combo.addItems([
        #    "density_neutron",
        #    "density_only",
        #    "neutron_only",
        #])
        #form.addRow("Method:", self.method_combo)

        self.rhob_combo = QComboBox()
        self.pef_combo = QComboBox()
        self.phit_combo = QComboBox()

        df = self.controller.get_analysis_df()
        cols = list(df.columns) if df is not None else []

        self.rhob_combo.addItems(cols)
        self.pef_combo.addItems(cols)
        self.phit_combo.addItems(cols)

        # auto-pick likely curves
        rhob_pick = _find_curve_case_insensitive(cols, RHOB_CANDS)
        pef_pick = _find_curve_case_insensitive(cols, PEF_CANDS)
        phit_pick = _find_curve_case_insensitive(cols, PHIT_CANDS)

        if rhob_pick is not None:
            idx = self.rhob_combo.findText(rhob_pick)
            if idx >= 0:
                self.rhob_combo.setCurrentIndex(idx)
        
        if pef_pick is not None:
            idx = self.pef_combo.findText(pef_pick)
            if idx >= 0:
                self.pef_combo.setCurrentIndex(idx)
                
        if phit_pick is not None:
            idx = self.phit_combo.findText(phit_pick)
            if idx >= 0:
               self.pef_combo.setCurrentIndex(idx)
  
                

        form.addRow("Density (RHOB):", self.rhob_combo)
        form.addRow("PEF (PEF):", self.pef_combo)
        form.addRow("PHIT (PHIT):", self.phit_combo)

        #self.matrix_density = QDoubleSpinBox()
        #self.matrix_density.setRange(1.0, 4.0)
        #self.matrix_density.setDecimals(3)
        #self.matrix_density.setSingleStep(0.01)
        #self.matrix_density.setValue(2.71)
        
        self.fluid_density = QDoubleSpinBox()
        self.fluid_density.setRange(0.5, 2.0)
        self.fluid_density.setDecimals(3)
        self.fluid_density.setSingleStep(0.01)
        self.fluid_density.setValue(1.10)
        
        #form.addRow("Matrix Density:", self.matrix_density)
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

        rhob_curve = self.rhob_combo.currentText().strip()
        pef_curve = self.pef_combo.currentText().strip()
        phit_curve = self.phit_combo.currentText().strip()

        curve_map = {
            "RHOB": rhob_curve,
            "PEF": pef_curve,
            "PHIT": phit_curve,
        }

        params = {
        #    "method": self.method_combo.currentText(),
        #    "matrix_density": self.matrix_density.value(),
            "fluid_density": self.fluid_density.value(),
        }

        try:
            result = self.engine.compute(df, curve_map, params)
            self.controller.update_analysis_df(result.df)


            print("New columns now in analysis_df:")
            print(list(self.controller.get_analysis_df().columns))

            QMessageBox.information(
                self,
                "Success",
                "\n".join(result.messages)
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))


# ============================================================
# Plugin wrapper
# ============================================================

class UmaaPlugin(BasePlugin):
    plugin_id = "umaa-rhomaa"
    display_name = "umaa-rhomaa"
    tooltip = "Compute Umaa-Rhomaa from density, pef and phit logs"

    def launch(self, controller):
        widget = UmaaWidget(controller)
        controller.set_center_widget(widget, self.display_name)