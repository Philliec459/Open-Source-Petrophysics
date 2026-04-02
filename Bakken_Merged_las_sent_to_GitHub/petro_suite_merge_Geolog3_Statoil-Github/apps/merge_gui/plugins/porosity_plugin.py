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

RHOB_CANDS = ["RHOB", "RHOZ", "DEN", "DENS", "RHO8"]
NPHI_CANDS = ["TNPH", "NPHI", "CNL", "NPOR", "NEUT"]


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

class PorosityEngine:
    def compute(self, df: pd.DataFrame, curve_map: dict[str, str], params: dict[str, Any]) -> PluginResult:
        out = df.copy()

        method = params.get("method", "density_neutron")
        matrix_density = float(params.get("matrix_density", 2.71))
        fluid_density = float(params.get("fluid_density", 1.10))

        rhob_curve = curve_map.get("RHOB", "")
        nphi_curve = curve_map.get("NPHI", "")

        if method == "density_only":
            if not rhob_curve or rhob_curve not in out.columns:
                raise ValueError("No density curve selected.")

            rhob = pd.to_numeric(out[rhob_curve], errors="coerce")
            out["POR_DEN"] = (matrix_density - rhob) / (matrix_density - fluid_density)
            out["POR_DEN"] = out["POR_DEN"].clip(lower=0.0, upper=0.60)

            return PluginResult(
                out,
                ["POR_DEN"],
                ["Density porosity computed."]
            )

        if method == "neutron_only":
            if not nphi_curve or nphi_curve not in out.columns:
                raise ValueError("No neutron curve selected.")

            out["PHIN"] = pd.to_numeric(out[nphi_curve], errors="coerce").clip(lower=0.0, upper=0.60)

            return PluginResult(
                out,
                ["PHIN"],
                ["Neutron porosity computed."]
            )

        if not rhob_curve or rhob_curve not in out.columns:
            raise ValueError("No density curve selected.")
        if not nphi_curve or nphi_curve not in out.columns:
            raise ValueError("No neutron curve selected.")

        rhob = pd.to_numeric(out[rhob_curve], errors="coerce")
        nphi = pd.to_numeric(out[nphi_curve], errors="coerce")

        out["POR_DEN"] = (matrix_density - rhob) / (matrix_density - fluid_density)
        out["POR_DEN"] = out["POR_DEN"].clip(lower=0.0, upper=0.60)

        out["PHIT_DN"] = np.sqrt((out["POR_DEN"] ** 2 + nphi ** 2) / 2.0)
        out["PHIT_DN"] = out["PHIT_DN"].clip(lower=0.0, upper=0.60)

        return PluginResult(
            out,
            ["POR_DEN", "PHIT_DN"],
            ["Density-neutron porosity computed."]
        )


# ============================================================
# UI widget
# ============================================================

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
        self.method_combo.addItems([
            "density_neutron",
            "density_only",
            "neutron_only",
        ])
        form.addRow("Method:", self.method_combo)

        self.rhob_combo = QComboBox()
        self.nphi_combo = QComboBox()

        df = self.controller.get_analysis_df()
        cols = list(df.columns) if df is not None else []

        self.rhob_combo.addItems(cols)
        self.nphi_combo.addItems(cols)

        # auto-pick likely curves
        rhob_pick = _find_curve_case_insensitive(cols, RHOB_CANDS)
        nphi_pick = _find_curve_case_insensitive(cols, NPHI_CANDS)

        if rhob_pick is not None:
            idx = self.rhob_combo.findText(rhob_pick)
            if idx >= 0:
                self.rhob_combo.setCurrentIndex(idx)

        if nphi_pick is not None:
            idx = self.nphi_combo.findText(nphi_pick)
            if idx >= 0:
                self.nphi_combo.setCurrentIndex(idx)

        form.addRow("Density (RHOB):", self.rhob_combo)
        form.addRow("Neutron (NPHI):", self.nphi_combo)

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

        rhob_curve = self.rhob_combo.currentText().strip()
        nphi_curve = self.nphi_combo.currentText().strip()

        curve_map = {
            "RHOB": rhob_curve,
            "NPHI": nphi_curve,
        }

        params = {
            "method": self.method_combo.currentText(),
            "matrix_density": self.matrix_density.value(),
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

class PorosityPlugin(BasePlugin):
    plugin_id = "porosity"
    display_name = "Porosity"
    tooltip = "Compute porosity from density and neutron logs"

    def launch(self, controller):
        widget = PorosityWidget(controller)
        controller.set_center_widget(widget, self.display_name)