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

RT_CANDS = [
    "RT", "RESD", "ILD", "LLD", "AT90", "AT60", "AT30", "AT20",
    "RILD", "RDEP", "RESDEEP"
]

PHI_CANDS = [
    "PHIT", "PHIE", "PHIT_NMR", "PHIE_NMR", "TCMR", "DPHZ", "NPOR"
]


def _find_curve_case_insensitive(cols, candidates):
    if cols is None:
        return None

    cols_list = list(cols)
    cols_upper = {str(c).upper(): c for c in cols_list}

    for cand in candidates:
        if cand in cols_list:
            return cand

    for cand in candidates:
        cu = cand.upper()
        if cu in cols_upper:
            return cols_upper[cu]

    for cand in candidates:
        cu = cand.upper()
        for col in cols_list:
            if cu in str(col).upper():
                return col

    return None


# ============================================================
# Core compute engine
# ============================================================

class ArchieEngine:
    def compute(self, df: pd.DataFrame, curve_map: dict[str, str], params: dict[str, Any]) -> PluginResult:
        out = df.copy()

        rt_curve = curve_map.get("RT", "")
        phi_curve = curve_map.get("PHI", "")

        if not rt_curve or rt_curve not in out.columns:
            raise ValueError("No resistivity curve selected.")
        if not phi_curve or phi_curve not in out.columns:
            raise ValueError("No porosity curve selected.")

        a = float(params.get("a", 1.0))
        m = float(params.get("m", 2.0))
        n = float(params.get("n", 2.0))
        rw = float(params.get("rw", 0.023))
        phi_mode = params.get("phi_mode", "total")

        rt = pd.to_numeric(out[rt_curve], errors="coerce").to_numpy(dtype=float)
        phi = pd.to_numeric(out[phi_curve], errors="coerce").to_numpy(dtype=float)

        sw = np.full(len(out), np.nan, dtype=float)

        valid = (
            np.isfinite(rt) & (rt > 0.0) &
            np.isfinite(phi) & (phi > 0.0)
        )

        if np.any(valid):
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                sw[valid] = ((a * rw) / (rt[valid] * (phi[valid] ** m))) ** (1.0 / n)

        sw = np.clip(sw, 0.0, 1.0)

        bvw = np.full(len(out), np.nan, dtype=float)
        good_bvw = np.isfinite(sw) & np.isfinite(phi)
        bvw[good_bvw] = phi[good_bvw] * sw[good_bvw]

        out["SW_ARCHIE"] = sw
        out["BVW_ARCHIE"] = bvw

        msg = (
            f"Archie saturation computed using RT='{rt_curve}', "
            f"PHI='{phi_curve}', a={a:g}, m={m:g}, n={n:g}, Rw={rw:g}, "
            f"phi_mode={phi_mode}."
        )

        return PluginResult(
            out,
            ["SW_ARCHIE", "BVW_ARCHIE"],
            [msg]
        )


# ============================================================
# UI widget
# ============================================================

class ArchieWidget(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.engine = ArchieEngine()
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        title = QLabel("Archie Water Saturation")
        title.setStyleSheet("font-size: 12pt; font-weight: bold;")
        layout.addWidget(title)

        form = QFormLayout()

        self.rt_combo = QComboBox()
        self.phi_combo = QComboBox()
        self.phi_mode_combo = QComboBox()
        self.phi_mode_combo.addItems(["total", "effective"])

        df = self.controller.get_analysis_df()
        cols = list(df.columns) if df is not None else []

        self.rt_combo.addItems(cols)
        self.phi_combo.addItems(cols)

        rt_pick = _find_curve_case_insensitive(cols, RT_CANDS)
        phi_pick = _find_curve_case_insensitive(cols, PHI_CANDS)

        if rt_pick is not None:
            idx = self.rt_combo.findText(rt_pick)
            if idx >= 0:
                self.rt_combo.setCurrentIndex(idx)

        if phi_pick is not None:
            idx = self.phi_combo.findText(phi_pick)
            if idx >= 0:
                self.phi_combo.setCurrentIndex(idx)

        self.a_spin = QDoubleSpinBox()
        self.a_spin.setRange(0.01, 10.0)
        self.a_spin.setDecimals(4)
        self.a_spin.setSingleStep(0.01)
        self.a_spin.setValue(1.0)

        self.m_spin = QDoubleSpinBox()
        self.m_spin.setRange(0.5, 5.0)
        self.m_spin.setDecimals(4)
        self.m_spin.setSingleStep(0.01)
        self.m_spin.setValue(2.0)

        self.n_spin = QDoubleSpinBox()
        self.n_spin.setRange(0.5, 5.0)
        self.n_spin.setDecimals(4)
        self.n_spin.setSingleStep(0.01)
        self.n_spin.setValue(2.0)

        self.rw_spin = QDoubleSpinBox()
        self.rw_spin.setRange(0.0001, 5.0)
        self.rw_spin.setDecimals(5)
        self.rw_spin.setSingleStep(0.001)
        self.rw_spin.setValue(0.023)

        form.addRow("Resistivity (RT):", self.rt_combo)
        form.addRow("Porosity:", self.phi_combo)
        form.addRow("Porosity Type:", self.phi_mode_combo)
        form.addRow("a:", self.a_spin)
        form.addRow("m:", self.m_spin)
        form.addRow("n:", self.n_spin)
        form.addRow("Rw:", self.rw_spin)

        layout.addLayout(form)

        run_btn = QPushButton("Compute Archie")
        run_btn.clicked.connect(self._run)
        layout.addWidget(run_btn)

        layout.addStretch()

    def _run(self):
        df = self.controller.get_analysis_df()
        if df is None or df.empty:
            QMessageBox.warning(self, "Error", "No data loaded.")
            return

        rt_curve = self.rt_combo.currentText().strip()
        phi_curve = self.phi_combo.currentText().strip()

        if rt_curve not in df.columns:
            QMessageBox.warning(self, "Error", f"Invalid RT curve: {rt_curve}")
            return

        if phi_curve not in df.columns:
            QMessageBox.warning(self, "Error", f"Invalid porosity curve: {phi_curve}")
            return

        curve_map = {
            "RT": rt_curve,
            "PHI": phi_curve,
        }

        params = {
            "a": self.a_spin.value(),
            "m": self.m_spin.value(),
            "n": self.n_spin.value(),
            "rw": self.rw_spin.value(),
            "phi_mode": self.phi_mode_combo.currentText(),
        }

        try:
            result = self.engine.compute(df, curve_map, params)
            self.controller.update_analysis_df(result.df)

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

class ArchiePlugin(BasePlugin):
    plugin_id = "archie"
    display_name = "Archie"
    tooltip = "Compute water saturation using Archie"

    def launch(self, controller):
        widget = ArchieWidget(controller)
        controller.set_center_widget(widget, self.display_name)