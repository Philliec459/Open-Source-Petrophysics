from __future__ import annotations

import pandas as pd

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QGroupBox,
    QTextEdit,
)

#from petrocore.workflow.phit_chartbook import compute_phit_chartbook
#from petrocore.workflow.phit_rms import compute_phit_rms


def _first_present(cols, candidates):
    s = set(cols)
    for c in candidates:
        if c in s:
            return c
    return None


def _find_candidates(cols, candidates):
    s = set(cols)
    out = [c for c in candidates if c in s]
    if out:
        return out

    found = []
    for c in cols:
        cu = c.upper()
        for pat in candidates:
            if pat.upper() in cu and c not in found:
                found.append(c)
    return found


class PhitPanel(QWidget):
    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self._build_ui()

    def _state(self):
        return self.controller.get_state()

    def _df(self):
        return getattr(self._state(), "analysis_df", None)

    def _params(self):
        state = self._state()
        params = getattr(state, "params", None)
        if not isinstance(params, dict):
            state.params = {}
            params = state.params
        return params

    def _build_ui(self):
        layout = QVBoxLayout(self)

        title = QLabel("Calculate PHIT")
        layout.addWidget(title)
        layout.addSpacing(4)


        box = QGroupBox("Density–Neutron Porosity")
        box_layout = QVBoxLayout(box)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Density curve:"))
        self.rhob_combo = QComboBox()
        row1.addWidget(self.rhob_combo)
        box_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Neutron curve:"))
        self.nphi_combo = QComboBox()
        row2.addWidget(self.nphi_combo)
        box_layout.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Method:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Density-Neutron Chartbook", "Density-Neutron RMS"])
        row3.addWidget(self.method_combo)
        box_layout.addLayout(row3)


 
        row4 = QHBoxLayout()
        row4.addWidget(QLabel("Chart:"))
        self.chart_combo = QComboBox()
        self.chart_combo.addItems(['TNPH ρf=1.19 (SLB)', 'CNL ρf=1.0 (SLB)', 'CNL ρf=1.1 (SLB)', 'TNPH ρf=1.00 (SLB)'])
        row4.addWidget(self.chart_combo)
        box_layout.addLayout(row4)



        self.calc_btn = QPushButton("Calculate PHIT")
        self.calc_btn.clicked.connect(self._on_calc_clicked)
        box_layout.addWidget(self.calc_btn)

        layout.addWidget(box)



        self.info_box = QTextEdit()
        self.info_box.setReadOnly(True)
        self.info_box.setMinimumHeight(150)
        layout.addWidget(self.info_box)

        layout.addStretch()



    def refresh(self):
        df = self._df()
        params = self._params()

        self.rhob_combo.clear()
        self.nphi_combo.clear()
        self.rhob_combo.addItem("")
        self.nphi_combo.addItem("")


        if df is None or df.empty:
            self.info_box.setPlainText("No analysis_df loaded.")
            return

        cols = list(df.columns)

        rhob_candidates = _find_candidates(cols, ["RHOZ", "RHOB", "DEN", "RHO8"])
        # tighter neutron candidates so we do not grab odd DNPH-style curves first
        tnph_candidates = _find_candidates(cols, ["TNPH", "NPHI", "NPOR", "CNL"])

        for c in rhob_candidates:
            self.rhob_combo.addItem(c)
        for c in tnph_candidates:
            self.nphi_combo.addItem(c)

        rhob_default = params.get("rhob_curve") or _first_present(cols, ["RHOZ", "RHOB", "DEN", "RHO8","ZDEN"])
        tnph_default = params.get("tnph_curve") or _first_present(cols, ["TNPH", "NPHI", "NPOR", "CNL","CNC","CNCF"])

        if rhob_default:
            idx = self.rhob_combo.findText(rhob_default)
            if idx >= 0:
                self.rhob_combo.setCurrentIndex(idx)

        if tnph_default:
            idx = self.nphi_combo.findText(tnph_default)
            if idx >= 0:
                self.nphi_combo.setCurrentIndex(idx)

        lines = [
            "PHIT inputs",
            "",
            f"rhob default = {rhob_default}",
            f"tnph default = {tnph_default}",
            "",
            f"rhob candidates = {rhob_candidates}",
            f"tnph candidates = {tnph_candidates}",
            f"tnph default = {tnph_default}",
            "", 
        ]
        self.info_box.setPlainText("\n".join(lines))



    def _on_calc_clicked(self):
        rhob = self.rhob_combo.currentText().strip()
        nphi = self.nphi_combo.currentText().strip()
        method = self.method_combo.currentText().strip()
        chart = self.chart_combo.currentText().strip()

        if not rhob or not nphi:
            print("[PHIT] Missing curve selection")
            return

        params = self._params()
        params["rhob_curve"] = rhob
        params["tnph_curve"] = nphi
        params["chart"] = chart

        print("Method of Xplot Porosity:", method, ", Chart Used:", chart)

        if method == "Density-Neutron Chartbook":

            self.controller.calculate_phit(rhob, nphi, method, chart)

        else:

            self.controller.calculate_phit_rms(rhob, nphi, method)




        lines2 = [
            "Method Used for PHIT",
            "",
            f"Method used to calculate PHIT = {method} using this chart: {chart}",
 
        ]
        self.info_box.setPlainText("\n".join(lines2))