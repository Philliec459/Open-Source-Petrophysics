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
    QGridLayout,
)


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

    # fallback partial matching
    found = []
    upper_cols = list(cols)
    for c in upper_cols:
        cu = c.upper()
        for pat in candidates:
            pu = pat.upper()
            if pu in cu and c not in found:
                found.append(c)
    return found


class CurvePickerPanel(QWidget):
    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        layout = QVBoxLayout(self)

        title = QLabel("Curve Selection")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        box = QGroupBox("Select Working Curves")
        grid = QGridLayout(box)

        self.combo_map = {}

        row = 0
        for label_text, key in [
            ("GR curve", "gr_curve"),
            ("CGR curve", "cgr_curve"),
            ("Density curve", "rhob_curve"),
            ("Neutron curve", "tnph_curve"),
            ("Sonic curve", "dtco_curve"),
            ("Resistivity curve", "rt_curve"),
            ("PHIT curve", "phit_curve"),
            ("NMR PHIT curve", "tcmr_curve"),
            ("NMR PHIE curve", "cmrp_curve"),
            ("BVIE curve", "bvie_curve"),
        ]:
            lbl = QLabel(label_text + ":")
            combo = QComboBox()
            combo.setMinimumWidth(240)
            self.combo_map[key] = combo
            grid.addWidget(lbl, row, 0)
            grid.addWidget(combo, row, 1)
            row += 1

        btn_row = QHBoxLayout()

        self.auto_btn = QPushButton("Auto Pick Curves")
        self.auto_btn.clicked.connect(self._auto_pick_curves)
        btn_row.addWidget(self.auto_btn)

        self.apply_btn = QPushButton("Apply Selected Curves")
        self.apply_btn.clicked.connect(self._apply_curves)
        btn_row.addWidget(self.apply_btn)

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh)
        btn_row.addWidget(self.refresh_btn)

        layout.addWidget(box)
        layout.addLayout(btn_row)

        self.info_box = QTextEdit()
        self.info_box.setReadOnly(True)
        self.info_box.setMinimumHeight(220)
        layout.addWidget(self.info_box)

        layout.addStretch()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _state(self):
        return self.controller.get_state()

    def _df(self):
        state = self._state()
        return getattr(state, "analysis_df", None)

    def _params(self):
        state = self._state()
        params = getattr(state, "params", None)
        if not isinstance(params, dict):
            state.params = {}
            params = state.params
        return params

    def _set_combo_items(self, combo: QComboBox, candidates: list[str], current_value: str | None):
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("")

        for c in candidates:
            combo.addItem(c)

        if current_value:
            idx = combo.findText(current_value)
            if idx >= 0:
                combo.setCurrentIndex(idx)

        combo.blockSignals(False)

    def _candidate_map(self, cols):
        return {
            "gr_curve": _find_candidates(cols, ["GR", "SGR", "GR_EDTC", "HSGR", "HCGR", "GRC"]),
            "cgr_curve": _find_candidates(cols, ["CGR", "HCGR", "ECGR"]),
            "rhob_curve": _find_candidates(cols, ["RHOB", "RHOZ", "DEN", "RHO8"]),
            "tnph_curve": _find_candidates(cols, ["TNPH", "NPHI", "NPOR", "CNL"]),
            "dtco_curve": _find_candidates(cols, ["DTCO", "DTC", "AC", "DT"]),
            "rt_curve": _find_candidates(cols, ["AT90", "AF90", "AO90", "ILD", "LLD", "RT", "RESD", "RDEP"]),
            "phit_curve": _find_candidates(cols, ["PHIT", "PHIT_CHART", "PHIT_NMR", "TCMR", "MPHI", "MPHS"]),
            "tcmr_curve": _find_candidates(cols, ["PHIT_NMR", "TCMR", "MPHS", "MPHI"]),
            "cmrp_curve": _find_candidates(cols, ["PHIE_NMR", "CMRP_3MS", "CMRP3MS", "CMRP", "MPHI"]),
            "bvie_curve": _find_candidates(cols, ["BVIE", "BVI_E", "BVI", "MBVI"]),
        }

    def _best_default_map(self, cols):
        return {
            "gr_curve": _first_present(cols, ["GR", "SGR", "GR_EDTC", "HSGR", "GRC", "HCGR"]),
            "cgr_curve": _first_present(cols, ["CGR", "HCGR", "ECGR"]),
            "rhob_curve": _first_present(cols, ["RHOZ", "RHOB", "DEN", "RHO8"]),
            "tnph_curve": _first_present(cols, ["TNPH", "NPHI", "NPOR", "CNL"]),
            "dtco_curve": _first_present(cols, ["DTCO", "DTC", "AC", "DT"]),
            "rt_curve": _first_present(cols, ["AT90", "AF90", "AO90", "ILD", "LLD", "RT", "RESD", "RDEP"]),
            "phit_curve": _first_present(cols, ["PHIT", "PHIT_CHART", "PHIT_NMR", "TCMR", "MPHI", "MPHS"]),
            "tcmr_curve": _first_present(cols, ["PHIT_NMR", "TCMR", "MPHS", "MPHI"]),
            "cmrp_curve": _first_present(cols, ["PHIE_NMR", "CMRP_3MS", "CMRP3MS", "CMRP", "MPHI"]),
            "bvie_curve": _first_present(cols, ["BVIE", "BVI_E", "BVI", "MBVI"]),
        }

    def _write_info(self, cols, candidates, defaults, params):
        lines = []
        lines.append("Detected candidate curves\n")
        for key in self.combo_map.keys():
            lines.append(f"{key}:")
            lines.append(f"  candidates = {candidates.get(key, [])}")
            lines.append(f"  default    = {defaults.get(key)}")
            lines.append(f"  selected   = {params.get(key)}")
            lines.append("")

        lines.append(f"Total columns in analysis_df: {len(cols)}")
        self.info_box.setPlainText("\n".join(lines))

    # ------------------------------------------------------------------
    # Public refresh
    # ------------------------------------------------------------------
    def refresh(self):
        df = self._df()
        params = self._params()

        for combo in self.combo_map.values():
            combo.clear()
            combo.addItem("")

        if df is None or df.empty:
            self.info_box.setPlainText("No analysis_df loaded.")
            return

        cols = list(df.columns)
        candidates = self._candidate_map(cols)
        defaults = self._best_default_map(cols)

        # If params are missing, seed them from defaults
        for key, default_val in defaults.items():
            if not params.get(key) and default_val:
                params[key] = default_val

        # Special rule: if PHIT exists after manual PHIT calc, prefer it
        if "PHIT" in cols:
            params["phit_curve"] = "PHIT"
        elif not params.get("phit_curve") and defaults.get("phit_curve"):
            params["phit_curve"] = defaults["phit_curve"]

        # Populate combos using candidates and current param selections
        for key, combo in self.combo_map.items():
            current_value = params.get(key)
            self._set_combo_items(combo, candidates.get(key, []), current_value)

        self._write_info(cols, candidates, defaults, params)

    # ------------------------------------------------------------------
    # Auto pick
    # ------------------------------------------------------------------
    def _auto_pick_curves(self):
        df = self._df()
        params = self._params()

        if df is None or df.empty:
            return

        cols = list(df.columns)
        defaults = self._best_default_map(cols)

        for key, val in defaults.items():
            if val:
                params[key] = val

        # Prefer PHIT if it already exists from the manual PHIT task
        if "PHIT" in cols:
            params["phit_curve"] = "PHIT"
        elif defaults.get("phit_curve"):
            params["phit_curve"] = defaults["phit_curve"]

        self.refresh()
        self.controller.refresh_ui()
        self.controller.refresh_plots()

    # ------------------------------------------------------------------
    # Apply
    # ------------------------------------------------------------------
    def _apply_curves(self):
        params = self._params()

        for key, combo in self.combo_map.items():
            val = combo.currentText().strip()
            if val:
                params[key] = val
            elif key in params:
                params[key] = None

        # If PHIT now exists in dataframe, use it as the active PHIT curve
        df = self._df()
        if df is not None and not df.empty and "PHIT" in df.columns:
            params["phit_curve"] = "PHIT"

        print("[CURVE PICKER] Updated working curves:")
        for key in self.combo_map.keys():
            print(f"  {key} = {params.get(key)}")

        self.refresh()
        self.controller.refresh_ui()
        self.controller.refresh_plots()