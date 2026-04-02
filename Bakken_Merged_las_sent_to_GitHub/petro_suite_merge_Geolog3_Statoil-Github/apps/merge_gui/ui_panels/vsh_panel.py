# apps/merge_gui/ui_panels/vsh_panel.py
from __future__ import annotations

import numpy as np
import pandas as pd

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QDoubleSpinBox,
    QMessageBox,
    QGroupBox,
    QTextEdit,
    QCheckBox,
)


VSH_PARAM_SPEC = {
    "vsh.gr_clean": dict(default=10.0, mn=0.0, mx=200.0, step=1.0, dec=1, label="GR clean"),
    "vsh.gr_shale": dict(default=205, mn=0.0, mx=300.0, step=1.0, dec=1, label="GR shale"),
    "vsh.hc_corr": dict(default=0.00, mn=-1.0, mx=1.0, step=0.01, dec=2, label="HC correction"),
    "vsh.nd_weight": dict(default=0.50, mn=0.0, mx=1.0, step=0.05, dec=2, label="N-D weight"),
    "vsh.dt_clean": dict(default=55.0, mn=20.0, mx=120.0, step=1.0, dec=1, label="DT clean"),
    "vsh.dt_shale": dict(default=90.0, mn=40.0, mx=220.0, step=1.0, dec=1, label="DT shale"),
    "vsh.rhob_clean": dict(default=2.65, mn=1.80, mx=3.20, step=0.01, dec=3, label="RHOB clean"),
    "vsh.rhob_shale": dict(default=2.35, mn=1.80, mx=3.20, step=0.01, dec=3, label="RHOB shale"),
    "vsh.nphi_clean": dict(default=0.03, mn=-0.20, mx=1.00, step=0.01, dec=3, label="NPHI clean"),
    "vsh.nphi_shale": dict(default=0.35, mn=-0.20, mx=1.00, step=0.01, dec=3, label="NPHI shale"),
}


class VshPanel(QWidget):
    """
    Vsh calculation panel.

    Supports:
      - Gamma Ray Vsh
      - Neutron-Density Vsh
      - HL-style combined workflow

    Inputs expected in state.analysis_df:
      - GR curve
      - optional DT curve
      - optional RHOB curve
      - optional NPHI/TNPH curve

    Outputs written to state.analysis_df:
      - VSH_GR
      - VSH_ND
      - VSH_HL
      - VSH
      - GR_USED
      - DT_USED
      - RHOB_USED
      - NPHI_USED
    """

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.param_widgets = {}
        self._build_ui()

    # ------------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------------
    def _state(self):
        return getattr(self.controller, "state", None)

    def _analysis_df(self):
        state = self._state()
        if state is None:
            return None
        return getattr(state, "analysis_df", None)

    def _params(self):
        state = self._state()
        if state is None:
            return {}
        if not hasattr(state, "params") or state.params is None:
            state.params = {}
        return state.params

    def _curve_picks(self):
        state = self._state()
        if state is None:
            return {}
        if not hasattr(state, "curve_picks") or state.curve_picks is None:
            state.curve_picks = {}
        if "vsh" not in state.curve_picks:
            state.curve_picks["vsh"] = {}
        return state.curve_picks["vsh"]

    def _curve_locks(self):
        state = self._state()
        if state is None:
            return {}
        if not hasattr(state, "curve_lock") or state.curve_lock is None:
            state.curve_lock = {}
        return state.curve_lock

    def _is_locked(self, key: str) -> bool:
        return bool(self._curve_locks().get(key, False))

    def _set_locked(self, key: str, value: bool = True):
        self._curve_locks()[key] = value

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        layout = QVBoxLayout(self)

        title = QLabel("Compute Shale Volumes")
        
        layout.addWidget(title)

        subtitle = QLabel(
            "Choose GR and optional sonic / neutron / density curves, then calculate Vsh."
        )
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        # --------------------------------------------------------------
        # Method selection
        # --------------------------------------------------------------
        method_box = QGroupBox("Method")
        method_layout = QVBoxLayout(method_box)

        self.use_gr_cb = QCheckBox("Use Gamma Ray Vsh")
        self.use_gr_cb.setChecked(True)

        self.use_nd_cb = QCheckBox("Use Neutron-Density Vsh")
        self.use_nd_cb.setChecked(True)

        self.use_hl_cb = QCheckBox("Use HL-style combined Vsh")
        self.use_hl_cb.setChecked(True)

        method_layout.addWidget(self.use_gr_cb)
        method_layout.addWidget(self.use_nd_cb)
        method_layout.addWidget(self.use_hl_cb)

        layout.addWidget(method_box)

        # --------------------------------------------------------------
        # Curve selection
        # --------------------------------------------------------------
        curve_box = QGroupBox("Curve Selection")
        curve_form = QFormLayout(curve_box)

        self.gr_combo = QComboBox()
        self.dt_combo = QComboBox()
        self.rhob_combo = QComboBox()
        self.nphi_combo = QComboBox()

        curve_form.addRow("GR curve:", self.gr_combo)
        curve_form.addRow("Sonic curve:", self.dt_combo)
        curve_form.addRow("Density curve:", self.rhob_combo)
        curve_form.addRow("Neutron curve:", self.nphi_combo)

        layout.addWidget(curve_box)

        # --------------------------------------------------------------
        # Parameters
        # --------------------------------------------------------------
        param_box = QGroupBox("Parameters")
        param_form = QFormLayout(param_box)

        for key, spec in VSH_PARAM_SPEC.items():
            w = QDoubleSpinBox()
            w.setDecimals(spec["dec"])
            w.setSingleStep(spec["step"])
            w.setRange(spec["mn"], spec["mx"])
            w.setValue(spec["default"])
            w.valueChanged.connect(self._store_params_from_ui)
            self.param_widgets[key] = w
            param_form.addRow(spec["label"] + ":", w)

        layout.addWidget(param_box)

        # --------------------------------------------------------------
        # Buttons
        # --------------------------------------------------------------
        btn_row = QHBoxLayout()

        self.refresh_btn = QPushButton("Refresh Curves")
        self.refresh_btn.clicked.connect(self.refresh)

        self.unlock_btn = QPushButton("Unlock Auto Picks")
        self.unlock_btn.clicked.connect(self._unlock_panel_picks)

        self.calc_btn = QPushButton("Calculate Vsh")
        self.calc_btn.clicked.connect(self.calculate_vsh)

        btn_row.addWidget(self.refresh_btn)
        btn_row.addWidget(self.unlock_btn)
        btn_row.addWidget(self.calc_btn)
        btn_row.addStretch(1)

        layout.addLayout(btn_row)

        # --------------------------------------------------------------
        # Status / notes
        # --------------------------------------------------------------
        self.status_label = QLabel("Ready.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        #self.notes = QTextEdit()
        #self.notes.setReadOnly(True)
        #self.notes.setMaximumHeight(160)
        #layout.addWidget(self.notes)


        self.info_box = QTextEdit()
        self.info_box.setReadOnly(True)
        self.info_box.setMaximumHeight(180)
        layout.addWidget(self.info_box)



        layout.addStretch(1)

        # user changes lock picks
        self.gr_combo.currentTextChanged.connect(
            lambda _: self._on_curve_changed("vsh.gr_curve")
        )
        self.dt_combo.currentTextChanged.connect(
            lambda _: self._on_curve_changed("vsh.dt_curve")
        )
        self.rhob_combo.currentTextChanged.connect(
            lambda _: self._on_curve_changed("vsh.rhob_curve")
        )
        self.nphi_combo.currentTextChanged.connect(
            lambda _: self._on_curve_changed("vsh.nphi_curve")
        )

        self._load_params_to_ui()
        self.refresh()

    # ------------------------------------------------------------------
    # UI/state sync
    # ------------------------------------------------------------------
    def _load_params_to_ui(self):
        params = self._params()
        for key, spec in VSH_PARAM_SPEC.items():
            val = params.get(key, spec["default"])
            self.param_widgets[key].blockSignals(True)
            self.param_widgets[key].setValue(float(val))
            self.param_widgets[key].blockSignals(False)

    def _store_params_from_ui(self):
        params = self._params()
        for key, widget in self.param_widgets.items():
            params[key] = widget.value()

    def _store_curve_picks_from_ui(self):
        picks = self._curve_picks()
        picks["gr_curve"] = self.gr_combo.currentText().strip()
        picks["dt_curve"] = self.dt_combo.currentText().strip()
        picks["rhob_curve"] = self.rhob_combo.currentText().strip()
        picks["nphi_curve"] = self.nphi_combo.currentText().strip()

    def _on_curve_changed(self, lock_key: str):
        self._set_locked(lock_key, True)
        self._store_curve_picks_from_ui()
        self._update_notes()

    def _unlock_panel_picks(self):
        for key in [
            "vsh.gr_curve",
            "vsh.dt_curve",
            "vsh.rhob_curve",
            "vsh.nphi_curve",
        ]:
            self._set_locked(key, False)

        picks = self._curve_picks()
        picks["gr_curve"] = ""
        picks["dt_curve"] = ""
        picks["rhob_curve"] = ""
        picks["nphi_curve"] = ""

        self.refresh()
        self.status_label.setText("Vsh picks unlocked. Auto-pick restored.")

    # ------------------------------------------------------------------
    # Curve selection logic
    # ------------------------------------------------------------------
    def _all_numeric_columns(self, df):
        out = []
        for c in df.columns:
            try:
                s = pd.to_numeric(df[c], errors="coerce")
                if s.notna().sum() > 0:
                    out.append(c)
            except Exception:
                pass
        return out

    def _candidates_from_names(self, cols, names):
        names_upper = [n.upper() for n in names]
        found = []
        for c in cols:
            cu = c.upper()
            if any(n in cu for n in names_upper):
                found.append(c)
        return found

    def _best_curve(self, cols, preferred):
        for p in preferred:
            for c in cols:
                if c.upper() == p.upper():
                    return c
        for p in preferred:
            for c in cols:
                if p.upper() in c.upper():
                    return c
        return cols[0] if cols else ""

    def _fill_combo(self, combo, cols, preferred, pick_key, stored_value=""):
        current = combo.currentText().strip()

        combo.blockSignals(True)
        combo.clear()
        combo.addItem("")
        for c in cols:
            combo.addItem(c)

        target = ""

        if self._is_locked(pick_key):
            if stored_value and stored_value in cols:
                target = stored_value
            elif current and current in cols:
                target = current
        else:
            if stored_value and stored_value in cols:
                target = stored_value
            else:
                target = self._best_curve(cols, preferred)

        if target:
            combo.setCurrentText(target)

        combo.blockSignals(False)

    def refresh(self):
        df = self._analysis_df()
        if df is None or df.empty:
            self.status_label.setText("No analysis dataframe loaded.")
            return

        self._load_params_to_ui()
        picks = self._curve_picks()

        cols = self._all_numeric_columns(df)

        gr_cols = self._candidates_from_names(
            cols, ["GR", "SGR", "CGR", "HCGR", "ECGR", "GRC"]
        )
        dt_cols = self._candidates_from_names(
            cols, ["DTCO", "DTC", "DT", "AC"]
        )
        rhob_cols = self._candidates_from_names(
            cols, ["RHOB", "RHOZ", "DEN", "RHO8"]
        )
        nphi_cols = self._candidates_from_names(
            cols, ["TNPH", "NPHI", "NPOR"]
        )

        self._fill_combo(
            self.gr_combo,
            gr_cols,
            ["GR", "SGR", "CGR", "HCGR"],
            "vsh.gr_curve",
            picks.get("gr_curve", ""),
        )
        self._fill_combo(
            self.dt_combo,
            dt_cols,
            ["DTCO", "DTC", "DT", "AC"],
            "vsh.dt_curve",
            picks.get("dt_curve", ""),
        )
        self._fill_combo(
            self.rhob_combo,
            rhob_cols,
            ["RHOB", "RHOZ", "DEN"],
            "vsh.rhob_curve",
            picks.get("rhob_curve", ""),
        )
        self._fill_combo(
            self.nphi_combo,
            nphi_cols,
            ["TNPH", "NPHI", "NPOR"],
            "vsh.nphi_curve",
            picks.get("nphi_curve", ""),
        )

        self._store_curve_picks_from_ui()
        self.status_label.setText("Vsh panel refreshed.")
        self._update_notes()

    def _update_notes(self):
        locks = self._curve_locks()

        def lock_txt(key):
            return "LOCKED" if locks.get(key, False) else "auto"

        txt = [
            "Recommended / current picks:",
            f"  GR:    {self.gr_combo.currentText() or 'None'}   [{lock_txt('vsh.gr_curve')}]",
            f"  Sonic: {self.dt_combo.currentText() or 'None'}   [{lock_txt('vsh.dt_curve')}]",
            f"  RHOB:  {self.rhob_combo.currentText() or 'None'}   [{lock_txt('vsh.rhob_curve')}]",
            f"  NPHI:  {self.nphi_combo.currentText() or 'None'}   [{lock_txt('vsh.nphi_curve')}]",
            "",
            f"Methods:",
            f"  GR   = {'on' if self.use_gr_cb.isChecked() else 'off'}",
            f"  N-D  = {'on' if self.use_nd_cb.isChecked() else 'off'}",
            f"  HL   = {'on' if self.use_hl_cb.isChecked() else 'off'}",
        ]
        #self.notes.setPlainText("\n".join(txt))
        self.info_box.setPlainText("\n".join(txt))

    # ------------------------------------------------------------------
    # Calculation helpers
    # ------------------------------------------------------------------
    def _safe_clip01(self, s):
        return pd.to_numeric(s, errors="coerce").clip(lower=0.0, upper=1.0)

    def _compute_vsh_gr(self, gr, gr_clean, gr_shale):
        denom = max(gr_shale - gr_clean, 1e-6)
        vsh_gr = (gr - gr_clean) / denom
        return vsh_gr.clip(lower=0.0, upper=1.0)

    def _compute_vsh_nd(self, rhob, nphi, rhob_clean, rhob_shale, nphi_clean, nphi_shale, nd_weight):
        # normalized neutron index
        denom_n = max(nphi_shale - nphi_clean, 1e-6)
        vsh_n = (nphi - nphi_clean) / denom_n

        # normalized density index
        # clean is denser than shale, so reverse
        denom_d = max(rhob_clean - rhob_shale, 1e-6)
        vsh_d = (rhob_clean - rhob) / denom_d

        vsh_nd = nd_weight * vsh_n + (1.0 - nd_weight) * vsh_d
        return vsh_nd.clip(lower=0.0, upper=1.0)

    def _compute_vsh_dt(self, dt, dt_clean, dt_shale):
        denom = max(dt_shale - dt_clean, 1e-6)
        vsh_dt = (dt - dt_clean) / denom
        return vsh_dt.clip(lower=0.0, upper=1.0)

    # ------------------------------------------------------------------
    # Calculation
    # ------------------------------------------------------------------


    
        
        
        
    def calculate_vsh(self):
        df = self._analysis_df()
        state = self._state()
    
        if df is None or df.empty or state is None:
            QMessageBox.warning(self, "Vsh", "No analysis dataframe loaded.")
            return
    
        self._store_params_from_ui()
        self._store_curve_picks_from_ui()
    
        picks = self._curve_picks()
        params = self._params()
    
        gr_curve = picks.get("gr_curve", "").strip()
        dt_curve = picks.get("dt_curve", "").strip()
        rhob_curve = picks.get("rhob_curve", "").strip()
        nphi_curve = picks.get("nphi_curve", "").strip()
    
        use_gr = self.use_gr_cb.isChecked()
        use_nd = self.use_nd_cb.isChecked()
        use_hl = self.use_hl_cb.isChecked()
    
        if not any([use_gr, use_nd, use_hl]):
            QMessageBox.warning(self, "Vsh", "Please select at least one Vsh method.")
            return
    
        if (use_gr or use_hl) and not gr_curve:
            QMessageBox.warning(self, "Vsh", "GR curve is required for GR or HL Vsh.")
            return
    
        if (use_nd or use_hl) and (not rhob_curve or not nphi_curve):
            QMessageBox.warning(self, "Vsh", "RHOB and NPHI curves are required for N-D or HL Vsh.")
            return
    
        # --------------------------------------------------------------
        # Build ZoI subset
        # --------------------------------------------------------------
        out = df.copy()
    
        if hasattr(self.controller, "_get_view_mask"):
            mask = self.controller._get_view_mask(out)
        else:
            mask = pd.Series(True, index=out.index)
    
        if mask is None or not mask.any():
            QMessageBox.warning(self, "Vsh", "No rows available in the selected ZoI.")
            return
    
        subset = out.loc[mask].copy()
    
        try:
            gr = pd.to_numeric(subset[gr_curve], errors="coerce") if gr_curve else None
            dt = pd.to_numeric(subset[dt_curve], errors="coerce") if dt_curve else None
            rhob = pd.to_numeric(subset[rhob_curve], errors="coerce") if rhob_curve else None
            nphi = pd.to_numeric(subset[nphi_curve], errors="coerce") if nphi_curve else None
        except Exception as e:
            QMessageBox.critical(self, "Vsh", f"Failed reading selected curves:\n{e}")
            return
    
        # parameters
        gr_clean = float(params.get("vsh.gr_clean", 10.0))
        gr_shale = float(params.get("vsh.gr_shale", 205.0))
        hc_corr = float(params.get("vsh.hc_corr", 0.00))
        nd_weight = float(params.get("vsh.nd_weight", 0.50))
        dt_clean = float(params.get("vsh.dt_clean", 55.0))
        dt_shale = float(params.get("vsh.dt_shale", 90.0))
        rhob_clean = float(params.get("vsh.rhob_clean", 2.65))
        rhob_shale = float(params.get("vsh.rhob_shale", 2.35))
        nphi_clean = float(params.get("vsh.nphi_clean", 0.03))
        nphi_shale = float(params.get("vsh.nphi_shale", 0.35))
    
        vsh_gr = None
        vsh_nd = None
        vsh_dt = None
        vsh_hl = None
    
        # --------------------------------------------------------------
        # GR Vsh
        # --------------------------------------------------------------
        if use_gr or use_hl:
            vsh_gr = self._compute_vsh_gr(gr, gr_clean, gr_shale)
            if hc_corr != 0.0:
                vsh_gr = (vsh_gr + hc_corr).clip(lower=0.0, upper=1.0)
    
            out.loc[mask, "VSH_GR"] = vsh_gr.values
            out.loc[mask, "GR_USED"] = gr.values
    
        # --------------------------------------------------------------
        # N-D Vsh
        # --------------------------------------------------------------
        if use_nd or use_hl:
            vsh_nd = self._compute_vsh_nd(
                rhob=rhob,
                nphi=nphi,
                rhob_clean=rhob_clean,
                rhob_shale=rhob_shale,
                nphi_clean=nphi_clean,
                nphi_shale=nphi_shale,
                nd_weight=nd_weight,
            )
    
            out.loc[mask, "VSH_ND"] = vsh_nd.values
            out.loc[mask, "RHOB_USED"] = rhob.values
            out.loc[mask, "NPHI_USED"] = nphi.values
    
        # --------------------------------------------------------------
        # DT Vsh (used only inside HL if available)
        # --------------------------------------------------------------
        if use_hl and dt is not None:
            vsh_dt = self._compute_vsh_dt(dt, dt_clean, dt_shale)
            out.loc[mask, "VSH_DT"] = vsh_dt.values
            out.loc[mask, "DT_USED"] = dt.values
    
        # --------------------------------------------------------------
        # HL-style combined
        # --------------------------------------------------------------
        if use_hl:
            parts = []
    
            if vsh_gr is not None:
                parts.append(vsh_gr)
    
            if vsh_nd is not None:
                parts.append(vsh_nd)
    
            if vsh_dt is not None:
                parts.append(vsh_dt)
    
            if not parts:
                QMessageBox.warning(
                    self,
                    "Vsh",
                    "HL selected, but not enough valid curves were available."
                )
                return
    
            vsh_hl = pd.concat(parts, axis=1).mean(axis=1).clip(lower=0.0, upper=1.0)
            out.loc[mask, "VSH_HL"] = vsh_hl.values
    
        # --------------------------------------------------------------
        # Final VSH
        # --------------------------------------------------------------
        if use_hl and vsh_hl is not None:
            out.loc[mask, "VSH"] = vsh_hl.values
            final_method = "HL"
        elif use_nd and vsh_nd is not None:
            out.loc[mask, "VSH"] = vsh_nd.values
            final_method = "N-D"
        elif use_gr and vsh_gr is not None:
            out.loc[mask, "VSH"] = vsh_gr.values
            final_method = "GR"
        else:
            QMessageBox.warning(self, "Vsh", "Could not determine final VSH output.")
            return
    
        state.analysis_df = out
    
        top, base = (None, None)
        if hasattr(self.controller, "_get_zoi_range"):
            top, base = self.controller._get_zoi_range()
    
        summary = []
        summary.append("Vsh calculation complete.")
        summary.append("")
        summary.append(f"Final VSH method: {final_method}")
        summary.append(f"GR curve:   {gr_curve or 'None'}")
        summary.append(f"DT curve:   {dt_curve or 'None'}")
        summary.append(f"RHOB curve: {rhob_curve or 'None'}")
        summary.append(f"NPHI curve: {nphi_curve or 'None'}")
        summary.append("")
    
        if top is not None and base is not None:
            summary.append(f"ZoI used: {top:.2f} to {base:.2f}")
        else:
            summary.append("ZoI used: full well")
    
        summary.append("")
        summary.append(f"GR clean = {gr_clean:.1f}")
        summary.append(f"GR shale = {gr_shale:.1f}")
        summary.append(f"HC corr = {hc_corr:.2f}")
        summary.append(f"N-D weight = {nd_weight:.2f}")
        summary.append(f"DT clean = {dt_clean:.1f}")
        summary.append(f"DT shale = {dt_shale:.1f}")
        summary.append(f"RHOB clean = {rhob_clean:.3f}")
        summary.append(f"RHOB shale = {rhob_shale:.3f}")
        summary.append(f"NPHI clean = {nphi_clean:.3f}")
        summary.append(f"NPHI shale = {nphi_shale:.3f}")
        summary.append("")
        summary.append(f"Valid VSH points in ZoI: {int(pd.to_numeric(out.loc[mask, 'VSH'], errors='coerce').notna().sum())}")
    
   

        self.status_label.setText("Vsh calculated.")
        #self.notes.clear()
        self.info_box.setPlainText("\n".join(summary))
        self._update_notes()

        if hasattr(self.controller, "refresh_ui"):
            self.controller.refresh_ui()
        elif hasattr(self.controller, "refresh_plots"):
            self.controller.refresh_plots()

                  
                    
    
 
    
    
    
    
 
    
    

