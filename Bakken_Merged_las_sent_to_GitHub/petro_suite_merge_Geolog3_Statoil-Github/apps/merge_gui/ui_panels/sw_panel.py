# apps/merge_gui/ui_panels/sw_panel.py
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
)


SW_PARAM_SPEC = {
    "sw.m_cem": dict(default=1.90, mn=1.00, mx=3.00, step=0.01, dec=2, label="m (cementation)"),
    "sw.n_sat": dict(default=2.00, mn=1.00, mx=3.00, step=0.01, dec=2, label="n (saturation)"),
    "sw.rw": dict(default=0.03, mn=0.001, mx=1.000, step=0.001, dec=4, label="Rw"),
    "sw.mslope": dict(default=1.00, mn=0.00, mx=5.00, step=0.01, dec=2, label="Vsh correction slope"),
    "sw.B": dict(default=12.10, mn=0.00, mx=50.00, step=0.10, dec=2, label="B (optional)"),
    "cbw_intercept": dict(default=0.1, mn=0.00, mx=1.00, step=0.01, dec=2, label="CBW intercept"),
}



def _first_present(cols, candidates):
    s = set(cols)
    for c in candidates:
        if c in s:
            return c
    return None




def compute_sal_kppm_from_rw75(Rw75):
    Rw75 = np.asarray(Rw75, dtype=float)
    x = np.clip(Rw75 - 0.0123, 1e-6, None)
    return (10 ** ((3.562 - np.log10(x)) / 0.955)) / 1000.0


def compute_bdacy(T_F, Rw):
    Rw = np.asarray(Rw, dtype=float)
    TC = (float(T_F) - 32.0) / 1.8
    TC_safe = max(TC, 1e-6)
    Rw_safe = np.clip(Rw, 1e-6, None)
    term1 = (1.0 - 0.83 * np.exp(-np.exp(-2.38 + (42.17 / TC_safe)) / Rw_safe))
    term2 = (-3.16 + 1.59 * np.log(TC_safe)) ** 2
    return term1 * term2








class SwPanel(QWidget):
    """
    Water saturation panel.

    Expected inputs in state.analysis_df:
      - porosity curve, usually PHIT
      - resistivity curve, usually RT
      - shale curve, usually VSH
      - optional QV / CBW / PHIT_NMR / BVW

    Outputs written to state.analysis_df:
      - SW
      - BVW
      - RT_USED
      - PHIT_USED
      - VSH_USED
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
        if "sw" not in state.curve_picks:
            state.curve_picks["sw"] = {}
        return state.curve_picks["sw"]

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

        title = QLabel("Calculate Waxman-Smits Water Saturations")
        layout.addWidget(title)

        subtitle = QLabel(
            "Choose porosity, resistivity, and shale curves, then calculate Sw and BVW."
        )
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        # --------------------------------------------------------------
        # Curve picks
        # --------------------------------------------------------------
        curve_box = QGroupBox("Curve Selection")
        curve_form = QFormLayout(curve_box)

        self.phit_combo = QComboBox()
        self.rt_combo = QComboBox()
        self.vsh_combo = QComboBox()
        #self.qv_combo = QComboBox()
        self.cbw_combo = QComboBox()

        curve_form.addRow("PHIT curve:", self.phit_combo)
        curve_form.addRow("Resistivity curve:", self.rt_combo)
        curve_form.addRow("Vsh curve:", self.vsh_combo)
        #curve_form.addRow("QV curve (optional):", self.qv_combo)
        curve_form.addRow("CBW curve (optional):", self.cbw_combo)

        layout.addWidget(curve_box)

        # --------------------------------------------------------------
        # Parameters
        # --------------------------------------------------------------
        param_box = QGroupBox("Parameters")
        param_form = QFormLayout(param_box)

        for key, spec in SW_PARAM_SPEC.items():
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

        self.unlock_btn = QPushButton("Load PHIT")
        self.unlock_btn.clicked.connect(self._unlock_panel_picks)

        self.calc_btn = QPushButton("Calculate Sw")
        self.calc_btn.clicked.connect(self.calculate_sw)

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

        self.notes = QTextEdit()
        self.notes.setReadOnly(True)
        self.notes.setMaximumHeight(140)
        layout.addWidget(self.notes)

        layout.addStretch(1)

        # user changes lock the pick
        self.phit_combo.currentTextChanged.connect(
            lambda _: self._on_curve_changed("sw.phit_curve")
        )
        self.rt_combo.currentTextChanged.connect(
            lambda _: self._on_curve_changed("sw.rt_curve")
        )
        self.vsh_combo.currentTextChanged.connect(
            lambda _: self._on_curve_changed("sw.vsh_curve")
        )
        #self.qv_combo.currentTextChanged.connect(
        # #   lambda _: self._on_curve_changed("sw.qv_curve")
        #)
        self.cbw_combo.currentTextChanged.connect(
            lambda _: self._on_curve_changed("sw.cbw_curve")
        )

        self._load_params_to_ui()
        self.refresh()

    # ------------------------------------------------------------------
    # UI/state sync
    # ------------------------------------------------------------------
    def _load_params_to_ui(self):
        params = self._params()
        for key, spec in SW_PARAM_SPEC.items():
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
        picks["phit_curve"] = self.phit_combo.currentText().strip()
        picks["rt_curve"] = self.rt_combo.currentText().strip()
        picks["vsh_curve"] = self.vsh_combo.currentText().strip()
        #picks["qv_curve"] = self.qv_combo.currentText().strip()
        picks["cbw_curve"] = self.cbw_combo.currentText().strip()

    def _on_curve_changed(self, lock_key: str):
        self._set_locked(lock_key, True)
        self._store_curve_picks_from_ui()
        self._update_notes()

    def _unlock_panel_picks(self):
        for key in [
            "sw.phit_curve",
            "sw.rt_curve",
            "sw.vsh_curve",
            #"sw.qv_curve",
            "sw.cbw_curve",
        ]:
            self._set_locked(key, False)

        picks = self._curve_picks()
        picks["phit_curve"] = ""
        picks["rt_curve"] = ""
        picks["vsh_curve"] = ""
        #picks["qv_curve"] = ""
        picks["cbw_curve"] = ""

        self.refresh()
        self.status_label.setText("Sw picks unlocked. Auto-pick restored.")

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

        phit_cols = self._candidates_from_names(
            cols, ["PHIT", "PHIE", "PHIT_NMR", "NPHI", "TNPH", "POR"]
        )
        rt_cols = self._candidates_from_names(
            cols, ["RT", "AF90","AT90","ILD", "LLD", "RESD", "RDEP", "RES"]
        )
        vsh_cols = self._candidates_from_names(
            cols, ["VSH_HL", "VSH_GR", "VSH", "VSHALE"]
        )
        #qv_cols = self._candidates_from_names(
        #    cols, ["QV", "QV_TOTAL", "QVNMR", "QV_NMR"]
        #)
        cbw_cols = self._candidates_from_names(
            cols, ["CBW"]
        )

        self._fill_combo(
            self.phit_combo,
            phit_cols,
            ["PHIT", "PHIE", "PHIT_NMR", "TNPH"],
            "sw.phit_curve",
            picks.get("phit_curve", ""),
        )
        self._fill_combo(
            self.rt_combo,
            rt_cols,
            ["RT","AF90","AT90", "ILD", "LLD", "RESD"],
            "sw.rt_curve",
            picks.get("rt_curve", ""),
        )
        self._fill_combo(
            self.vsh_combo,
            vsh_cols,
            ["VSH_HL", "VSH_GR", "VSH_ND","VSH"],
            "sw.vsh_curve",
            picks.get("vsh_curve", ""),
        )
        '''
        self._fill_combo(
            self.qv_combo,
            qv_cols,
            ["QV", "QV_TOTAL", "QVNMR"],
            "sw.qv_curve",
            picks.get("qv_curve", ""),'''
        #)
        self._fill_combo(
            self.cbw_combo,
            cbw_cols,
            ["CBW", "SWB"],
            "sw.cbw_curve",
            picks.get("cbw_curve", ""),
        )

        self._store_curve_picks_from_ui()
        self.status_label.setText("Sw panel refreshed.")
        self._update_notes()

    def _update_notes(self):
        locks = self._curve_locks()

        def lock_txt(key):
            return "LOCKED" if locks.get(key, False) else "auto"

        txt = [
            "Recommended / current picks:",
            f"  PHIT: {self.phit_combo.currentText() or 'None'}   [{lock_txt('sw.phit_curve')}]",
            f"  RT:   {self.rt_combo.currentText() or 'None'}   [{lock_txt('sw.rt_curve')}]",
            f"  VSH:  {self.vsh_combo.currentText() or 'None'}   [{lock_txt('sw.vsh_curve')}]",
            #f"  QV:   {self.qv_combo.currentText() or 'None'}   [{lock_txt('sw.qv_curve')}]",
            f"  CBW:  {self.cbw_combo.currentText() or 'None'}   [{lock_txt('sw.cbw_curve')}]",
        ]
        self.notes.setPlainText("\n".join(txt))

    # ------------------------------------------------------------------
    # Calculation
    # ------------------------------------------------------------------   
    def calculate_sw_archie(self):
        df = self._analysis_df()
        state = self._state()
    
        if df is None or df.empty or state is None:
            QMessageBox.warning(self, "Sw", "No analysis dataframe loaded.")
            return
    
        self._store_params_from_ui()
        self._store_curve_picks_from_ui()
    
        picks = self._curve_picks()
        params = self._params()
    
        phit_curve = picks.get("phit_curve", "").strip()
        rt_curve = picks.get("rt_curve", "").strip()
        vsh_curve = picks.get("vsh_curve", "").strip()
        cbw_curve = picks.get("cbw_curve", "").strip()
    
        missing = []
        if not phit_curve:
            missing.append("PHIT")
        if not rt_curve:
            missing.append("RT")
        if not vsh_curve:
            missing.append("VSH")
    
        if missing:
            QMessageBox.warning(
                self,
                "Sw",
                f"Missing required curve selections: {', '.join(missing)}"
            )
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
            QMessageBox.warning(self, "Sw", "No rows available in the selected ZoI.")
            return
    
        subset = out.loc[mask].copy()
    
        try:
            phit = pd.to_numeric(subset[phit_curve], errors="coerce")
            rt = pd.to_numeric(subset[rt_curve], errors="coerce")
            vsh = pd.to_numeric(subset[vsh_curve], errors="coerce")
        except Exception as e:
            QMessageBox.critical(self, "Sw", f"Failed reading selected curves:\n{e}")
            return
    
    
        cbw = None
        if cbw_curve and cbw_curve in subset.columns:
            cbw = pd.to_numeric(subset[cbw_curve], errors="coerce")
    
        # parameters
        m = float(params.get("sw.m_cem", 1.90))
        n = float(params.get("sw.n_sat", 2.00))
        rw = float(params.get("sw.rw", 0.03))
        mslope = float(params.get("sw.mslope", 1.00))
        B = float(params.get("sw.B", 12.10))
        cbw_intercept = float(params.get("cbw_intercept", 0.1))
    
        # clean arrays
        phit = phit.clip(lower=1e-6, upper=0.60)
        rt = rt.clip(lower=1e-6)
        vsh = vsh.clip(lower=0.0, upper=1.0)
    
        # Base Archie with Vsh correction
        rt_eff = rt * (1.0 + mslope * vsh)
        sw_archie = ((rw) / (rt * (phit ** m))) ** (1.0 / n)
    

        cbw_used  = cbw_intercept * vsh

        '''
        if cbw is not None and cbw.notna().sum() > 0:
            cbw = cbw.clip(lower=0.0)
            sw = sw_archie * (1.0 + 0.25 * cbw)
            cbw_used = cbw
        else:
            cbw_est = cbw_intercept * vsh
            sw = sw_archie * (1.0 + 0.25 * cbw_est)
            cbw_used = cbw_est
        '''
    
        sw = sw_archie

        sw = sw.clip(lower=0.0, upper=1.0)
        bvw = (sw * phit).clip(lower=0.0)

        phie = (phit - cbw_used).clip(lower=0.0)
        phie = np.minimum(phie, phit)
  
        # --------------------------------------------------------------
        # Store outputs only into ZoI rows
        # --------------------------------------------------------------
        #####out.loc[mask, "PHIT_USED"] = phit.values
        #####out.loc[mask, "RT_USED"] = rt.values
        ####out.loc[mask, "VSH_USED"] = vsh.values
        ######out.loc[mask, "RT_EFF"] = rt_eff.values
        out.loc[mask, "CBW"] = cbw_used.values
        out.loc[mask, "PHIE"] = phie.values
        out.loc[mask, "SW"] = sw.values
        out.loc[mask, "BVW"] = bvw.values
        
        state = self.controller.get_state()
        state.analysis_df = out
        
        if hasattr(self.controller, "rebuild_view"):
            self.controller.rebuild_view()
        
        if hasattr(self.controller, "refresh_plots"):
            self.controller.refresh_plots()
        elif hasattr(self.controller, "update_plots"):
            self.controller.update_plots()
 
       #if qv is not None:
        #   out.loc[mask, "QV_USED"] = qv.values
    
        #if cbw is not None:

 




    # ------------------------------------------------------------------
    # Waxman-Smits Calculation
    # ------------------------------------------------------------------   
    def calculate_sw(self):
        df = self._analysis_df()
        state = self._state()
    
        if df is None or df.empty or state is None:
            QMessageBox.warning(self, "Sw", "No analysis dataframe loaded.")
            return
    
        self._store_params_from_ui()
        self._store_curve_picks_from_ui()
    
        picks = self._curve_picks()
        params = self._params()
    
        phit_curve = picks.get("phit_curve", "").strip()
        rt_curve = picks.get("rt_curve", "").strip()
        vsh_curve = picks.get("vsh_curve", "").strip()
        cbw_curve = picks.get("cbw_curve", "").strip()
    
        missing = []
        if not phit_curve:
            missing.append("PHIT")
        if not rt_curve:
            missing.append("RT")
        if not vsh_curve:
            missing.append("VSH")
    
        if missing:
            QMessageBox.warning(
                self,
                "Sw",
                f"Missing required curve selections: {', '.join(missing)}"
            )
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
            QMessageBox.warning(self, "Sw", "No rows available in the selected ZoI.")
            return
    
        subset = out.loc[mask].copy()
    
        try:
            phit = pd.to_numeric(subset[phit_curve], errors="coerce")
            rt = pd.to_numeric(subset[rt_curve], errors="coerce")
            vsh = pd.to_numeric(subset[vsh_curve], errors="coerce")
        except Exception as e:
            QMessageBox.critical(self, "Sw", f"Failed reading selected curves:\n{e}")
            return



    
    
        cbw = None
        if cbw_curve and cbw_curve in subset.columns:
            cbw = pd.to_numeric(subset[cbw_curve], errors="coerce")
    
        # parameters
        m_cem   = float(params.get("sw.m_cem", 1.90))
        n_sat   = float(params.get("sw.n_sat", 2.00))
        rw      = float(params.get("sw.rw", 0.03))
        mslope  = float(params.get("sw.mslope", 1.00))

        cbw_intercept = float(params.get("cbw_intercept", 0.1))

        T_F    = 150
        den_fl = 1.1
    
        # clean arrays
        phit = phit.clip(lower=1e-6, upper=0.60)
        rt = rt.clip(lower=1e-6)
        vsh = vsh.clip(lower=0.0, upper=1.0)


        # --- Scalar Rw75, SAL, and B
        rw = max(float(rw), 1e-6)
        T_F = 150
        Rw75 = ((T_F + 6.77) * rw) / (75.0 + 6.77)
        SAL = float(compute_sal_kppm_from_rw75(Rw75))
        B = float(compute_bdacy(T_F, rw))


        if cbw is not None and cbw.notna().sum() > 0:
            cbw = np.clip(cbw, 0.0, None)
        else:
            cbw = cbw_intercept * vsh
            cbw = np.clip(cbw, 0.0, None)

        Swb = np.clip(cbw / phit, 0.0, 1.0)

        # --- Qv from scalar SAL
        denom = (0.6425 / np.sqrt(max(den_fl * SAL, 1e-12)) + 0.22)
        Qv = np.clip(Swb / denom, 0.0, 5.0)
        



   
        MSTAR = vsh * mslope + m_cem

        try:
            from petrocore.workflow.waxman_smits import waxman_smits_sw_iterative
    
            Sw_cp = waxman_smits_sw_iterative(
                rt=rt,
                phit=phit,
                qv=Qv,
                rw=rw,          # scalar
                m= MSTAR,        # array
                n=n_sat,
                B=B,            # scalar
                max_iter=60,
                tol=1e-6,
                sw0=None,
            )
            Sw_cp = np.clip(np.asarray(Sw_cp, dtype=float), 1e-4, 1.0)
    

        except Exception as e:
            print(f"[WS] iterative solver unavailable, using fallback: {e}")
            good = np.isfinite(rt) & np.isfinite(phit) & (rt > 0) & (phit > 0)
            Sw_cp = np.full(len(subset), np.nan, dtype=float)
            Sw_cp[good] = (
                (rw / rt[good]) *
                (1.0 / np.clip(phit[good] ** MSTAR[good], 1e-12, None))
            ) ** (1.0 / n_sat)
            Sw_cp = np.clip(Sw_cp, 1e-4, 1.0)



        
   
        # --- Outputs
        bvwt = phit * Sw_cp
        bvwe = np.clip(phit * Sw_cp - cbw, 0.0, None)
    
        phie = np.clip(phit - cbw, 0.0, None)
        phie = np.minimum(phie, phit)
 
        #print("Qv",Qv, ", MSTAR",MSTAR,", B",B,", Sw_cp",Sw_cp,", BVWE",bvwe, ", rw",rw, ", SAL",SAL,", phie",phie)
   
        # --------------------------------------------------------------
        # Store outputs only into ZoI rows
        # --------------------------------------------------------------
        #####out.loc[mask, "PHIT_USED"] = phit
        #####out.loc[mask, "RT_USED"] = rt
        #####out.loc[mask, "VSH_USED"] = vsh
        out.loc[mask, "CBW"] = cbw
        out.loc[mask, "PHIE"] = phie
        out.loc[mask, "SW"] = Sw_cp
        out.loc[mask, "BVW"] = bvwe
        
        state = self.controller.get_state()
        state.analysis_df = out
        
        if hasattr(self.controller, "rebuild_view"):
            self.controller.rebuild_view()
        
        if hasattr(self.controller, "refresh_plots"):
            self.controller.refresh_plots()


        elif hasattr(self.controller, "update_plots"):
            self.controller.update_plots()

        
        self.status_label.setText(
            f"Sw calculated using PHIT={phit_curve}, RT={rt_curve}, "
            f"VSH={vsh_curve}, CBW={cbw_curve or 'estimated'}."
        )


        lines = [
            "Method Used for SW",
            "",
            "Method used to calculate SW = Waxman-Smits",
            "",
            f"PHIT curve: {phit_curve or 'None'}",
            f"RT curve: {rt_curve or 'None'}",
            f"VSH curve: {vsh_curve or 'None'}",
            f"CBW curve: {cbw_curve or 'estimated from VSH'}",
            "",
            f"m = {m_cem:.2f}",
            f"n = {n_sat:.2f}",
            f"Rw = {rw:.4f}",
            f"Vsh correction slope = {mslope:.2f}",
            f"CBW intercept = {cbw_intercept:.2f}",
            f"SAL = {SAL:.4f} kppm",
            f"B = {B:.4f}",
        ]
        self.notes.setPlainText("\n".join(lines))






 

    
    
    
