##CurvePickerPanel
from __future__ import annotations



from PySide6.QtWidgets import QWidget, QFormLayout, QComboBox, QPushButton, QLabel
from PySide6.QtWidgets import QComboBox  # already imported, just FYI

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import os
from pathlib import Path

CHARTBOOK_FILES = {
    "TNPH ρf=1.00 (SLB)": "TNPH_1pt0.xlsx",
    "CNL ρf=1.0 (SLB)"  : "CNL_1pt0.xlsx",
    "CNL ρf=1.1 (SLB)"  : "CNL_1pt1.xlsx",
    "TNPH ρf=1.19 (SLB)": "TNPH_1pt19.xlsx",
}
DEFAULT_CHARTBOOK_KEY = "CNL ρf=1.1 (SLB)"



DEFAULT_CANDIDATES: Dict[str, List[str]] = {
    "gr_curve":   ["GR_EDTC", "HSGR", "GR", "SGR", "HCGR"],
    "cgr_curve":  ["HCGR", "CGR", "GR_EDTC"],
    "rhob_curve": ["RHOZ", "RHOB"],
    "tnph_curve": ["NPOR", "TNPH", "NPHI", "CNL"],
    "rt_curve":   ["AT90", "AF90", "AT60", "AF60", "AT30", "AF30", "AT20", "AF20", "ILD", "RT"],
    "dtco_curve": ["DTCO", "DTC", "AC"],
    "pef_curve":  ["PEFZ", "PEF"],
    "tcmr_curve": ["PHIT_NMR", "TCMR"],
    "cmrp_curve": ["PHIE_NMR", "CMRP_3MS", "CMRP3MS", "CMRP"],
    "cbw_curve":  ["CBW"],
    "ffi_curve":  ["FFI", "CMFF"],
    "bvie_curve": ["BVIE", "BVI_E"],
}


def _charts_dir() -> Path:
    # curve_picker_panel.py is .../apps/merge_gui/ui_panels/
    # charts live in .../apps/merge_gui/data/
    here = Path(__file__).resolve()
    return (here.parents[1] / "data")  # ui_panels -> merge_gui -> data

def _chart_path(filename: str) -> str:
    p = _charts_dir() / filename
    return str(p)






def first_present(cols: List[str], candidates: List[str]) -> Optional[str]:
    s = set(cols)
    for c in candidates:
        if c in s:
            return c
    return None

def _norm(x, xmin, xmax):
    x = np.asarray(x, dtype=float)
    return (x - xmin) / (xmax - xmin)

def build_chart_payload(df_chart: pd.DataFrame,
                        neutron_col="Neutron",
                        rhob_col="RHOB",
                        por_col="Porosity",
                        rhoma_col="Rho_Matrix",
                        cnl_min=-0.05, cnl_max=0.60,
                        rhob_min=1.90, rhob_max=3.00):
    cnl = pd.to_numeric(df_chart[neutron_col], errors="coerce").to_numpy(float)
    rho = pd.to_numeric(df_chart[rhob_col], errors="coerce").to_numpy(float)
    por = pd.to_numeric(df_chart[por_col], errors="coerce").to_numpy(float)
    rma = pd.to_numeric(df_chart[rhoma_col], errors="coerce").to_numpy(float)

    m = np.isfinite(cnl) & np.isfinite(rho) & np.isfinite(por) & np.isfinite(rma)
    cnl, rho, por, rma = cnl[m], rho[m], por[m], rma[m]

    X = np.column_stack([
        _norm(cnl, cnl_min, cnl_max),
        _norm(rho, rhob_min, rhob_max),
    ])
    tree = cKDTree(X)

    return dict(tree=tree, por=por, rma=rma,
                cnl_min=cnl_min, cnl_max=cnl_max,
                rhob_min=rhob_min, rhob_max=rhob_max)



def chartbook_knn(payload, tnph, rhob, k=3, eps=1e-6):
    """
    kNN inverse-distance weighted interpolation on digitized ND chartbook.

    Returns full-length arrays:
      phit_est, rhomaa_est
    with NaN where TNPH or RHOB are not finite.
    """

    tnph = np.asarray(tnph, dtype=float)
    rhob = np.asarray(rhob, dtype=float)

    n = len(tnph)
    phit = np.full(n, np.nan, dtype=float)
    rhomaa = np.full(n, np.nan, dtype=float)

    # Only query where both logs are finite
    valid = np.isfinite(tnph) & np.isfinite(rhob)
    if not np.any(valid):
        return phit, rhomaa

    # ---- Clip to chart limits (prevents wild extrapolation)
    tn = np.clip(tnph[valid], payload["cnl_min"], payload["cnl_max"])
    rb = np.clip(rhob[valid], payload["rhob_min"], payload["rhob_max"])

    # ---- Normalize to chart space
    q = np.column_stack([
        _norm(tn, payload["cnl_min"], payload["cnl_max"]),
        _norm(rb, payload["rhob_min"], payload["rhob_max"]),
    ])

    # ---- KDTree query
    dist, idx = payload["tree"].query(q, k=k, workers=-1)

    # Ensure 2D outputs if k == 1
    if k == 1:
        dist = dist[:, None]
        idx = idx[:, None]

    # ---- Inverse-distance weights
    w = 1.0 / np.maximum(dist, eps)

    por_n = payload["por"][idx]
    rma_n = payload["rma"][idx]

    phit_valid = (w * por_n).sum(axis=1) / w.sum(axis=1)
    rhomaa_valid = (w * rma_n).sum(axis=1) / w.sum(axis=1)

    # ---- Insert back into full-length arrays
    phit[valid] = phit_valid
    rhomaa[valid] = rhomaa_valid

    return phit, rhomaa






class CurvePickerPanel(QWidget):
    """
    Dropdowns for selecting the active curve per family.
    Stores selections in controller.state.params and refreshes plots.
    """

    def __init__(self, controller, candidates: Dict[str, List[str]] | None = None):
        super().__init__()
        self.controller = controller
        self.candidates = candidates or DEFAULT_CANDIDATES

        self.form = QFormLayout(self)
        self.info = QLabel("Load a dataset to populate curve lists.")
        self.form.addRow(self.info)

        self.combos: Dict[str, QComboBox] = {}

        
        self.btn_autopick = QPushButton("Auto-pick from candidates")
        self.btn_apply = QPushButton("Apply + Refresh Plots")
        self.btn_chartbook_phi = QPushButton("1) Calc Chartbook PHI (NEUT–RHOB)")
        
        # --- Chartbook fluid density selector (SLB digitized charts)
        self.cb_chartbook = QComboBox()
        for k in CHARTBOOK_FILES.keys():
            self.cb_chartbook.addItem(k, userData=k)
        # default selection
        idx0 = self.cb_chartbook.findText(DEFAULT_CHARTBOOK_KEY)
        self.cb_chartbook.setCurrentIndex(idx0 if idx0 >= 0 else 0)
        
        self.btn_autopick.clicked.connect(self.autopick)
        self.btn_apply.clicked.connect(self.apply_to_state)
        self.btn_chartbook_phi.clicked.connect(self.calc_chartbook_phi)
        
        # When density changes, rebuild payload (but don’t recompute curves until button press)
        self.cb_chartbook.currentIndexChanged.connect(self._on_chartbook_changed)
        
        self.form.addRow(self.btn_autopick)
        self.form.addRow("Chartbook (fluid density)", self.cb_chartbook)
        self.form.addRow(self.btn_chartbook_phi)
        self.form.addRow(self.btn_apply)
        

    def _on_chartbook_changed(self, *_):
        # Drop cached payload so next compute rebuilds with selected SLB chart
        if hasattr(self, "_chartbook_payload"):
            delattr(self, "_chartbook_payload")
        self.info.setText("Chartbook selection changed. Click '1) Calc Chartbook PHI' to recompute.")



    def populate_from_columns(self, columns: List[str]):
        # wipe existing combo rows
        for cb in self.combos.values():
            cb.blockSignals(True)
            cb.clear()
            cb.blockSignals(False)
        self.combos = {}
    
        # Fixed rows in the form:
        #   Row 0: info label
        #   Tail rows (bottom, always kept):
        #     - Auto-pick button
        #     - Chartbook dropdown
        #     - Chartbook compute button
        #     - Apply button
        FIXED_TAIL_ROWS = 4
        KEEP_ROWS = 1 + FIXED_TAIL_ROWS  # info + tail controls
    
        # Remove any previously inserted curve-combo rows (rows between info and tail controls)
        while self.form.rowCount() > KEEP_ROWS:
            self.form.removeRow(1)
    
        self.info.setText(f"{len(columns)} curves available. Choose active curves:")
    
        # Insert curve-family combos above the fixed tail controls
        insert_at = self.form.rowCount() - FIXED_TAIL_ROWS
    
        for key in self.candidates.keys():
            cb = QComboBox()
            cb.addItem("(none)", userData=None)
            for col in columns:
                cb.addItem(col, userData=col)
    
            cb.currentIndexChanged.connect(self._on_change)
            self.combos[key] = cb
    
            self.form.insertRow(insert_at, key, cb)
            insert_at += 1  # next combo goes under the previous one
    
        # Set initial picks without triggering lots of refreshes
        for cb in self.combos.values():
            cb.blockSignals(True)
        self.autopick()
        for cb in self.combos.values():
            cb.blockSignals(False)
    
        # One clean update at end
        self.apply_to_state()



    def _columns_from_state(self) -> List[str]:
        # Prefer Dataset if present
        ds = getattr(self.controller.state, "dataset", None)
        if ds is not None:
            return list(ds.data.columns)

        # fallback: analysis_df if present
        df = getattr(self.controller.state, "analysis_df", None)
        if df is not None:
            return list(df.columns)

        return []

    def autopick(self):
        cols = self._columns_from_state()
        if not cols:
            return

        for key, cand in self.candidates.items():
            cb = self.combos.get(key)
            if cb is None:
                continue

            pick = first_present(cols, cand)
            if pick is None:
                cb.setCurrentIndex(0)
            else:
                idx = cb.findText(pick)
                cb.setCurrentIndex(idx if idx >= 0 else 0)

    def _on_change(self, *_):
        # If you prefer “Apply only”, comment this out
        self.apply_to_state()

    def apply_to_state(self):
        p = getattr(self.controller.state, "params", None)
        if p is None:
            self.controller.state.params = {}
            p = self.controller.state.params

        for key, cb in self.combos.items():
            val = cb.currentData()
            if val is None:
                p.pop(key, None)
            else:
                p[key] = val

        # refresh plots using new active curves
        if hasattr(self.controller, "update_plots"):
            self.controller.update_plots()
        else:
            self.controller.refresh_plots()


    def calc_chartbook_phi(self):
        df = getattr(self.controller.state, "analysis_df", None)
        if df is None or df.empty:
            self.info.setText("No analysis_df loaded.")
            return
    
        p = getattr(self.controller.state, "params", {}) or {}
        rhob_curve = p.get("rhob_curve") or first_present(list(df.columns), self.candidates["rhob_curve"])
        tnph_curve = p.get("tnph_curve") or first_present(list(df.columns), self.candidates["tnph_curve"])
    
        if not rhob_curve or not tnph_curve:
            self.info.setText("Need RHOB and TNPH families present to compute Chartbook PHI.")
            return
    
        rhob = pd.to_numeric(df[rhob_curve], errors="coerce").to_numpy(float)
        tnph = pd.to_numeric(df[tnph_curve], errors="coerce").to_numpy(float)
    
        # Cache payload (build once)
        if not hasattr(self, "_chartbook_payload"):
            key = DEFAULT_CHARTBOOK_KEY
            if hasattr(self, "cb_chartbook") and self.cb_chartbook is not None:
                try:
                    key = self.cb_chartbook.currentData() or DEFAULT_CHARTBOOK_KEY
                except RuntimeError:
                    key = DEFAULT_CHARTBOOK_KEY
        
            rel_name = CHARTBOOK_FILES.get(key, CHARTBOOK_FILES[DEFAULT_CHARTBOOK_KEY])
            file_path = _chart_path(rel_name)
        
            
            df_chart = pd.read_excel(file_path, index_col=False)
            #self.controller.state.chartbook_df = df_chart   # <-- ADD THIS LINE
            self.controller.state.chartbook_df = df_chart
            self._chartbook_payload = build_chart_payload(df_chart)


        
        phit, rhomaa = chartbook_knn(self._chartbook_payload, tnph, rhob, k=3)
    
        df["PHIT_CHART"] = phit
        df["RHOMAA_CHART"] = rhomaa
        self.controller.state.analysis_df = df
    
        # Make new curves visible in dropdowns + refresh plots
        self.populate_from_columns(list(df.columns))
    


