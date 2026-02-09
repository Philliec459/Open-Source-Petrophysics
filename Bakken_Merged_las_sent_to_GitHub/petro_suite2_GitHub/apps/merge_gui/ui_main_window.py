# apps/merge_gui/ui_main_window.py
from __future__ import annotations

print(">>> LOADING ui_main_window.py <<<")

import os
import re
import pandas as pd
import lasio

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QMainWindow,
    QDockWidget,
    QTextEdit,
    QFileDialog,
)

from petrocore.models.dataset import Dataset
from petrocore.workflow.state import WorkflowState
from apps.merge_gui.controllers.workflow_controller import WorkflowController

from apps.merge_gui.ui_panels.depth_panel import DepthRangePanel
from apps.merge_gui.ui_panels.tops_panel import TopsPanel
from apps.merge_gui.ui_panels.plots_panel import PlotsPanel
from apps.merge_gui.ui_panels.curve_picker_panel import CurvePickerPanel
from apps.merge_gui.controllers.workflow_controller import WorkflowController
from apps.merge_gui.ui_panels.plots_panel import PlotsPanel


# -----------------------------
# Families + diagnostics
# -----------------------------
def default_families_map():
    return {
        "GR":   ["GR_EDTC","HSGR", "GR", "SGR", "HGR"],
        "CGR":  ["HCGR", "CGR"],
        "RHOB": ["RHOZ", "RHOB"],
        "TNPH": ["TNPH", "NPHI", "CNL", "NPOR"],
        "RT":   ["AT90", "AF90", "AO90", "ILD", "RT","RD"],
        "TCMR": ["PHIT_NMR", "TCMR", "MPHIS"],
        "CMRP": ["PHIE_NMR", "CMRP_3MS", "CMRP3MS", "CMRP", "MPHI"],
        "CBW":  ["CBW"],
        "BVIE": ["BVIE", "BVI_E","MBVI"],
        "FFI":  ["FFI", "CMFF","MFFI"],
        "PEF":  ["PEFZ", "PEF","PE"],
        "DTCO": ["DTCO", "DT", "AC"],
    }


def quick_curve_audit(df: pd.DataFrame):
    cols = list(df.columns)

    rt_cands = ["AT90", "AF90", "AO90", "ILD", "RT", "RES", "RLA1", "RDEP", "RD", "RXOZ", "RXO8"]
    nmr_cands = [
        "PHIT_NMR", "PHIE_NMR", "TCMR", "CMRP_3MS", "CMRP3MS", "CMRP",
        "CBW", "FFI", "BVI","MBVI", "BVIE", "MPHI", "MPHIS"
    ]

    def present(cands):
        return [c for c in cands if c in cols]

    print("\n=== CURVE AUDIT ===")
    print("Columns:", len(cols))
    print("Rt present:", present(rt_cands))
    print("NMR present:", present(nmr_cands))


def resolve_family_names(df: pd.DataFrame, families_map: dict):
    cols = set(df.columns)
    resolved = {}
    for fam, cands in families_map.items():
        hit = next((c for c in cands if c in cols), None)
        resolved[fam] = hit

    print("\n=== RESOLVED FAMILY CURVES (key) ===")
    for fam in ("RT", "TCMR", "CMRP", "CBW", "FFI", "BVIE"):
        print(f"{fam:5s} -> {resolved.get(fam)}")

    return resolved


# -----------------------------
# Main window
# -----------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Petro Workflow GUI")

        self.state = WorkflowState()

        self.console = QTextEdit()
        self.console.setReadOnly(True)

        # 1) Create controller first (plots_panel can be None for the moment)
        self.controller = WorkflowController(self.state, None, self.console)

        # 2) Now create PlotsPanel with controller
        self.plots_panel = PlotsPanel(self.controller)
        self.plots_panel.state = self.state
        self.setCentralWidget(self.plots_panel)

        # 3) Attach plots_panel back to controller
        self.controller.plots_panel = self.plots_panel

        # 4) Other panels that need controller
        self.curve_picker = CurvePickerPanel(self.controller)

        self.resize(1400, 800)

        self._build_menu()
        self._build_docks()


    # -------------------------
    # UI docks
    # -------------------------
    def _build_docks(self):
        # ---- Curves dock (left) ----
        dock_curves = QDockWidget("Curves", self)
        dock_curves.setWidget(self.curve_picker)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock_curves)
    
        # ---- Controls + Tops as TABBED dock ----
        self.depth_panel = DepthRangePanel(self.controller)
        self.tops_panel = TopsPanel(self.controller)
    
        dock_controls = QDockWidget("Controls / Tops", self)
        dock_controls.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
    
        from PySide6.QtWidgets import QTabWidget
        tabs = QTabWidget()
        tabs.addTab(self.depth_panel, "Depth")
        tabs.addTab(self.tops_panel, "Tops")
    
        dock_controls.setWidget(tabs)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock_controls)
    
        # ---- Log / console at bottom ----
        dock_log = QDockWidget("Log", self)
        dock_log.setWidget(self.console)
        dock_log.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetClosable)

        self.addDockWidget(Qt.BottomDockWidgetArea, dock_log)
    
        # ---- Let Qt auto-arrange nicely ----
        self.resizeDocks([dock_curves, dock_controls], [200, 260], Qt.Horizontal)
        self.resizeDocks([dock_log], [140], Qt.Vertical)

    # -------------------------
    # Menu
    # -------------------------
    def _build_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")

        act_open = QAction("Open LAS…", self)
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self._on_open_las)
        file_menu.addAction(act_open)

        act_tops_core = QAction("Open Tops (Core)…", self)
        act_tops_core.triggered.connect(self._on_open_tops_core)
        file_menu.addAction(act_tops_core)

        file_menu.addSeparator()

        act_quit = QAction("Quit", self)
        act_quit.setShortcut("Ctrl+Q")
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

    # -------------------------
    # File -> Open LAS
    # -------------------------
    def _on_open_las(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open LAS",
            "",
            "LAS files (*.las *.LAS);;All files (*.*)"
        )
        if not path:
            return
        self.open_las_file(path)

    def open_las_file(self, path: str):
        las = lasio.read(path)

        # ---- WELL name from LAS header (preferred over filename) ----
        try:
            well_name = str(las.well.WELL.value).strip()
        except Exception:
            well_name = ""
        
        if well_name:
            self.controller.state.well_name = well_name
            self.console.append(f"[WELL] LAS WELL = {well_name}")
        else:
            self.controller.state.well_name = ""
            self.console.append("[WELL] LAS WELL not found; will fallback to filename")

        
        df = las.df()
        df.index = df.index.astype(float)

        # Diagnostics
        quick_curve_audit(df)
        resolve_family_names(df, default_families_map())

        ds = Dataset(
            data=df,
            families_map=default_families_map(),
            name=os.path.basename(path),
        )

        # send dataset into workflow
        self.controller.set_dataset(ds)

        # update curve picker
        self.curve_picker.populate_from_columns(list(df.columns))

        self.console.append(f"Loaded LAS: {os.path.basename(path)}")

        # after controller.set_dataset(ds)
        dmin, dmax = float(df.index.min()), float(df.index.max())
        self.depth_panel.set_depth_limits(dmin, dmax)





    # -------------------------
    # File -> Open Tops (Core)
    # -------------------------

    def _infer_well_name_for_tops(self) -> str:
        # 1) Prefer LAS header-derived well name
        w = getattr(self.controller.state, "well_name", "") or ""
        w = str(w).strip()
        if w:
            return w
    
        # 2) Fallback: LAS base filename
        ds = getattr(self.controller.state, "dataset", None)
        if ds is None:
            return ""
        name = str(getattr(ds, "name", "")).strip()
        if name.lower().endswith(".las"):
            name = name[:-4]
        return name
    



    
    def _on_open_tops_core(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Tops (Core) Excel",
            "",
            "Excel files (*.xlsx *.xls);;All files (*.*)"
        )
        if not path:
            return

        well_key = self._infer_well_name_for_tops()
        self.open_tops_core_xlsx(path, well_key)

    def open_tops_core_xlsx(self, tops_xlsx_path: str, well_key: str):
        """
        Reads master tops file with columns:
          - Welll Name  (note 3 'l' per your file)
          - Well No
          - Formation
          - Top (ft)

        Filters to selected well (normalized) using LAS base filename.
        Stores state.tops_df with columns: Top, Depth.
        """
        def normalize_name(s: str) -> str:
            return re.sub(r"\s+", " ", str(s).strip().upper())

        if not os.path.exists(tops_xlsx_path):
            self.console.append(f"⚠️ Tops file not found: {os.path.abspath(tops_xlsx_path)}")
            return

        try:
            df = pd.read_excel(
                tops_xlsx_path,
                usecols=["Welll Name", "Well No", "Formation", "Top (ft)"]
            ).copy()
        except Exception as e:
            self.console.append(f"⚠️ Failed reading tops Excel: {e}")
            return

        df.columns = ["well_name", "well_no", "formation", "top_ft"]
        df["well_name_norm"] = df["well_name"].apply(normalize_name)
        df["formation"] = df["formation"].astype(str).str.strip()
        df["top_ft"] = pd.to_numeric(df["top_ft"], errors="coerce")

        df = df.dropna(subset=["top_ft"]).sort_values(["well_name_norm", "top_ft"]).reset_index(drop=True)

        well_norm = normalize_name(well_key)
        df_well = df[df["well_name_norm"] == well_norm].copy()
        df_well = df_well.sort_values("top_ft").reset_index(drop=True)

        if df_well.empty:
            self.console.append(f"⚠️ No tops found for well='{well_key}' (norm='{well_norm}')")
            self.controller.state.tops_df = pd.DataFrame(columns=["Top", "Depth"])
            self.controller.refresh_plots()
            return

        tops_df = pd.DataFrame({
            "Top": df_well["formation"].astype(str),
            "Depth": df_well["top_ft"].astype(float),
        })

        self.controller.state.tops_df = tops_df
        self.controller.refresh_plots()
        self.console.append(f"✅ Tops loaded for '{well_key}' | n={len(tops_df)}")

    # -------------------------
    # Demo loader
    # -------------------------
    def demo_load(self, ds: Dataset):
        self.controller.set_dataset(ds)

        df = ds.data
        dmin, dmax = float(df.index.min()), float(df.index.max())
        self.depth_panel.set_depth_limits(dmin, dmax)
