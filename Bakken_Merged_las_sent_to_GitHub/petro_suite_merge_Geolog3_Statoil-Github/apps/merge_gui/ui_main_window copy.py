from pathlib import Path

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QIcon, QFont
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QTreeWidget,
    QTreeWidgetItem,
    QStackedWidget,
    QSplitter,
    QTabWidget,
    QGroupBox,
    QStyle,
    QToolButton,
    QSizePolicy,
)

from petrocore.workflow.state import WorkflowState
from apps.merge_gui.controllers.main_controller import MainController
from petrocore.services.curve_set_service import (
    load_curve_set_config,
    classify_curves_with_winners,
)

# Core / workflow panels
from apps.merge_gui.ui_panels.workflow_home_panel import WorkflowHomePanel
from apps.merge_gui.ui_panels.load_data_panel import LoadDataPanel
from apps.merge_gui.ui_panels.tops_panel import TopsPanel
from apps.merge_gui.ui_panels.phit_panel import PhitPanel
from apps.merge_gui.ui_panels.vsh_panel import VshPanel
from apps.merge_gui.ui_panels.sw_panel import SwPanel
from apps.merge_gui.ui_panels.well_summary_panel import WellSummaryPanel

# Workspace / plot panels
from apps.merge_gui.ui_panels.plots_panel_initial import PlotsPanelInitial
from apps.merge_gui.ui_panels.plots_panel_vsh import PlotsPanelVsh
from apps.merge_gui.ui_panels.plots_panel_ws import PlotsPanelWS
from apps.merge_gui.ui_panels.plots_panel_final import PlotsPanelFinal
from apps.merge_gui.ui_panels.save_well_panel import SaveWellPanel
from apps.merge_gui.ui_panels.open_project_panel import OpenProjectPanel


from PySide6.QtWidgets import QLabel

def make_title(text):
    lbl = QLabel(text)
    f = QFont()
    f.setPointSize(11)
    f.setBold(True)
    lbl.setFont(f)
    return lbl



NAV_GROUPS = {
    "Home": [
        ("Home", "home"),
    ],
    "Load Well Data": [
        ("Open Project", "open_project"),
        ("Load Data", "load_data"),
        ("Initial Plot", "initial_plot"),
    ],
    "Prelim Calculations": [
        ("Calc PHIT", "phit"),
        ("Calc Vsh", "vsh"),
        ("Calc Sw", "sw"),
        ("Well Summary", "summary"),
    ],
    "Interactive Petrophysical Workspace": [
        ("Vsh HL", "workspace_vsh"),
        ("Sw WS", "workspace_sw"),
    ],
    "Final Depth Plot":[
        ("Final Depth Plot", "final_plot"),
    ],
    "Project":[
        ("Save Well", "save_well"),
    ],
}

ICON_MAP = {
    "home": ["nav/home.svg", "home.svg", "home.png"],
    "open_project": ["nav/load_data.svg", "load_data.svg", "folder_open.png"],
    "load_data": ["nav/load_data.svg", "load_data.svg", "folder_open.png"],
    "initial_plot": ["nav/initial_plot.svg", "initial_plot.svg", "seismograph.png", "chart.png"],
    "phit": ["workflow/phit.svg", "phit.svg", "calculate.png", "sigma.png"],
    "vsh": ["workflow/vsh.svg", "vsh.svg", "change_history.png", "shale.png"],
    "sw": ["workflow/sw.svg", "sw.svg", "initial.png", "water.png"],
    "summary": ["workflow/summary.svg", "summary.svg", "report.png"],
    "workspace_vsh": ["workspace/workspace_vsh.svg", "workspace_vsh.svg", "workspace.png", "shale.png"],
    "workspace_sw": ["workspace/workspace_sw.svg", "workspace_sw.svg", "workspace.png", "sw.png"],
    "final_plot": ["workspace/final_plot.svg", "final_plot.svg", "logs.png", "final_depth.png"],
    "save_well": ["project/save_well.svg", "save_well.svg", "save.png", "disk.png"],
}

LABEL_TO_ICON_KEY = {
    "Home": "home",
    "Load Data": "load_data",
    "Initial Plot": "initial_plot",
    "Calc PHIT": "phit",
    "Calc Vsh": "vsh",
    "Calc Sw": "sw",
    "Well Summary": "summary",
    "Vsh HL": "workspace_vsh",
    "Sw WS": "workspace_sw",
    "Save Well": "save_well",
}



# --- Font hierarchy ---
FONT_SMALL = QFont("Segoe UI", 9)     # left panel / data
FONT_MED   = QFont("Segoe UI", 10)    # default
FONT_LARGE = QFont("Segoe UI", 11)    # right workflow panels
FONT_TITLE = QFont("Segoe UI", 12, QFont.Bold)



class MainWindow(QMainWindow):
    """
    3-pane petro_suite_merge layout

    LEFT
      - Upper: Well data / merged data by sets
      - Lower: Tops / Zone

    CENTER
      - Main stacked canvas starting on Home

    RIGHT
      - Grouped workflow launcher with icon + text buttons
    """

    def __init__(self):
        super().__init__()

        self.setWindowTitle("petro_suite")
        self.resize(1500, 920)
        self.move(40, 40)
        self.setMinimumSize(1200, 780)

        self.state = WorkflowState()
        if not hasattr(self.state, "params") or not isinstance(self.state.params, dict):
            self.state.params = {}

        self.controller = MainController(self.state, self)

        self.panel_map: dict[str, QWidget] = {}
        self.nav_button_map: dict[str, QToolButton] = {}
        self._yaml_winners_map: dict[str, str] = {}

        self._build_ui()
        self._build_panels()
        self._populate_navigation()

        self._select_panel("Home")
        self.refresh_data_views()


        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #f4f6f8;
                color: #22303c;
                font-size: 12pt;
            }
    
            QLabel {
                font-weight: normal;
                color: #22303c;
            }
    
            QGroupBox {
                background-color: white;
                border: 1px solid #d6dde5;
                border-radius: 4px;
                margin-top: 10px;
                padding-top: 8px;
                font-weight: bold;
            }
    
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px 0 4px;
            }
    
            QPushButton {
                background-color: #e8edf2;
                border: 1px solid #c8d1dc;
                border-radius: 4px;
                padding: 5px 10px;
            }
    
            QPushButton:hover {
                background-color: #dde5ec;
            }
    
            QComboBox, QDoubleSpinBox, QLineEdit, QTextEdit {
                background-color: white;
                border: 1px solid #cfd7df;
                border-radius: 3px;
                padding: 4px;
            }
    
            QTreeWidget, QListWidget, QTableWidget, QTableView {
                background-color: white;
                border: 1px solid #d6dde5;
            }
    
            QHeaderView::section {
                background-color: #edf1f5;
                border: 1px solid #d6dde5;
                padding: 4px;
                font-weight: bold;
            }
    
            QTabWidget::pane {
                border: 1px solid #d6dde5;
                background: white;
            }
    
            QTabBar::tab {
                background: #e9eef3;
                border: 1px solid #d6dde5;
                padding: 6px 12px;
                margin-right: 2px;
            }
    
            QTabBar::tab:selected {
                background: white;
                border-bottom: 1px solid white;
            }
        """)
    


    # ------------------------------------------------------------------
    # UI shell
    # ------------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        outer_layout = QHBoxLayout(central)
        outer_layout.setContentsMargins(4, 4, 4, 4)
        outer_layout.setSpacing(4)

        self.main_splitter = QSplitter(Qt.Horizontal)

        # ==============================================================
        # LEFT PANE
        # ==============================================================
        self.left_splitter = QSplitter(Qt.Vertical)
        self.left_splitter.setChildrenCollapsible(False)

        # -------- Left upper: well data / merged data by sets ----------
        self.left_top_container = QWidget()
        self.left_top_layout = QVBoxLayout(self.left_top_container)
        self.left_top_layout.setContentsMargins(0, 0, 0, 0)
        self.left_top_layout.setSpacing(4)

        self.data_group = QGroupBox("Well Data / Curve Sets")
        self.data_group_layout = QVBoxLayout(self.data_group)
        self.data_group_layout.setContentsMargins(6, 6, 6, 6)
        self.data_group_layout.setSpacing(4)

        self.data_tabs = QTabWidget()

        tree_css = """
            QTreeWidget {
                font-size: 11px;
            }
        """
 
        self.merged_tree = QTreeWidget()
        self.merged_tree.setHeaderHidden(True)
        self.merged_tree.setAlternatingRowColors(True)
        self.merged_tree.setStyleSheet(tree_css)
        
        

        self.raw_tree = QTreeWidget()
        self.raw_tree.setHeaderHidden(True)
        self.raw_tree.setAlternatingRowColors(True)
        self.raw_tree.setStyleSheet(tree_css)
        
      

        self.data_tabs.addTab(self.merged_tree, "Well Data")
        self.data_tabs.addTab(self.raw_tree, "Raw / Loaded Data")

        self.data_group_layout.addWidget(self.data_tabs)
        self.left_top_layout.addWidget(self.data_group)

        # -------- Left lower: tops panel ----------
        self.left_bottom_container = QWidget()
        self.left_bottom_layout = QVBoxLayout(self.left_bottom_container)
        self.left_bottom_layout.setContentsMargins(0, 0, 0, 0)
        self.left_bottom_layout.setSpacing(4)
        self.left_bottom_container.setMinimumHeight(250)

        self.tops_group = QGroupBox("Tops / Zone")
        self.tops_group_layout = QVBoxLayout(self.tops_group)
        self.tops_group_layout.setContentsMargins(6, 6, 6, 6)
        self.tops_group_layout.setSpacing(4)

        self.left_bottom_layout.addWidget(self.tops_group)

        self.left_splitter.addWidget(self.left_top_container)
        self.left_splitter.addWidget(self.left_bottom_container)
        self.left_splitter.setStretchFactor(0, 3)
        self.left_splitter.setStretchFactor(1, 3)
        self.left_splitter.setSizes([320, 360])

        # ==============================================================
        # CENTER PANE
        # ==============================================================
        self.stack = QStackedWidget()


        workflow_font = QFont()
        workflow_font.setPointSize(10)   # start here (try 11 if needed)
        workflow_font.setBold(False)
        
        self.stack.setFont(workflow_font)
        
        
        self.stack.setStyleSheet("""
            QLabel {
                font-weight: normal;
            }
        """)
        # ==============================================================
        # RIGHT PANE
        # ==============================================================
        self.right_container = QWidget()
        self.right_layout = QVBoxLayout(self.right_container)
        self.right_layout.setContentsMargins(0, 0, 0, 0)
        self.right_layout.setSpacing(4)

        self.workflow_group = QGroupBox("Workflow")
        self.workflow_layout = QVBoxLayout(self.workflow_group)
        self.workflow_layout.setContentsMargins(6, 6, 6, 6)
        self.workflow_layout.setSpacing(6)

        self.workflow_buttons_host = QWidget()
        self.workflow_buttons_layout = QVBoxLayout(self.workflow_buttons_host)
        self.workflow_buttons_layout.setContentsMargins(0, 0, 0, 0)
        self.workflow_buttons_layout.setSpacing(8)

        self.workflow_layout.addWidget(self.workflow_buttons_host)
        self.right_layout.addWidget(self.workflow_group)

        # ==============================================================
        # MAIN SPLITTER
        # ==============================================================
        self.main_splitter.addWidget(self.left_splitter)
        self.main_splitter.addWidget(self.stack)
        self.main_splitter.addWidget(self.right_container)

        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setStretchFactor(2, 0)
        self.main_splitter.setSizes([180, 1000, 260])

        outer_layout.addWidget(self.main_splitter)

    # ------------------------------------------------------------------
    # Panels
    # ------------------------------------------------------------------
    def _build_panels(self):
        # Left lower permanent tops panel
        self.tops_panel = TopsPanel(self.controller)
        self.tops_group_layout.addWidget(self.tops_panel)
    
        # Center stacked panels
        self.home_panel = WorkflowHomePanel(self.controller)
        self.open_project_panel = OpenProjectPanel(self.controller)
        self.load_data_panel = LoadDataPanel(self.controller)
        self.initial_plot_panel = PlotsPanelInitial(self.controller)
        self.phit_panel = PhitPanel(self.controller)
        self.vsh_panel = VshPanel(self.controller)
        self.sw_panel = SwPanel(self.controller)
        self.summary_panel = WellSummaryPanel(self.controller)
        self.vsh_workspace_panel = PlotsPanelVsh(self.controller)
        self.ws_workspace_panel = PlotsPanelWS(self.controller)
        self.final_plot_panel = PlotsPanelFinal(self.controller)
        self.save_well_panel = SaveWellPanel(self.controller)

    
        ordered_panels = [
            ("Home", self.home_panel),
            ("Open Project", self.open_project_panel),
            ("Load Data", self.load_data_panel),
            ("Initial Plot", self.initial_plot_panel),
            ("Calc PHIT", self.phit_panel),
            ("Calc Vsh", self.vsh_panel),
            ("Calc Sw", self.sw_panel),
            ("Well Summary", self.summary_panel),
            ("Vsh HL", self.vsh_workspace_panel),
            ("Sw WS", self.ws_workspace_panel),
            ("Final Depth Plot", self.final_plot_panel),
            ("Save Well", self.save_well_panel),
        ]
    
        for name, panel in ordered_panels:
            self.stack.addWidget(panel)
            self.panel_map[name] = panel
    

    # ------------------------------------------------------------------
    # Right-side workflow navigation
    # ------------------------------------------------------------------
    def _populate_navigation(self):
        self.nav_button_map.clear()

        while self.workflow_buttons_layout.count():
            item = self.workflow_buttons_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        group_font = QFont()
        group_font.setPointSize(12)  
        group_font.setBold(True)
        #self.stack.setFont(group_font)
        #group_font.setPointSize(group_font.pointSize() + 1)

        for group_name, items in NAV_GROUPS.items():
            group_box = QGroupBox(group_name)
            group_box.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

            group_layout = QVBoxLayout(group_box)
            group_layout.setContentsMargins(6, 8, 6, 8)
            group_layout.setSpacing(4)

            group_box.setFont(group_font)

            for label, _icon_key in items:
                btn = QToolButton()
                btn.setText(label)
                btn.setIcon(self._icon_for_name(label))
                btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
                btn.setIconSize(QSize(24, 24))
                btn.setMinimumHeight(34)
                btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
                btn.setCheckable(True)
                btn.setAutoExclusive(False)

                btn.setStyleSheet(
                    """
                    QToolButton {
                        text-align: left;
                        padding: 6px 8px;
                        border-radius: 6px;
                    }
                    QToolButton:hover {
                        background: #eef2ff;
                    }
                    QToolButton:checked {
                        background: #2563eb;
                        color: white;
                        font-weight: bold;
                    }
                    """
                )

                btn.clicked.connect(lambda checked=False, name=label: self._select_panel(name))

                group_layout.addWidget(btn)
                self.nav_button_map[label] = btn

            self.workflow_buttons_layout.addWidget(group_box)

        self.workflow_buttons_layout.addStretch(1)

    # ------------------------------------------------------------------
    # Panel switching
    # ------------------------------------------------------------------
    def _select_panel(self, name: str):
        panel = self.panel_map.get(name)
        if panel is None:
            return

        self.stack.setCurrentWidget(panel)

        for panel_name, btn in self.nav_button_map.items():
            btn.blockSignals(True)
            btn.setChecked(panel_name == name)
            btn.blockSignals(False)

        if hasattr(panel, "refresh"):
            try:
                panel.refresh()
            except Exception as e:
                print(f"[UI] refresh failed for {name}: {e}")

        if hasattr(panel, "update_all"):
            try:
                panel.update_all(self.state)
            except Exception as e:
                print(f"[UI] update_all failed for {name}: {e}")

    # ------------------------------------------------------------------
    # Left-side data trees
    # ------------------------------------------------------------------
    def refresh_data_views(self):
        self._refresh_raw_tree()
        self._refresh_merged_tree()

    def _refresh_raw_tree(self):
        self.raw_tree.clear()

        state = self.state
        runs = getattr(state, "las_runs", None) or getattr(state, "raw_runs", None) or []

        if not runs:
            self.raw_tree.addTopLevelItem(QTreeWidgetItem(["No raw/loaded well data yet"]))
            self.raw_tree.expandAll()
            return

        well_name = self._state_well_name(default="Loaded Well Data")
        root = QTreeWidgetItem([well_name])
        self.raw_tree.addTopLevelItem(root)

        for i, run in enumerate(runs, start=1):
            run_name = getattr(run, "name", None) or getattr(run, "well_name", None) or f"Run {i}"
            run_item = QTreeWidgetItem([str(run_name)])
            root.addChild(run_item)

            df = getattr(run, "df", None)
            if df is None:
                df = getattr(run, "dataframe", None)

            cols = list(df.columns) if df is not None and hasattr(df, "columns") else []
            sets_map = self._classify_columns_into_sets(cols)
            self._populate_set_tree(run_item, sets_map)

        self.raw_tree.expandAll()

    def _refresh_merged_tree(self):
        self.merged_tree.clear()

        state = self.state
        merged_df = getattr(state, "merged_df", None)
        if merged_df is None:
            merged_df = getattr(state, "analysis_df", None)
        if merged_df is None:
            merged_df = getattr(state, "working_df", None)

        if merged_df is None or not hasattr(merged_df, "columns"):
            self.merged_tree.addTopLevelItem(QTreeWidgetItem(["No merged data yet"]))
            self.merged_tree.expandAll()
            return

        root = QTreeWidgetItem([self._state_well_name(default="Well")])
        self.merged_tree.addTopLevelItem(root)

        cols = list(merged_df.columns)
        sets_map = self._classify_columns_into_sets(cols)
        self._populate_set_tree(root, sets_map)

        self.merged_tree.expandAll()

    def _populate_set_tree(self, parent_item: QTreeWidgetItem, sets_map: dict[str, list[str]]):
        for set_name in sorted(sets_map.keys()):
            set_item = QTreeWidgetItem([set_name])
            parent_item.addChild(set_item)

            for curve in sorted(sets_map[set_name]):
                set_item.addChild(QTreeWidgetItem([curve]))

    def _classify_columns_into_sets(self, cols: list[str]) -> dict[str, list[str]]:
        """
        Classify columns using the YAML curve-set configuration and store winners.
        """
        try:
            yaml_path = Path(__file__).resolve().parents[2] / "petrocore" / "config" / "curve_sets.yaml"
            config = load_curve_set_config(yaml_path)

            sets_map, winners_map = classify_curves_with_winners(cols, config)
            self._yaml_winners_map = winners_map or {}

            if not sets_map:
                self._yaml_winners_map = {}
                return {"MISC": list(cols)}

            clean_map: dict[str, list[str]] = {}
            assigned = set()

            for set_name, curve_list in sets_map.items():
                if not curve_list:
                    continue
                clean_map[set_name] = list(curve_list)
                assigned.update(curve_list)

            misc = [c for c in cols if c not in assigned]
            if misc:
                clean_map.setdefault("MISC", []).extend(misc)

            return clean_map

        except Exception as e:
            print(f"[UI] YAML set classification failed: {e}")
            self._yaml_winners_map = {}
            return {"MISC": list(cols)}

    def _state_well_name(self, default: str = "Well") -> str:
        state = self.controller.get_state()
        for attr in ("well_name", "well", "uwi", "name"):
            value = getattr(state, attr, None)
            if value:
                return str(value)
        return default

    # ------------------------------------------------------------------
    # Icons
    # ------------------------------------------------------------------
    def _icon_for_name(self, name: str) -> QIcon:
        """
        Looks in:
            apps/merge_gui/assets/icons/
            apps/merge_gui/icons/
            apps/merge_gui/resources/icons/
            ./icons/
        Falls back to Qt standard icons.
        """
        key = LABEL_TO_ICON_KEY.get(name)

        here = Path(__file__).resolve().parent
        candidate_dirs = [
            here.parent / "assets" / "icons",
            here.parent / "icons",
            here.parent / "resources" / "icons",
            Path.cwd() / "icons",
        ]

        if key:
            for folder in candidate_dirs:
                for rel_path in ICON_MAP.get(key, []):
                    p = folder / rel_path
                    if p.exists():
                        return QIcon(str(p))

        style = self.style()
        fallback_map = {
            "Home": QStyle.SP_DirHomeIcon,
            "Load Data": QStyle.SP_DialogOpenButton,
            "Initial Plot": QStyle.SP_FileDialogDetailedView,
            "Calc PHIT": QStyle.SP_ComputerIcon,
            "Calc Vsh": QStyle.SP_ArrowRight,
            "Calc Sw": QStyle.SP_DriveNetIcon,
            "Well Summary": QStyle.SP_FileDialogInfoView,
            "Vsh HL": QStyle.SP_DesktopIcon,
            "Sw WS": QStyle.SP_DesktopIcon,
            "Final Depth Plot": QStyle.SP_FileDialogContentsView,
        }
        return style.standardIcon(fallback_map.get(name, QStyle.SP_FileIcon))

    # ------------------------------------------------------------------
    # Refresh hooks used by controller
    # ------------------------------------------------------------------
    def refresh_plots(self):
        self.refresh_all_panels()

    def refresh_all_panels(self):
        self.refresh_data_views()

        if hasattr(self.tops_panel, "refresh"):
            try:
                self.tops_panel.refresh()
            except Exception as e:
                print(f"[UI] tops refresh failed: {e}")

        for name, panel in self.panel_map.items():
            if hasattr(panel, "refresh"):
                try:
                    panel.refresh()
                except Exception as e:
                    print(f"[UI] refresh failed for {name}: {e}")

            if hasattr(panel, "update_all"):
                try:
                    panel.update_all(self.state)
                except Exception as e:
                    print(f"[UI] update_all failed for {name}: {e}")
