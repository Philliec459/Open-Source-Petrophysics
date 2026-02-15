# merge_gui/ui/main_window.py

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QMainWindow, QDockWidget, QTabWidget, QTextEdit,
    QWidget, QVBoxLayout, QLabel
)

from petrocore.viz.log_canvas_pg import LogCanvasPG
from petrocore.models.dataset import Dataset

from petrocore.workflow.state import WorkflowState
from petrocore.workflow.pipeline import WorkflowRegistry

from apps.merge_gui.controllers.workflow_controller import WorkflowController
from apps.merge_gui.ui_panels.plots_panel import PlotsPanel


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Merge/QC + Petro Workflow")
        self.resize(1600, 900)

        # -------------------------
        # A) CENTRAL UI: tabs
        # -------------------------
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Tab 1: Log canvas
        self.log_canvas = LogCanvasPG()
        self.tabs.addTab(self.log_canvas, "Log Canvas")

        # Tab 2: Workflow plots (your new workflow UI lives here)
        self.state = WorkflowState()
        self.plots_panel = PlotsPanel()
        self.tabs.addTab(self.plots_panel, "Workflow Plots")

        # Placeholder tabs (keep for now)
        self.tabs.addTab(self._placeholder("QC"), "QC")
        self.tabs.addTab(self._placeholder("Alignment Score"), "Alignment Score")

        # -------------------------
        # B) CONTROLLER + LOG
        # -------------------------
        self.console = QTextEdit()
        self.console.setReadOnly(True)

        self.controller = WorkflowController(
            state=self.state,
            plots_panel=self.plots_panel,
            log_console=self.console
        )

        # Configure workflow steps / defaults
        self._configure_workflow()

        # -------------------------
        # C) DOCKS (existing merge/qc layout)
        # -------------------------
        self.addDockWidget(Qt.LeftDockWidgetArea,  self._dock("Project",         self._placeholder("Project Tree")))
        self.addDockWidget(Qt.LeftDockWidgetArea,  self._dock("Curve Inventory", self._placeholder("Curve list + families")))
        self.addDockWidget(Qt.RightDockWidgetArea, self._dock("Alignment",       self._placeholder("Bulk shift + anchors")))
        self.addDockWidget(Qt.RightDockWidgetArea, self._dock("Merge",           self._placeholder("Merge params + preview")))
        self.addDockWidget(Qt.RightDockWidgetArea, self._dock("Export",          self._placeholder("Export options")))

        dock_log = QDockWidget("Workflow Log", self)
        dock_log.setWidget(self.console)
        self.addDockWidget(Qt.BottomDockWidgetArea, dock_log)

    # -------------------------
    # UI helpers
    # -------------------------
    def _dock(self, title: str, widget: QWidget) -> QDockWidget:
        d = QDockWidget(title, self)
        d.setWidget(widget)
        d.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        return d

    def _placeholder(self, text: str) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(QLabel(text))
        lay.addStretch(1)
        return w

    # -------------------------
    # Workflow config
    # -------------------------
    def _configure_workflow(self):
        self.state.registry = WorkflowRegistry()

        from petrocore.workflow.steps_petrophysics import (
            step_vsh_hl,
            step_cbw,
            step_waxman_smits,
            step_lith_opt,
        )

        self.state.registry.add("vsh_hl", "Hodges–Lehmann Vsh", step_vsh_hl, enabled_by_default=True)
        self.state.registry.add("cbw",    "Clay Bound Water + PHIE", step_cbw, enabled_by_default=True)
        self.state.registry.add("ws",     "Waxman–Smits Sw + BVW", step_waxman_smits, enabled_by_default=True)
        self.state.registry.add("lith",   "Lithology Optimization", step_lith_opt, enabled_by_default=False)

        self.state.enabled_steps = {s.key: s.enabled_by_default for s in self.state.registry.steps}

        self.state.params.update({
            "gr_curve": "GR",
            "rhob_curve": "RHOZ",
            "tnph_curve": "TNPH",
            "rt_curve": "RT",
            "Rw": 0.08,
            "m": 2.0,
            "n": 2.0,
            "B": 4.5,
        })

    # -------------------------
    # Quick test hook
    # -------------------------
    def demo_load(self, ds: Dataset):
        # 1) Log canvas
        self.log_canvas.set_dataset(ds)
        self.log_canvas.set_tracks(self.log_canvas.standard_4track_template(ds))

        # 2) Workflow side
        self.controller.set_dataset(ds)   # IMPORTANT: controller expects Dataset (not df)
        self.console.append(f"Demo loaded dataset: {ds.name}")
