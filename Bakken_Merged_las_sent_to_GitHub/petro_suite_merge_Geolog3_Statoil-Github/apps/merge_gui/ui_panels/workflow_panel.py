# apps/merge_gui/ui_panels/workflow_panel.py
from __future__ import annotations

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel

from apps.merge_gui.widgets.vsh_controls import VshControlsWidget
from apps.merge_gui.widgets.sw_controls import SwControlsWidget


class WorkflowPanel(QWidget):
    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Vsh Controls"))
        layout.addWidget(VshControlsWidget(controller=self.controller))

        layout.addWidget(QLabel("Sw Controls"))
        layout.addWidget(SwControlsWidget(controller=self.controller))

        layout.addStretch(1)

    def refresh(self):
        pass