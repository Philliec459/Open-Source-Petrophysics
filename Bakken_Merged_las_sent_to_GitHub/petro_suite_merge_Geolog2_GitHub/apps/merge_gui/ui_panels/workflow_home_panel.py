from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QFrame,
    QHBoxLayout,
    QPushButton,
)


class WorkflowHomePanel(QWidget):
    """
    Home panel for petro_suite_merge.

    This is intentionally lightweight and merge-focused.
    It provides orientation, not computation.
    """

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        # ------------------------------------------------------------------
        # Title
        # ------------------------------------------------------------------
        title = QLabel("PetroSuite Merge")
        title.setStyleSheet("font-size: 26px; font-weight: bold;")
        layout.addWidget(title)

        subtitle = QLabel(
            "Load multiple LAS runs, organize curves into families, inspect depth alignment, "
            "and prepare a clean merged dataset."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("font-size: 14px; color: #444;")
        layout.addWidget(subtitle)

        # ------------------------------------------------------------------
        # Workflow box
        # ------------------------------------------------------------------
        workflow_box = QFrame()
        workflow_box.setFrameShape(QFrame.StyledPanel)
        workflow_box.setStyleSheet("""
            QFrame {
                background: #f7f7f7;
                border: 1px solid #d9d9d9;
                border-radius: 8px;
            }
        """)
        workflow_layout = QVBoxLayout(workflow_box)
        workflow_layout.setContentsMargins(16, 16, 16, 16)
        workflow_layout.setSpacing(10)

        workflow_title = QLabel("Suggested Workflow")
        workflow_title.setStyleSheet("font-size: 18px; font-weight: bold;")
        workflow_layout.addWidget(workflow_title)

        steps = [
            "1. Load one or more LAS files in the Load panel.",
            "2. Review available runs and curves.",
            "3. Organize curves by family in Curve Selection.",
            "4. Plot key curves to inspect overlap and depth agreement.",
            "5. Use the merged result as the basis for later petrophysical workflows.",
        ]

        for step in steps:
            lbl = QLabel(step)
            lbl.setWordWrap(True)
            lbl.setStyleSheet("font-size: 13px;")
            workflow_layout.addWidget(lbl)

        layout.addWidget(workflow_box)

        # ------------------------------------------------------------------
        # Current session summary
        # ------------------------------------------------------------------
        self.summary_box = QFrame()
        self.summary_box.setFrameShape(QFrame.StyledPanel)
        self.summary_box.setStyleSheet("""
            QFrame {
                background: white;
                border: 1px solid #d9d9d9;
                border-radius: 8px;
            }
        """)
        summary_layout = QVBoxLayout(self.summary_box)
        summary_layout.setContentsMargins(16, 16, 16, 16)
        summary_layout.setSpacing(8)

        summary_title = QLabel("Current Session")
        summary_title.setStyleSheet("font-size: 18px; font-weight: bold;")
        summary_layout.addWidget(summary_title)

        self.lbl_runs = QLabel("Loaded LAS runs: 0")
        self.lbl_runs.setStyleSheet("font-size: 13px;")
        summary_layout.addWidget(self.lbl_runs)

        self.lbl_analysis = QLabel("Active analysis dataframe: not set")
        self.lbl_analysis.setStyleSheet("font-size: 13px;")
        summary_layout.addWidget(self.lbl_analysis)

        self.lbl_tops = QLabel("Tops loaded: no")
        self.lbl_tops.setStyleSheet("font-size: 13px;")
        summary_layout.addWidget(self.lbl_tops)

        layout.addWidget(self.summary_box)

        # ------------------------------------------------------------------
        # Navigation hints
        # ------------------------------------------------------------------
        nav_box = QFrame()
        nav_box.setFrameShape(QFrame.StyledPanel)
        nav_box.setStyleSheet("""
            QFrame {
                background: #fcfcfc;
                border: 1px solid #d9d9d9;
                border-radius: 8px;
            }
        """)
        nav_layout = QVBoxLayout(nav_box)
        nav_layout.setContentsMargins(16, 16, 16, 16)
        nav_layout.setSpacing(8)

        nav_title = QLabel("Where to go next")
        nav_title.setStyleSheet("font-size: 18px; font-weight: bold;")
        nav_layout.addWidget(nav_title)

        nav_text = QLabel(
            "Use the navigation tree on the right to move through the Petrophysical workflow:\n\n"
            "• Load LAS Files\n"
            "• Curve Selection\n"
            "• Petrophysical Calculations and Depth Plots"
        )
        nav_text.setWordWrap(True)
        nav_text.setStyleSheet("font-size: 13px;")
        nav_layout.addWidget(nav_text)

        layout.addWidget(nav_box)

        # ------------------------------------------------------------------
        # Bottom row
        # ------------------------------------------------------------------
        bottom_row = QHBoxLayout()
        bottom_row.addStretch()

        self.btn_refresh = QPushButton("Refresh Summary")
        self.btn_refresh.clicked.connect(self.refresh)
        bottom_row.addWidget(self.btn_refresh)

        layout.addLayout(bottom_row)
        layout.addStretch()

    def _state(self):
        return self.controller.get_state()

    def refresh(self):
        state = self._state()

        las_runs = getattr(state, "las_runs", [])
        analysis_df = getattr(state, "analysis_df", None)
        tops_df = getattr(state, "tops_df", None)

        n_runs = len(las_runs) if isinstance(las_runs, list) else 0
        self.lbl_runs.setText(f"Loaded LAS runs: {n_runs}")

        if analysis_df is None:
            self.lbl_analysis.setText("Active analysis dataframe: not set")
        else:
            try:
                rows, cols = analysis_df.shape
                self.lbl_analysis.setText(
                    f"Active analysis dataframe: {rows:,} rows × {cols} columns"
                )
            except Exception:
                self.lbl_analysis.setText("Active analysis dataframe: set")

        tops_loaded = tops_df is not None
        self.lbl_tops.setText(f"Tops loaded: {'yes' if tops_loaded else 'no'}")