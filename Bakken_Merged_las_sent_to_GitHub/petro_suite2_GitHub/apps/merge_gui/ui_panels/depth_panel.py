# apps/merge_gui/ui_panels/depth_panel.py


from __future__ import annotations

print(">>> LOADING depth_panel.py from:", __file__)


from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QDoubleSpinBox, QPushButton
)

class DepthRangePanel(QWidget):
    """
    Manual depth interval selector.
    - Top/Base spin boxes (ft)
    - Apply button calls controller.set_depth_range(top, base)
    - Clear button resets analysis_df to full dataset
    """

    def __init__(self, controller):
        super().__init__()
        self.controller = controller

        self.lab_limits = QLabel("Depth limits: (no dataset)")
        self.lab_limits.setWordWrap(True)

        self.top_spin = QDoubleSpinBox()
        self.top_spin.setDecimals(2)
        self.top_spin.setRange(-1e9, 1e9)
        self.top_spin.setSingleStep(1.0)
        self.top_spin.setKeyboardTracking(False)

        self.base_spin = QDoubleSpinBox()
        self.base_spin.setDecimals(2)
        self.base_spin.setRange(-1e9, 1e9)
        self.base_spin.setSingleStep(1.0)
        self.base_spin.setKeyboardTracking(False)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Top (ft)"))
        row1.addWidget(self.top_spin)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Base (ft)"))
        row2.addWidget(self.base_spin)

        self.btn_apply = QPushButton("Apply Interval")
        self.btn_clear = QPushButton("Clear Interval")

        self.btn_apply.clicked.connect(self._apply)
        self.btn_clear.clicked.connect(self._clear)

        # Optional: apply when user finishes editing spinboxes
        self.top_spin.editingFinished.connect(self._apply)
        self.base_spin.editingFinished.connect(self._apply)

        layout = QVBoxLayout(self)
        layout.addWidget(self.lab_limits)
        layout.addLayout(row1)
        layout.addLayout(row2)
        layout.addWidget(self.btn_apply)
        layout.addWidget(self.btn_clear)
        layout.addStretch(1)

    def set_depth_limits(self, dmin: float, dmax: float):
        """Called after dataset load."""
        self.lab_limits.setText(f"Depth limits: {dmin:.2f} – {dmax:.2f} ft")

        # set sensible ranges
        self.top_spin.setRange(dmin, dmax)
        self.base_spin.setRange(dmin, dmax)

        # initialize values
        self.top_spin.setValue(dmin)
        self.base_spin.setValue(dmax)

    def _apply(self):
        t = float(self.top_spin.value())
        b = float(self.base_spin.value())
        # Calls your controller method (must exist)
        self.controller.set_depth_range(t, b)

    def _clear(self):
        # reset to full dataset
        ds = getattr(self.controller.state, "dataset", None)
        if ds is not None:
            self.controller.state.depth_top = None
            self.controller.state.depth_base = None
            self.controller.state.analysis_df = ds.data
            self.controller.state.analysis_df_view = ds.data
            self.controller.refresh_plots()
