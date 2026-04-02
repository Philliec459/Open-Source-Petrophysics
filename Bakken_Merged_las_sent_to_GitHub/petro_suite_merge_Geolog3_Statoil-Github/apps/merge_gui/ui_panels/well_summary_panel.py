from __future__ import annotations

import pandas as pd

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QTextEdit,
    QGroupBox,
)


class WellSummaryPanel(QWidget):
    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self._build_ui()

    # ---------------------------------------------------------
    # UI
    # ---------------------------------------------------------
    def _build_ui(self):
        layout = QVBoxLayout(self)

        title = QLabel("Well Summary")
        layout.addWidget(title)

        # ---- Well Info ----
        self.info_box = QTextEdit()
        self.info_box.setReadOnly(True)
        self.info_box.setMinimumHeight(120)

        box1 = QGroupBox("Well Info")
        b1_layout = QVBoxLayout(box1)
        b1_layout.addWidget(self.info_box)

        # ---- Working Curves ----
        self.curve_box = QTextEdit()
        self.curve_box.setReadOnly(True)
        self.curve_box.setMinimumHeight(140)

        box2 = QGroupBox("Selected Working Curves")
        b2_layout = QVBoxLayout(box2)
        b2_layout.addWidget(self.curve_box)

        # ---- Derived Curves ----
        self.derived_box = QTextEdit()
        self.derived_box.setReadOnly(True)
        self.derived_box.setMinimumHeight(140)

        box3 = QGroupBox("Derived Curves in analysis_df")
        b3_layout = QVBoxLayout(box3)
        b3_layout.addWidget(self.derived_box)

        layout.addWidget(box1)
        layout.addWidget(box2)
        layout.addWidget(box3)

        layout.addStretch()

    # ---------------------------------------------------------
    # Refresh
    # ---------------------------------------------------------
    def refresh(self):
        state = self.controller.get_state()
        df = getattr(state, "analysis_df", None)
        params = getattr(state, "params", {})

        # ---- Well Info ----
        if df is None or df.empty:
            self.info_box.setPlainText("No well loaded.")
            self.curve_box.setPlainText("")
            self.derived_box.setPlainText("")
            return

        nrows = len(df)

        # Depth handling
        depth = None
        if "DEPT" in df.columns:
            depth = pd.to_numeric(df["DEPT"], errors="coerce")
        else:
            try:
                depth = pd.to_numeric(pd.Index(df.index), errors="coerce")
            except Exception:
                depth = None

        if depth is not None and depth.notna().any():
            dmin = float(depth.min())
            dmax = float(depth.max())
            depth_str = f"{dmin:.2f} → {dmax:.2f}"
        else:
            depth_str = "Unknown"

        info_lines = [
            f"Rows: {nrows}",
            f"Depth range: {depth_str}",
            f"Columns: {len(df.columns)}",
        ]

        self.info_box.setPlainText("\n".join(info_lines))

        # ---- Working Curves ----
        keys = [
            "gr_curve",
            "cgr_curve",
            "rhob_curve",
            "tnph_curve",
            "dtco_curve",
            "rt_curve",
            "phit_curve",
            "vsh_curve",
        ]

        curve_lines = []
        for k in keys:
            val = params.get(k)
            curve_lines.append(f"{k:15s} = {val}")

        self.curve_box.setPlainText("\n".join(curve_lines))

        # ---- Derived Curves ----
        derived_keywords = [
            "PHIT",
            "POR",
            "VSH",
            "SW",
            "BVW",
            "BVO",
            "PHIE",
            "MSTAR",
        ]

        derived = []
        for c in df.columns:
            cu = c.upper()
            if any(k in cu for k in derived_keywords):
                derived.append(c)

        derived.sort()

        if not derived:
            self.derived_box.setPlainText("No derived curves yet.")
        else:
            self.derived_box.setPlainText("\n".join(derived))