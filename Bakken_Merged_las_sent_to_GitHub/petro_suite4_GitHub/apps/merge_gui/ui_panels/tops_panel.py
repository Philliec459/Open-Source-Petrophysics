# apps/merge_gui/ui_panels/tops_panel.py
from __future__ import annotations

import pandas as pd

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTableWidget, QTableWidgetItem,
    QMessageBox, QAbstractItemView
)





class TopsPanel(QWidget):
    """
    Displays tops from controller.state.tops_df and lets user apply a zone.

    Expected state.tops_df schemas:
      A) Preferred: columns ['Top', 'TopDepth', 'BaseDepth']
      B) Fallback:  columns ['Top', 'Depth']   (BaseDepth inferred from next row)
    """

    def __init__(self, controller):
        super().__init__()
        self.controller = controller

        self.info = QLabel("No tops loaded.")
        self.info.setWordWrap(True)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Formation", "Top (ft)", "Base (ft)"])

        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)  # <-- multi-select

        self.table.setSortingEnabled(False)

        self.btn_refresh = QPushButton("LoadTops")
        self.btn_apply = QPushButton("Apply Zone")
        self.btn_clear = QPushButton("Clear Zone")

        self.btn_refresh.clicked.connect(self.refresh_from_state)
        self.btn_apply.clicked.connect(self.apply_selected_zone)
        self.btn_clear.clicked.connect(self.clear_zone)

        self.info.setText("Select tops (Shift/Cmd for range). Apply Zone to set interval.")

        row = QHBoxLayout()
        row.addWidget(self.btn_refresh)
        row.addWidget(self.btn_apply)
        row.addWidget(self.btn_clear)

        layout = QVBoxLayout(self)
        layout.addWidget(self.info)
        layout.addLayout(row)
        layout.addWidget(self.table)

        # Try initial fill
        self.refresh_from_state()

    # -------------------------
    # Data ingestion
    # -------------------------
    def refresh_from_state(self):
        df = getattr(self.controller.state, "tops_df", None)
        if df is None or len(df) == 0:
            self.info.setText("No tops loaded. Use File → Open Tops (Core)…")
            self._set_rows([])
            return

        # Normalize into (formation, top, base)
        rows = self._normalize_tops_df(df)
        self._set_rows(rows)
        self.info.setText(f"Tops loaded: {len(rows)}")

    def _normalize_tops_df(self, df: pd.DataFrame):
        cols = set(df.columns)

        # Preferred
        if {"Top", "TopDepth", "BaseDepth"}.issubset(cols):
            out = []
            for _, r in df.iterrows():
                top = r.get("TopDepth")
                base = r.get("BaseDepth")
                name = r.get("Top", "")
                if pd.isna(top) or pd.isna(base):
                    continue
                out.append((str(name), float(top), float(base)))
            return out

        # Fallback: Top + Depth, infer base from next depth
        if {"Top", "Depth"}.issubset(cols):
            dff = df.copy()
            dff["Depth"] = pd.to_numeric(dff["Depth"], errors="coerce")
            dff = dff.dropna(subset=["Depth"]).sort_values("Depth").reset_index(drop=True)

            out = []
            for i in range(len(dff)):
                name = dff.loc[i, "Top"]
                top = float(dff.loc[i, "Depth"])
                # base = next top depth, if exists; else None
                if i < len(dff) - 1:
                    base = float(dff.loc[i + 1, "Depth"])
                else:
                    base = top
                out.append((str(name), top, base))
            return out

        # Unknown schema
        self.info.setText(f"tops_df has unexpected columns: {list(df.columns)}")
        return []

    def _set_rows(self, rows):
        self.table.setRowCount(0)
        for formation, top, base in rows:
            r = self.table.rowCount()
            self.table.insertRow(r)

            self.table.setItem(r, 0, QTableWidgetItem(formation))
            self.table.setItem(r, 1, QTableWidgetItem(f"{top:.2f}"))
            self.table.setItem(r, 2, QTableWidgetItem(f"{base:.2f}"))

        self.table.resizeColumnsToContents()

    # -------------------------
    # Actions
    # -------------------------


    def apply_selected_zone(self):
        # selectedRows() returns QModelIndex list (one per selected row)
        sel = self.table.selectionModel().selectedRows()
        rows = sorted({ix.row() for ix in sel})
    
        if not rows:
            QMessageBox.information(self, "Apply Zone", "Select one or more tops first.")
            return
    
        # Read depths from the table (Top, Base columns already displayed)
        tops = []
        bases = []
        for r in rows:
            try:
                top = float(self.table.item(r, 1).text())
                base = float(self.table.item(r, 2).text())
            except Exception:
                continue
            tops.append(top)
            bases.append(base)
    
        if not tops:
            QMessageBox.warning(self, "Apply Zone", "Could not parse selected depths.")
            return
    
        # Range spanning selection
        top_depth = min(tops)
        base_depth = max(bases)
    
        self.controller.set_depth_range(top_depth, base_depth)





    def clear_zone(self):
        # Clear depth_top/depth_base and reset analysis_df to full dataset
        self.controller.state.depth_top = None
        self.controller.state.depth_base = None

        ds = getattr(self.controller.state, "dataset", None)
        if ds is not None:
            self.controller.state.analysis_df = ds.data
            self.controller.state.analysis_df_view = ds.data

        self.controller.refresh_plots()
