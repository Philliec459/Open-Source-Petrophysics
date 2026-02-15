# apps/merge_gui/ui_main_window.py

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from PySide6.QtWidgets import (
    QMainWindow, QDockWidget, QTabWidget, QWidget, QVBoxLayout, QLabel,
    QFileDialog, QTreeWidget, QTreeWidgetItem, QComboBox, QHBoxLayout,
    QSlider, QPushButton, QMessageBox, QTextEdit,
    QDoubleSpinBox, QCheckBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction

import lasio

from petrocore.viz.log_canvas_pg import LogCanvasPG
from petrocore.models.dataset import Dataset
from petrocore.alignment.window_shift import windowed_bulk_shifts


# -----------------------------
# Simple run container
# -----------------------------
@dataclass
class LasRun:
    name: str
    path: str
    df: pd.DataFrame  # index = depth (ft)


def _las_to_df(path: str) -> pd.DataFrame:
    las = lasio.read(path)
    df = las.df()  # index is depth already

    # Ensure float depth ascending
    df.index = pd.to_numeric(df.index, errors="coerce").astype(float)
    df = df[~df.index.isna()].sort_index()

    # Ensure numeric columns
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Merge/QC")
        self.resize(1700, 950)

        # -----------------------------
        # State
        # -----------------------------
        self.runs: List[LasRun] = []
        self.base_idx: int = 0
        self.mov_idx: int = 1

        # bulk shift (ft)
        self.shift_ft: float = 0.0

        # slider scaling
        self.shift_step_ft: float = 0.25   # resolution
        self.shift_range_ft: float = 50.0  # +/- range

        # window shift results
        self.window_shifts_df: Optional[pd.DataFrame] = None

        # Curve-family preference for shifting (GR)
        self.gr_candidates = ["GR_EDTC","HSGR", "GR", "SGR", "HGR", "ECGR"]

        # -----------------------------
        # Menu
        # -----------------------------
        file_menu = self.menuBar().addMenu("File")

        open_las_act = QAction("Open LAS Runs…", self)
        open_las_act.triggered.connect(self.open_las_runs)
        file_menu.addAction(open_las_act)

        open_parquet_act = QAction("Open Parquet…", self)
        open_parquet_act.triggered.connect(self.open_parquet)
        file_menu.addAction(open_parquet_act)

        # -----------------------------
        # Central tabs
        # -----------------------------
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.log_canvas = LogCanvasPG()
        self.tabs.addTab(self.log_canvas, "Log Canvas")
        self.tabs.addTab(self._placeholder("QC"), "QC")
        self.tabs.addTab(self._placeholder("Alignment Score"), "Alignment Score")

        # -----------------------------
        # Docks
        # -----------------------------
        self.project_tree = QTreeWidget()
        self.project_tree.setHeaderLabels(["Runs / Curves"])
        self.addDockWidget(Qt.LeftDockWidgetArea, self._dock("Project", self.project_tree))

        self.addDockWidget(
            Qt.LeftDockWidgetArea,
            self._dock("Curve Inventory", self._placeholder("Later: families + missing curves"))
        )

        self.align_panel = self._build_alignment_panel()
        self.addDockWidget(Qt.RightDockWidgetArea, self._dock("Alignment", self.align_panel))

        self.addDockWidget(Qt.RightDockWidgetArea, self._dock("Merge", self._placeholder("Merge params + preview toggles")))
        self.addDockWidget(Qt.RightDockWidgetArea, self._dock("Export", self._placeholder("Export options")))

        # Start with empty project tree
        self._refresh_project_tree()

    # -----------------------------
    # Alignment UI
    # -----------------------------
    def _build_alignment_panel(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)

        # Base / moving run selectors
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Base run:"))
        self.base_combo = QComboBox()
        self.base_combo.currentIndexChanged.connect(self._on_base_changed)
        row1.addWidget(self.base_combo)
        lay.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Moving run:"))
        self.mov_combo = QComboBox()
        self.mov_combo.currentIndexChanged.connect(self._on_mov_changed)
        row2.addWidget(self.mov_combo)
        lay.addLayout(row2)

        # Curve selector
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Shift curve:"))
        self.curve_combo = QComboBox()
        self.curve_combo.currentIndexChanged.connect(lambda _: self.update_overlay())
        row3.addWidget(self.curve_combo)
        lay.addLayout(row3)

        # -----------------------------
        # SHIFT WINDOW (window top + len)
        # -----------------------------
        rowW = QHBoxLayout()
        rowW.addWidget(QLabel("Window top:"))
        self.win_top = QDoubleSpinBox()
        self.win_top.setDecimals(1)
        self.win_top.setRange(-1e9, 1e9)
        self.win_top.setSingleStep(10.0)
        self.win_top.setValue(0.0)
        self.win_top.valueChanged.connect(lambda _: self.update_overlay())
        rowW.addWidget(self.win_top)

        rowW.addWidget(QLabel("Len:"))
        self.win_len = QDoubleSpinBox()
        self.win_len.setDecimals(1)
        self.win_len.setRange(10.0, 2000.0)
        self.win_len.setSingleStep(10.0)
        self.win_len.setValue(150.0)
        self.win_len.valueChanged.connect(lambda _: self.update_overlay())
        rowW.addWidget(self.win_len)
        lay.addLayout(rowW)

      
      



        self.only_window = QCheckBox("Show window only")
        
        # Window navigation (Prev / Next window)
        self.prev_win_btn = QPushButton("◀ Prev window")
        self.next_win_btn = QPushButton("Next window ▶")
        self.prev_win_btn.clicked.connect(lambda: self.step_window(-1))
        self.next_win_btn.clicked.connect(lambda: self.step_window(+1))
        
        win_nav = QHBoxLayout()
        win_nav.addWidget(self.prev_win_btn)
        win_nav.addWidget(self.next_win_btn)
        win_nav.addStretch(1)
        lay.addLayout(win_nav)
        
        # Run navigation (Prev / Next run)  (if you actually want these)
        self.prev_btn = QPushButton("◀ Prev")
        self.next_btn = QPushButton("Next ▶")
        # TODO: connect these to something meaningful
        # self.prev_btn.clicked.connect(self.prev_run)
        # self.next_btn.clicked.connect(self.next_run)
        
        nav = QHBoxLayout()
        nav.addWidget(self.prev_btn)
        nav.addWidget(self.next_btn)
        nav.addStretch(1)
        lay.addLayout(nav)

        self.only_window.setChecked(True)
        self.only_window.stateChanged.connect(lambda _: self.update_overlay())
        lay.addWidget(self.only_window)













        
        self.only_window.setChecked(True)
        self.only_window.stateChanged.connect(lambda _: self.update_overlay())
        lay.addWidget(self.only_window)

        # -----------------------------
        # Bulk shift slider (scaled)
        # -----------------------------
        self.shift_label = QLabel("Shift: 0.00 ft (moving run depth - shift)")
        lay.addWidget(self.shift_label)

        self.shift_slider = QSlider(Qt.Horizontal)
        self.shift_slider.setMinimum(int(-self.shift_range_ft / self.shift_step_ft))
        self.shift_slider.setMaximum(int(+self.shift_range_ft / self.shift_step_ft))
        self.shift_slider.setSingleStep(1)
        self.shift_slider.setPageStep(int(5 / self.shift_step_ft))
        self.shift_slider.setValue(0)
        self.shift_slider.valueChanged.connect(self._on_shift_changed)
        lay.addWidget(self.shift_slider)

        self.lock_btn = QPushButton("Lock shift")
        self.lock_btn.clicked.connect(self._lock_shift)
        lay.addWidget(self.lock_btn)

        self.compute_windows_btn = QPushButton("Compute window shifts (window len)")
        self.compute_windows_btn.clicked.connect(self.compute_window_shifts)
        lay.addWidget(self.compute_windows_btn)

        self.results_box = QTextEdit()
        self.results_box.setReadOnly(True)
        self.results_box.setMinimumHeight(220)
        lay.addWidget(self.results_box)

        lay.addStretch(1)
        return w

    # -----------------------------
    # Callbacks
    # -----------------------------
    def _on_base_changed(self, idx: int):
        self.base_idx = idx
        self._rebuild_curve_choices()
        self.update_overlay()

    def _on_mov_changed(self, idx: int):
        self.mov_idx = idx
        self._rebuild_curve_choices()

        # reset window top to moving run top
        if self.runs:
            try:
                self.win_top.setValue(float(self.runs[self.mov_idx].df.index.min()))
            except Exception:
                pass

        self.update_overlay()

    def _on_shift_changed(self, v: int):
        self.shift_ft = float(v) * float(self.shift_step_ft)
        self.shift_label.setText(f"Shift: {self.shift_ft:.2f} ft (moving run depth - shift)")
        self.update_overlay()

    def _lock_shift(self):
        if not self.runs or self.base_idx == self.mov_idx:
            return

        base_name = self.runs[self.base_idx].name
        mov_name = self.runs[self.mov_idx].name
        QMessageBox.information(
            self,
            "Locked",
            f"Locked bulk shift\n\nBase: {base_name}\nMoving: {mov_name}\nShift: {self.shift_ft:.2f} ft\n\n"
            "(Next: store in shift_map / build shift curve)"
        )

    # -----------------------------
    # File actions
    # -----------------------------
    def open_las_runs(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Open LAS Runs", "", "LAS (*.las *.LAS)")
        if not paths:
            return

        new_runs: List[LasRun] = []
        for p in paths:
            try:
                df = _las_to_df(p)
                name = os.path.splitext(os.path.basename(p))[0]
                new_runs.append(LasRun(name=name, path=p, df=df))
            except Exception as e:
                QMessageBox.critical(self, "LAS Load Error", f"Failed to load:\n{p}\n\n{e}")
                return

        self.runs = new_runs

        # Defaults
        self.base_idx = 0
        self.mov_idx = 1 if len(self.runs) > 1 else 0

        # Reset shift + outputs
        self.shift_slider.setValue(0)
        self.shift_ft = 0.0
        self.window_shifts_df = None
        self.results_box.setText("")

        # Initialize window top from moving run (AFTER mov_idx is set)
        if len(self.runs) >= 2:
            self.win_top.setValue(float(self.runs[self.mov_idx].df.index.min()))

        # Update UI
        self._refresh_project_tree()
        self._refresh_run_selectors()
        self._rebuild_curve_choices()
        self.update_overlay()

    def open_parquet(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Parquet", "", "Parquet (*.parquet)")
        if not path:
            return
        ds = Dataset.from_parquet(path)
        self.demo_load(ds)

    # -----------------------------
    # Window shifts (compute + show)
    # -----------------------------
    def compute_window_shifts(self):
        if len(self.runs) < 2:
            QMessageBox.information(self, "Need Runs", "Load at least two LAS runs first.")
            return
        if self.base_idx == self.mov_idx:
            QMessageBox.information(self, "Pick Different Runs", "Base run and moving run must be different.")
            return
        if self.curve_combo.count() == 0:
            QMessageBox.information(self, "No Common Curves", "No common shift curve found between base and moving run.")
            return

        curve = self.curve_combo.currentText()
        df_base = self.runs[self.base_idx].df
        df_mov = self.runs[self.mov_idx].df

        win_ft = float(self.win_len.value())

        try:
            out = windowed_bulk_shifts(
                df_base, df_mov, curve,
                win_ft=win_ft, step_ft=win_ft,
                shift_min=-30.0, shift_max=30.0, shift_step=0.5
            )
        except Exception as e:
            QMessageBox.critical(self, "Window Shift Error", str(e))
            return

        self.window_shifts_df = out

        txt = []
        txt.append(f"Computed {len(out)} windows for curve: {curve}")
        txt.append(f"Base:   {self.runs[self.base_idx].name}")
        txt.append(f"Moving: {self.runs[self.mov_idx].name}")
        txt.append("")
        txt.append("First 10 windows:")
        txt.append(str(out.head(10)))
        txt.append("")
        txt.append("Shift stats (ft):")
        txt.append(str(out["shift_ft"].describe()))
        self.results_box.setText("\n".join(txt))

    # -----------------------------
    # Overlay plotting (bulk shift)
    # -----------------------------
    def update_overlay(self):
        if len(self.runs) < 2:
            return
        if self.base_idx == self.mov_idx:
            return
        if self.curve_combo.count() == 0:
            return

        curve = self.curve_combo.currentText()
        df_base = self.runs[self.base_idx].df
        df_mov = self.runs[self.mov_idx].df
        if curve not in df_base.columns or curve not in df_mov.columns:
            return

        # Base
        z_base = df_base.index.values.astype(float)
        x_base = pd.to_numeric(df_base[curve], errors="coerce").values.astype(float)

        # Window restriction
        win_top = float(self.win_top.value())
        win_len = float(self.win_len.value())
        win_base = win_top + win_len

        if self.only_window.isChecked():
            mwin = (z_base >= win_top) & (z_base <= win_base)
            if mwin.sum() < 5:
                return
            z_base = z_base[mwin]
            x_base = x_base[mwin]

        # Moving (shifted)
        z_mov = df_mov.index.values.astype(float)
        x_mov = pd.to_numeric(df_mov[curve], errors="coerce").values.astype(float)

        z_mov_warp = z_mov - self.shift_ft

        m = np.isfinite(z_mov_warp) & np.isfinite(x_mov)
        if m.sum() < 5:
            return

        order = np.argsort(z_mov_warp[m])
        z_src = z_mov_warp[m][order]
        x_src = x_mov[m][order]

        x_mov_on_base = np.interp(z_base, z_src, x_src, left=np.nan, right=np.nan)

        base_name = f"{curve}_BASE({self.runs[self.base_idx].name})"
        mov_name  = f"{curve}_MOV_SHIFTED({self.runs[self.mov_idx].name})"

        df_plot = pd.DataFrame(
            {base_name: x_base, mov_name: x_mov_on_base},
            index=z_base,
        )

        # Force colored pens: base black solid, moving red dashed
        import pyqtgraph as pg
        self.log_canvas.curve_pens = {
            base_name: pg.mkPen(color=(0, 0, 0), width=2),
            mov_name:  pg.mkPen(color=(200, 0, 0), width=2, style=Qt.DashLine),
        }

        ds = Dataset(data=df_plot, families_map={"GR": list(df_plot.columns)})

        from petrocore.viz.log_canvas_pg import TrackSpec
        if curve.upper().endswith("GR") or curve.upper() in ["GR", "HSGR", "SGR", "ECGR", "HGR"]:
            tracks = [TrackSpec(name=f"Overlay: {curve}", curves=list(df_plot.columns), x_range=(0, 150))]
        else:
            tracks = [TrackSpec(name=f"Overlay: {curve}", curves=list(df_plot.columns))]

        self.log_canvas.set_dataset(ds)
        self.log_canvas.set_tracks(tracks)

    # -----------------------------
    # Project tree (runs + curves)
    # -----------------------------
    def _refresh_project_tree(self):
        self.project_tree.clear()

        if not self.runs:
            self.project_tree.addTopLevelItem(QTreeWidgetItem(["(no runs loaded)"]))
            return

        for i, r in enumerate(self.runs):
            top = QTreeWidgetItem([f"Run {i}: {r.name}"])
            top.addChild(QTreeWidgetItem([f"Depth: {r.df.index.min():.2f} – {r.df.index.max():.2f}"]))
            curves = QTreeWidgetItem([f"Curves ({len(r.df.columns)})"])

            for c in list(r.df.columns)[:200]:
                curves.addChild(QTreeWidgetItem([c]))
            if len(r.df.columns) > 200:
                curves.addChild(QTreeWidgetItem([f"... ({len(r.df.columns)-200} more)"]))

            top.addChild(curves)
            self.project_tree.addTopLevelItem(top)

        self.project_tree.expandToDepth(1)

    # -----------------------------
    # Selectors + curve choices
    # -----------------------------
    def _refresh_run_selectors(self):
        self.base_combo.blockSignals(True)
        self.mov_combo.blockSignals(True)

        self.base_combo.clear()
        self.mov_combo.clear()

        for r in self.runs:
            self.base_combo.addItem(r.name)
            self.mov_combo.addItem(r.name)

        self.base_combo.setCurrentIndex(min(self.base_idx, max(0, len(self.runs) - 1)))
        self.mov_combo.setCurrentIndex(min(self.mov_idx, max(0, len(self.runs) - 1)))

        self.base_combo.blockSignals(False)
        self.mov_combo.blockSignals(False)

    def _rebuild_curve_choices(self):
        self.curve_combo.blockSignals(True)
        self.curve_combo.clear()

        if len(self.runs) < 2:
            self.curve_combo.blockSignals(False)
            return

        base_cols = list(self.runs[self.base_idx].df.columns)
        mov_cols = list(self.runs[self.mov_idx].df.columns)

        for c in self.gr_candidates:
            if c in base_cols and c in mov_cols:
                self.curve_combo.addItem(c)

        if self.curve_combo.count() == 0:
            common = sorted(set(base_cols).intersection(set(mov_cols)))
            for c in common[:50]:
                self.curve_combo.addItem(c)

        self.curve_combo.blockSignals(False)

    # -----------------------------
    # Demo helper
    # -----------------------------
    def demo_load(self, ds: Dataset):
        self.log_canvas.set_dataset(ds)
        self.log_canvas.set_tracks(LogCanvasPG.standard_4track_template(ds))

    # -----------------------------
    # UI helpers
    # -----------------------------
    def _dock(self, title: str, widget: QWidget) -> QDockWidget:
        d = QDockWidget(title, self)
        d.setWidget(widget)
        d.setObjectName(title.replace(" ", "_"))
        d.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        return d

    def _placeholder(self, text: str) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(QLabel(text))
        lay.addStretch(1)
        return w
