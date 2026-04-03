# apps/merge_gui/ui_main_window.py

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PySide6.QtWidgets import (
    QMainWindow, QDockWidget, QTabWidget, QWidget, QVBoxLayout, QLabel,
    QFileDialog, QTreeWidget, QTreeWidgetItem, QComboBox, QHBoxLayout,
    QSlider, QPushButton, QMessageBox, QTextEdit,
    QDoubleSpinBox, QCheckBox
)
from PySide6.QtCore import Qt, QTimer
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
    df: pd.DataFrame
    well_name: str = ""
    curve_units: dict | None = None
    las: object = None
    
def _las_to_df(path: str):
    las = lasio.read(path)
    df = las.df()

    df.index = pd.to_numeric(df.index, errors="coerce").astype(float)
    df = df[~df.index.isna()].sort_index()

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    well_name = ""
    try:
        well_name = str(las.well.WELL.value).strip()
    except Exception:
        pass

    curve_units = {}
    try:
        for curve in las.curves:
            mnemonic = str(curve.mnemonic).strip()
            unit = str(curve.unit).strip() if curve.unit is not None else ""
            curve_units[mnemonic] = unit
    except Exception:
        pass

    return df, well_name, curve_units, las





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
        
 
        self.auto_shift_curve = None
        self.aligned_runs = {}        
 
 
        self.merged_df = None
        self.merged_well_name = ""
        self.merged_curve_units = {}

        self.shift_table_df = pd.DataFrame(columns=[
            "base_run",
            "moving_run",
            "base_curve",
            "moving_curve",
            "window_top",
            "window_base",
            "z_center",
            "shift_ft",
            "corr",
            "n",
            "source",
            "is_active",
        ])
    
        # manually locked shift picks
        self.shift_picks = []        

        # Curve-family preference for shifting (GR)
        self.gr_candidates = ["GR_EDTC","GR", "HGR", "HSGR", "SGR",  "ECGR"]

        self.gr_coverage_fig = None

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


    
    def _find_gr_candidates(self, cols):
        """
        Return GR-family curve candidates from a list/index of curve names.
        """
        if cols is None or len(cols) == 0:
            return []
    
        preferred = [
            "GR_EDTC","GR","HSGR", "SGR", "CGR", "HCGR",  "ECGR"
        ]
    
        cols_upper = {str(c).strip().upper(): c for c in cols}
        found = []
    
        for name in preferred:
            if name in cols_upper:
                found.append(cols_upper[name])
    
        for c in cols:
            cu = str(c).strip().upper()
            if c not in found and (
                cu == "GR"
                or "GR" in cu
                or cu.endswith("SGR")
                or cu.endswith("CGR")
            ):
                found.append(c)
    
        return found




    def _pick_best_gr_for_run(self, df: pd.DataFrame):
        if df is None or df.empty:
            return None
    
        candidates = self._find_gr_candidates(df.columns)
        if not candidates:
            return None
    
        # prefer your existing GR priority
        preferred = ["GR_EDTC", "GR", "HSGR", "SGR", "CGR", "HCGR", "ECGR"]
    
        cmap = {str(c).strip().upper(): c for c in candidates}
        for name in preferred:
            if name in cmap:
                return cmap[name]
    
        return candidates[0]
    
    
    def _close_gr_coverage_popup(self):
        if self.gr_coverage_fig is not None:
            try:
                plt.close(self.gr_coverage_fig)
            except Exception:
                pass
            self.gr_coverage_fig = None
    
    
    
    
    
    
    
    def _show_gr_coverage_popup(self):
        if not self.runs:
            return
    
        prepared = []
        for run in self.runs:
            try:
                df = run.df.copy()
            except Exception:
                continue
    
            if df is None or df.empty:
                continue
    
            df.index = pd.to_numeric(df.index, errors="coerce")
            df = df[~df.index.isna()].sort_index()
    
            if df.empty:
                continue
    
            gr_col = self._pick_best_gr_for_run(df)
    
            prepared.append({
                "name": run.name,
                "df": df,
                "gr_col": gr_col,
                "zmin": float(df.index.min()),
                "zmax": float(df.index.max()),
            })
    
        if not prepared:
            return
    
        global_top = min(r["zmin"] for r in prepared)
        global_base = max(r["zmax"] for r in prepared)
    
        n = len(prepared)
    
        # 👇 NARROWER FIGURE
        fig_width = max(4, n * 0.4)
        fig, axes = plt.subplots(
            1, n,
            figsize=(fig_width, 10),
            sharey=True
        )
    
        if n == 1:
            axes = [axes]
    
        for ax, run in zip(axes, prepared):
            df = run["df"]
            gr_col = run["gr_col"]
    
            ax.set_facecolor("whitesmoke")
            ax.set_xlim(0, 1)
            ax.set_ylim(global_base, global_top)
    
            # remove ticks
            ax.set_xticks([])
            ax.set_yticks([])
    
            if gr_col is not None and gr_col in df.columns:
                s = pd.to_numeric(df[gr_col], errors="coerce").copy()
                s[s <= -999] = np.nan
                valid = s.notna().astype(int).values
    
                # 👇 RED FILL
                ax.fill_betweenx(df.index.values, 0, valid, color="red")
    
            # 👇 VERTICAL LABEL ON TOP
            ax.text(
                0.5, 0.995,                # x=center, y=top INSIDE axis
                run["name"],
                transform=ax.transAxes,   # use axis coordinates (0–1)
                ha="center",
                va="top",
                rotation=90,
                fontsize=8,
            )
    
        # only first axis shows depth label
        axes[0].set_ylabel("Depth", fontsize=11)
    
        fig.suptitle("GR Coverage - Base Run Selection", fontsize=12)
    
        # 👇 tighten spacing between tracks
        fig.subplots_adjust(wspace=0.02)
    
        self.gr_coverage_fig = fig
    
        plt.show(block=False)
        plt.pause(0.1)
    
    
    
    

  

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

        # Base shift curve
        row3a = QHBoxLayout()
        row3a.addWidget(QLabel("Base curve:"))
        self.base_curve_combo = QComboBox()
        self.base_curve_combo.currentIndexChanged.connect(lambda _: self.update_overlay())
        row3a.addWidget(self.base_curve_combo)
        lay.addLayout(row3a)



        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Moving run:"))
        self.mov_combo = QComboBox()
        self.mov_combo.currentIndexChanged.connect(self._on_mov_changed)
        row2.addWidget(self.mov_combo)
        lay.addLayout(row2)

        
        # Moving shift curve
        row3b = QHBoxLayout()
        row3b.addWidget(QLabel("Moving curve:"))
        self.mov_curve_combo = QComboBox()
        self.mov_curve_combo.currentIndexChanged.connect(lambda _: self.update_overlay())
        row3b.addWidget(self.mov_curve_combo)
        lay.addLayout(row3b)


        self.shift_view_combo = QComboBox()
        self.shift_view_combo.addItems(["Original", "Auto-shifted", "Final edited"])
        self.shift_view_combo.currentIndexChanged.connect(lambda _: self.update_overlay())

        but = QHBoxLayout()
        but.addWidget(self.shift_view_combo)
        but.addStretch(1)
        lay.addLayout(but)
        
        
        
        
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
        #self.prev_btn = QPushButton("◀ Prev")
        #self.next_btn = QPushButton("Next ▶")
        # TODO: connect these to something meaningful
        # self.prev_btn.clicked.connect(self.prev_run)
        # self.next_btn.clicked.connect(self.next_run)
        
        #nav = QHBoxLayout()
        #nav.addWidget(self.prev_btn)
        #nav.addWidget(self.next_btn)
        #nav.addStretch(1)
        #lay.addLayout(nav)

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

        self.lock_btn = QPushButton("Lock Manual Depth shifts if Any")
        self.lock_btn.clicked.connect(self._lock_shift)
        lay.addWidget(self.lock_btn)

        self.compute_windows_btn = QPushButton("Compute Auto window shifts (window len)")
        self.compute_windows_btn.clicked.connect(self.compute_window_shifts)
        lay.addWidget(self.compute_windows_btn)

        self.apply_auto_btn = QPushButton("Apply auto shifts")
        self.apply_auto_btn.clicked.connect(self.apply_auto_shift_to_moving_run)
        lay.addWidget(self.apply_auto_btn)


        self.save_shift_btn = QPushButton("Save Shift Table")
        self.save_shift_btn.clicked.connect(self.save_shift_table)
        lay.addWidget(self.save_shift_btn)


        self.merge_btn = QPushButton("Build merged dataset")
        self.merge_btn.clicked.connect(self.build_merged_dataset)
        lay.addWidget(self.merge_btn)
        
        self.save_merge_btn = QPushButton("Save merged LAS")
        self.save_merge_btn.clicked.connect(self.save_merged_las)
        lay.addWidget(self.save_merge_btn)
        


        
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

    def _first_valid_depth(df, curve_name):
        if df is None or curve_name not in df.columns:
            return None
    
        s = pd.to_numeric(df[curve_name], errors="coerce").copy()
        s[s <= -999] = np.nan
    
        valid_depths = df.index[s.notna()]
        if len(valid_depths) == 0:
            return None
    
        return float(valid_depths.min())
    


    
    def _first_valid_depth(self, df, curve_name):
        if df is None or not curve_name or curve_name not in df.columns:
            return None
    
        s = pd.to_numeric(df[curve_name], errors="coerce").copy()
        s[s <= -999] = np.nan
    
        valid_depths = df.index[s.notna()]
        if len(valid_depths) == 0:
            return None
    
        return float(valid_depths.min())
    
    
    def _set_initial_window_top_for_current_run(self):
        if not self.runs:
            return
    
        try:
            df_mov = self.runs[self.mov_idx].df
            df_base = self.runs[self.base_idx].df if self.base_idx is not None else None
    
            mov_curve = self.mov_curve.currentText().strip() if hasattr(self, "mov_curve") else None
            base_curve = self.base_curve.currentText().strip() if hasattr(self, "base_curve") else None
    
            mov_top = self._first_valid_depth(df_mov, mov_curve)
            base_top = self._first_valid_depth(df_base, base_curve) if df_base is not None else None
    
            if mov_top is not None and base_top is not None:
                top = max(mov_top, base_top)
            elif mov_top is not None:
                top = mov_top
            else:
                top = float(df_mov.index.min())
    
            self.win_top.blockSignals(True)
            self.win_top.setValue(top)
            self.win_top.blockSignals(False)
    
            print(f"Initial window top set to {top}")
    
            self.update_overlay()
    
        except Exception as e:
            print("Failed to set initial window top:", e)
    
    
    
        
    def _on_mov_changed(self, idx: int):
        self.mov_idx = idx
        self._rebuild_curve_choices()
    
        # Delay until combo boxes finish updating
        QTimer.singleShot(0, self._set_initial_window_top_for_current_run)
    
    







    def _on_shift_changed(self, v: int):
        self.shift_ft = float(v) * float(self.shift_step_ft)
        self.shift_label.setText(f"Shift: {self.shift_ft:.2f} ft (moving run depth - shift)")
        self.update_overlay()

    
    def _lock_shift(self):
        if not self.runs or self.base_idx == self.mov_idx:
            return
    
        base_name = self.runs[self.base_idx].name
        mov_name = self.runs[self.mov_idx].name
    
        base_curve = self.base_curve_combo.currentText() if hasattr(self, "base_curve_combo") else ""
        mov_curve = self.mov_curve_combo.currentText() if hasattr(self, "mov_curve_combo") else ""
    
        win_top = float(self.win_top.value())
        win_len = float(self.win_len.value())
        win_base = win_top + win_len
        z_center = 0.5 * (win_top + win_base)
    
        shift_val = float(self.shift_ft)
    
        # old lightweight list, if you still want to keep it
        pick = {
            "base_run": base_name,
            "moving_run": mov_name,
            "window_top": win_top,
            "window_base": win_base,
            "shift_ft": shift_val,
        }
        self.shift_picks.append(pick)
    
        # append to master shift table
        manual_row = pd.DataFrame([{
            "base_run": base_name,
            "moving_run": mov_name,
            "base_curve": base_curve,
            "moving_curve": mov_curve,
            "window_top": win_top,
            "window_base": win_base,
            "z_center": z_center,
            "shift_ft": shift_val,
            "corr": np.nan,
            "n": np.nan,
            "source": "manual",
            "is_active": True,
        }])
    
        if self.shift_table_df is None or self.shift_table_df.empty:
            self.shift_table_df = manual_row.copy()
        else:
            self.shift_table_df = pd.concat([self.shift_table_df, manual_row], ignore_index=True)
    
        self._refresh_shift_picks_display()
    
        QMessageBox.information(
            self,
            "Locked",
            f"Locked manual shift\n\n"
            f"Base: {base_name}\n"
            f"Moving: {mov_name}\n"
            f"Base curve: {base_curve}\n"
            f"Moving curve: {mov_curve}\n"
            f"Window: {win_top:.1f} - {win_base:.1f} ft\n"
            f"Shift: {shift_val:.2f} ft"
        )
        
        
    
    
    
    def _refresh_shift_picks_display(self):
        if not self.shift_picks:
            self.results_box.setPlainText("No locked shift picks yet.")
            return
        
        self.results_box.setText("Loaded new LAS runs. Previous shift data cleared.")
    
        lines = []
        lines.append("Locked shift picks:")
        lines.append("")
    
        for i, p in enumerate(self.shift_picks, start=1):
            lines.append(
                f"{i:02d}. "
                f"{p['moving_run']} vs {p['base_run']} | "
                f"{p['window_top']:.1f}-{p['window_base']:.1f} ft | "
                f"shift = {p['shift_ft']:+.2f} ft"
            )
    
        self.results_box.setPlainText("\n".join(lines))


    def _norm_cols_map(self, cols):
        return {str(c).strip().upper(): c for c in cols}
    
    def _first_present(self, cols, candidates):
        cmap = self._norm_cols_map(cols)
        for cand in candidates:
            key = str(cand).strip().upper()
            if key in cmap:
                return cmap[key]
        return None    


    def save_shift_table(self):
        if getattr(self, "shift_table_df", None) is None or self.shift_table_df.empty:
            QMessageBox.information(self, "No Shift Table", "No shift data to save.")
            return
    
        # default file name based on runs
        base_name = self.runs[self.base_idx].name if self.runs else "base"
        mov_name = self.runs[self.mov_idx].name if self.runs else "moving"
        default_name = f"{base_name}__{mov_name}_shift_table.csv"
    
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Shift Table",
            default_name,
            "CSV Files (*.csv)"
        )
    
        if not path:
            return
    
        try:
            self.shift_table_df.to_csv(path, index=False)
    
            QMessageBox.information(
                self,
                "Saved",
                f"Shift table saved to:\n{path}"
            )
    
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))
    
    

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
                df, well_name, curve_units, las = _las_to_df(p)
                name = os.path.splitext(os.path.basename(p))[0]
        
                new_runs.append(
                    LasRun(
                        name=name,
                        path=p,
                        df=df,
                        well_name=well_name,
                        curve_units=curve_units,
                        las=las,
                    )
                )
        
            except Exception as e:
                QMessageBox.critical(self, "LAS Load Error", f"Failed to load:\n{p}\n\n{e}")
                return
        
                
                
  

        self.runs = new_runs
        
         # -----------------------------
        # CLEAR SHIFT STATE (IMPORTANT)
        # -----------------------------
        self.window_shifts_df = None
        self.shift_picks = []
        self.auto_shift_curve = None
        self.aligned_runs = {}
        
        # clear master shift table
        if hasattr(self, "shift_table_df"):
            self.shift_table_df = self.shift_table_df.iloc[0:0].copy()
        else:
            self.shift_table_df = pd.DataFrame()
        
        # clear results box
        self.results_box.setText("")       
        
        

        # Defaults
        self.base_idx = 0
        self.mov_idx = 1 if len(self.runs) > 1 else 0

        # Reset shift + outputs

        # Reset shift + outputs
        self.shift_slider.setValue(0)
        self.shift_ft = 0.0
        self.window_shifts_df = None
        self.shift_picks = []
        self.results_box.setText("")


        # Initialize window top from moving run (AFTER mov_idx is set)
        if len(self.runs) >= 2:
            self.win_top.setValue(float(self.runs[self.mov_idx].df.index.min()))

        # Update UI
        self._refresh_project_tree()
        self._refresh_run_selectors()
        self._rebuild_curve_choices()
        self.update_overlay()

        self._close_gr_coverage_popup()
        self._show_gr_coverage_popup()







    def open_parquet(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Parquet", "", "Parquet (*.parquet)")
        if not path:
            return
        ds = Dataset.from_parquet(path)
        self.demo_load(ds)



    
    def step_window(self, direction: int):
        if not self.runs or self.base_idx == self.mov_idx:
            return
    
        step = float(self.win_len.value()) * 0.5
        new_top = float(self.win_top.value()) + direction * step
    
        try:
            df_base = self.runs[self.base_idx].df
            df_mov = self.runs[self.mov_idx].df
    
            overlap_top = max(float(df_base.index.min()), float(df_mov.index.min()))
            overlap_bot = min(float(df_base.index.max()), float(df_mov.index.max()))
            win_len = float(self.win_len.value())
    
            new_top = max(overlap_top, min(new_top, overlap_bot - win_len))
            self.win_top.setValue(new_top)
        except Exception:
            pass
    
        
    def get_runs_for_merge(self):
        if not self.runs:
            return []
    
        merged_inputs = []
    
        for i, run in enumerate(self.runs):
            if i == self.base_idx:
                merged_inputs.append(run.df.copy())
            elif hasattr(self, "aligned_runs") and i in self.aligned_runs:
                merged_inputs.append(self.aligned_runs[i].copy())
            else:
                # use original if not yet auto-aligned
                merged_inputs.append(run.df.copy())
    
        return merged_inputs   

    def merge_aligned_runs(self, aligned_run_dfs, depth_step=0.5):
        """
        Merge aligned runs into one dataframe.
    
        - Builds a common depth grid
        - Interpolates all curves
        - Prefers first run (base) in overlap
        """
        if not aligned_run_dfs:
            return None
    
        zmin = min(df.index.min() for df in aligned_run_dfs if len(df.index) > 0)
        zmax = max(df.index.max() for df in aligned_run_dfs if len(df.index) > 0)
    
        depth_grid = np.arange(zmin, zmax + depth_step, depth_step)
        merged = pd.DataFrame(index=depth_grid)
    
        for df in aligned_run_dfs:
            if df is None or df.empty:
                continue
    
            for c in df.columns:
                vals = np.interp(
                    depth_grid,
                    df.index.values.astype(float),
                    pd.to_numeric(df[c], errors="coerce").values.astype(float),
                    left=np.nan,
                    right=np.nan
                )
    
                if c not in merged.columns:
                    merged[c] = vals
                else:
                    mask = merged[c].isna()
                    merged.loc[mask, c] = vals[mask]
    
        merged.index.name = "DEPT"
        return merged
    
    def build_merged_dataset(self):
        run_list = self.get_runs_for_merge()
        if not run_list:
            QMessageBox.information(self, "No Runs", "No runs available to merge.")
            return
    
        merged_df = self.merge_aligned_runs(run_list, depth_step=0.5)
        if merged_df is None or merged_df.empty:
            QMessageBox.warning(self, "Merge Failed", "Merged dataframe is empty.")
            return
    
        self.merged_df = merged_df
        
        
        base_run = self.runs[self.base_idx]
        self.merged_well_name = base_run.well_name if getattr(base_run, "well_name", "") else base_run.name
        
        # start with base run units, then fill missing from other runs
        merged_units = {}
        for i, run in enumerate(self.runs):
            units = getattr(run, "curve_units", {}) or {}
            for k, v in units.items():
                if k not in merged_units or not merged_units[k]:
                    merged_units[k] = v
        
        self.merged_curve_units = merged_units       
        
        
        
    
        lines = []
        lines.append("Merged dataset created")
        lines.append("")
        lines.append(f"Rows:    {len(merged_df)}")
        lines.append(f"Curves:  {len(merged_df.columns)}")
        lines.append(f"Depth:   {merged_df.index.min():.2f} - {merged_df.index.max():.2f}")
        lines.append("")
        lines.append("First 20 curves:")
        for c in list(merged_df.columns)[:20]:
            lines.append(f"  {c}")
    
        self.results_box.setPlainText("\n".join(lines))
        QMessageBox.information(self, "Merge Complete", "Merged dataset has been built.")
    
    
        
    
    
    def save_merged_las(self):
        if self.merged_df is None or self.merged_df.empty:
            QMessageBox.information(self, "No Merged Data", "Build the merged dataset first.")
            return
    
        well_name = self.merged_well_name.strip() if self.merged_well_name else "MERGED"
        safe_well_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in well_name)
    
        default_name = f"{safe_well_name}_merged.las"
    
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Merged LAS",
            default_name,
            "LAS Files (*.las)"
        )
        if not path:
            return
    
        try:
            base_run = self.runs[self.base_idx] if self.runs else None
            las = lasio.LASFile()
    
            # ---------------------------------
            # Copy header metadata from base LAS
            # ---------------------------------
            if base_run is not None and getattr(base_run, "las", None) is not None:
                base_las = base_run.las
    
                # Version section
                try:
                    for item in base_las.version:
                        las.version[item.mnemonic] = item.value
                except Exception:
                    pass
    
                # Well section
                try:
                    for item in base_las.well:
                        las.well[item.mnemonic] = item.value
                except Exception:
                    pass
    
                # Params section
                try:
                    for item in base_las.params:
                        las.params[item.mnemonic] = item.value
                except Exception:
                    pass
    
                # Other section text
                try:
                    las.other = base_las.other
                except Exception:
                    pass
    
            # ---------------------------------
            # Overwrite key well headers
            # ---------------------------------
            las.well.WELL.value = well_name
            las.well.STRT.value = float(self.merged_df.index.min())
            las.well.STOP.value = float(self.merged_df.index.max())
    
            if len(self.merged_df.index) > 1:
                step = float(np.median(np.diff(self.merged_df.index.values.astype(float))))
            else:
                step = 0.5
    
            las.well.STEP.value = step
            las.well.NULL.value = -999.25
    
            # ---------------------------------
            # Clear existing curves and write merged curves
            # ---------------------------------
            try:
                las.curves = []
            except Exception:
                pass
    
            # Depth curve
            las.append_curve("DEPT", self.merged_df.index.values.astype(float), unit="ft")
    
            # Data curves with units
            curve_units = self.merged_curve_units if hasattr(self, "merged_curve_units") else {}
    
            for col in self.merged_df.columns:
                vals = pd.to_numeric(self.merged_df[col], errors="coerce").fillna(-999.25).values
                unit = curve_units.get(str(col), "")
                las.append_curve(str(col), vals, unit=unit)
    
            las.write(path, version=2.0)
    
            QMessageBox.information(
                self,
                "Saved",
                f"Merged LAS saved to:\n{path}"
            )
    
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))
    
    
    
    def compute_window_shifts(self):
        if len(self.runs) < 2:
            QMessageBox.information(self, "Need Runs", "Load at least two LAS runs first.")
            return
    
        if self.base_idx == self.mov_idx:
            QMessageBox.information(self, "Pick Different Runs", "Base run and moving run must be different.")
            return
    
        if self.base_curve_combo.count() == 0 or self.mov_curve_combo.count() == 0:
            QMessageBox.information(
                self,
                "No Shift Curves",
                "No GR-type shift curves found for one or both runs."
            )
            return
    
        base_curve = self.base_curve_combo.currentText()
        mov_curve = self.mov_curve_combo.currentText()
    
        if not base_curve or not mov_curve:
            QMessageBox.information(self, "No Shift Curve", "Please select both base and moving curves.")
            return
    
        df_base = self.runs[self.base_idx].df
        df_mov = self.runs[self.mov_idx].df.copy()
    
        if base_curve not in df_base.columns:
            QMessageBox.warning(self, "Missing Base Curve", f"Base curve '{base_curve}' not found.")
            return
    
        if mov_curve not in df_mov.columns:
            QMessageBox.warning(self, "Missing Moving Curve", f"Moving curve '{mov_curve}' not found.")
            return
    
        # Make moving curve temporarily available under base curve name
        if mov_curve != base_curve:
            df_mov[base_curve] = pd.to_numeric(df_mov[mov_curve], errors="coerce")
    
        curve = base_curve
        win_ft = float(self.win_len.value())
    
        try:
            out = windowed_bulk_shifts(
                df_base,
                df_mov,
                curve,
                win_ft=win_ft,
                step_ft=20.0,
                shift_min=-30.0,
                shift_max=30.0,
                shift_step=0.5
            )
    
            shifts_df = out[0] if isinstance(out, tuple) else out
    
        except Exception as e:
            QMessageBox.critical(self, "Window Shift Error", str(e))
            return
    
        if shifts_df is None or len(shifts_df) == 0:
            QMessageBox.warning(self, "No Results", "windowed_bulk_shifts returned no results.")
            return
    
        self.window_shifts_df = shifts_df.copy()
    
        # -------------------------------------------------------
        # Normalize auto shifts into self.shift_table_df
        # -------------------------------------------------------
        auto_df = shifts_df.copy()
    
        # Normalize window depth columns
        if {"z0", "z1"}.issubset(auto_df.columns):
            auto_df["window_top"] = pd.to_numeric(auto_df["z0"], errors="coerce")
            auto_df["window_base"] = pd.to_numeric(auto_df["z1"], errors="coerce")
    
        elif {"top_depth", "base_depth"}.issubset(auto_df.columns):
            auto_df["window_top"] = pd.to_numeric(auto_df["top_depth"], errors="coerce")
            auto_df["window_base"] = pd.to_numeric(auto_df["base_depth"], errors="coerce")
    
        elif {"window_top", "window_base"}.issubset(auto_df.columns):
            auto_df["window_top"] = pd.to_numeric(auto_df["window_top"], errors="coerce")
            auto_df["window_base"] = pd.to_numeric(auto_df["window_base"], errors="coerce")
    
        elif {"z_center"}.issubset(auto_df.columns):
            # fallback if only z_center exists
            auto_df["z_center"] = pd.to_numeric(auto_df["z_center"], errors="coerce")
            auto_df["window_top"] = auto_df["z_center"] - 0.5 * win_ft
            auto_df["window_base"] = auto_df["z_center"] + 0.5 * win_ft
    
        else:
            auto_df["window_top"] = np.nan
            auto_df["window_base"] = np.nan
    
        # z_center
        if "z_center" in auto_df.columns:
            auto_df["z_center"] = pd.to_numeric(auto_df["z_center"], errors="coerce")
        else:
            auto_df["z_center"] = 0.5 * (
                pd.to_numeric(auto_df["window_top"], errors="coerce") +
                pd.to_numeric(auto_df["window_base"], errors="coerce")
            )
    
        # numeric columns
        if "shift_ft" in auto_df.columns:
            auto_df["shift_ft"] = pd.to_numeric(auto_df["shift_ft"], errors="coerce")
        else:
            auto_df["shift_ft"] = np.nan
    
        if "corr" in auto_df.columns:
            auto_df["corr"] = pd.to_numeric(auto_df["corr"], errors="coerce")
        else:
            auto_df["corr"] = np.nan
    
        if "n" in auto_df.columns:
            auto_df["n"] = pd.to_numeric(auto_df["n"], errors="coerce")
        else:
            auto_df["n"] = np.nan
    
        # metadata
        pair_base = self.runs[self.base_idx].name
        pair_mov = self.runs[self.mov_idx].name
    
        auto_df["base_run"] = pair_base
        auto_df["moving_run"] = pair_mov
        auto_df["base_curve"] = base_curve
        auto_df["moving_curve"] = mov_curve
        auto_df["source"] = "auto"
        auto_df["is_active"] = True
    
        # keep only standard columns
        cols = [
            "base_run",
            "moving_run",
            "base_curve",
            "moving_curve",
            "window_top",
            "window_base",
            "z_center",
            "shift_ft",
            "corr",
            "n",
            "source",
            "is_active",
        ]
    
        for c in cols:
            if c not in auto_df.columns:
                auto_df[c] = np.nan
    
        auto_df = auto_df[cols].reset_index(drop=True)
    
        # Remove existing auto rows for this run pair, keep manual rows
        if self.shift_table_df is None or self.shift_table_df.empty:
            self.shift_table_df = auto_df.copy()
        else:
            keep_mask = ~(
                (self.shift_table_df["source"] == "auto") &
                (self.shift_table_df["base_run"] == pair_base) &
                (self.shift_table_df["moving_run"] == pair_mov)
            )
            self.shift_table_df = self.shift_table_df.loc[keep_mask].copy()
            self.shift_table_df = pd.concat([self.shift_table_df, auto_df], ignore_index=True)
    
        # -------------------------------------------------------
        # Results box stays as quick inspection
        # -------------------------------------------------------
        txt = []
        txt.append(f"Computed {len(shifts_df)} raw windows")
        txt.append(f"Stored {len(auto_df)} auto rows in shift_table_df")
        txt.append(f"Base:   {pair_base}")
        txt.append(f"Moving: {pair_mov}")
        txt.append(f"Base curve:   {base_curve}")
        txt.append(f"Moving curve: {mov_curve}")
        txt.append(f"Window length: {win_ft:.1f} ft")
        txt.append(f"Step: 20.0 ft")
        txt.append("")
    
        txt.append("First 10 raw windows:")
        if hasattr(shifts_df, "head"):
            txt.append(shifts_df.head(10).to_string(index=False))
        else:
            txt.append(str(shifts_df)[:3000])
    
        txt.append("")
        txt.append("First 10 stored rows for this run pair:")
        pair_rows = self.shift_table_df[
            (self.shift_table_df["base_run"] == pair_base) &
            (self.shift_table_df["moving_run"] == pair_mov)
        ].copy()
    
        if len(pair_rows) > 0:
            txt.append(pair_rows.head(10).to_string(index=False))
        else:
            txt.append("(none)")
    
        self.results_box.setPlainText("\n".join(txt))
    

    # -----------------------------
    # Window shifts (compute + show)
    # -----------------------------


        
    
    def build_auto_shift_curve(self):
        if self.window_shifts_df is None or len(self.window_shifts_df) == 0:
            QMessageBox.information(self, "No Auto Shifts", "Run 'Compute window shifts' first.")
            return None
    
        if not self.runs or self.base_idx == self.mov_idx:
            return None
    
        df_mov = self.runs[self.mov_idx].df
    
        out = self.window_shifts_df
        if isinstance(out, tuple):
            out = out[0]
    
        if out is None or len(out) == 0:
            QMessageBox.information(self, "No Auto Shifts", "Run 'Compute window shifts' first.")
            return None
    
        out = out.copy()
    
        if "shift_ft" not in out.columns:
            QMessageBox.warning(self, "Missing Column", "window_shifts_df has no 'shift_ft' column.")
            return None
    
        # ---------------------------------
        # Determine window centers
        # ---------------------------------
        if {"top_depth", "base_depth"}.issubset(out.columns):
            zc = 0.5 * (pd.to_numeric(out["top_depth"], errors="coerce") +
                        pd.to_numeric(out["base_depth"], errors="coerce"))
        elif {"z_top", "z_bot"}.issubset(out.columns):
            zc = 0.5 * (pd.to_numeric(out["z_top"], errors="coerce") +
                        pd.to_numeric(out["z_bot"], errors="coerce"))
        elif {"z0", "z1"}.issubset(out.columns):
            zc = 0.5 * (pd.to_numeric(out["z0"], errors="coerce") +
                        pd.to_numeric(out["z1"], errors="coerce"))
        elif "z_center" in out.columns:
            zc = pd.to_numeric(out["z_center"], errors="coerce")
        elif "window_top" in out.columns:
            win_ft = float(self.win_len.value())
            zc = pd.to_numeric(out["window_top"], errors="coerce") + 0.5 * win_ft
        else:
            QMessageBox.warning(
                self,
                "Missing Depth Columns",
                "Could not determine window centers from auto-shift output."
            )
            return None
    
        sh = pd.to_numeric(out["shift_ft"], errors="coerce")
    
        # ---------------------------------
        # Basic validity
        # ---------------------------------
        good = np.isfinite(zc) & np.isfinite(sh)
    
        # Filter by correlation, if available
        corr_threshold = 0.75
        if "corr" in out.columns:
            corr = pd.to_numeric(out["corr"], errors="coerce")
            good &= np.isfinite(corr) & (corr >= corr_threshold)
    
        # Filter by sample count, if available
        min_n = 150
        if "n" in out.columns:
            npts = pd.to_numeric(out["n"], errors="coerce")
            good &= np.isfinite(npts) & (npts >= min_n)
    
        # Reject absurd shifts
        max_abs_shift = 5.0
        good &= np.abs(sh) <= max_abs_shift
    
        n_raw = len(out)
        n_good = int(np.sum(good))
    
        if n_good < 2:
            debug_lines = []
            debug_lines.append(f"Raw windows: {n_raw}")
            debug_lines.append(f"Valid after filtering: {n_good}")
            debug_lines.append(f"corr threshold: >= {corr_threshold}")
            debug_lines.append(f"n threshold: >= {min_n}")
            debug_lines.append(f"|shift_ft| threshold: <= {max_abs_shift}")
            if "corr" in out.columns:
                debug_lines.append("")
                debug_lines.append("Raw corr values:")
                debug_lines.append(pd.to_numeric(out["corr"], errors="coerce").to_string(index=False))
            QMessageBox.warning(
                self,
                "Too Few Points",
                "Need at least 2 valid auto-shift windows after filtering.\n\n"
                + "\n".join(debug_lines)
            )
            return None
    
        zc = np.asarray(zc[good], dtype=float)
        sh = np.asarray(sh[good], dtype=float)
    
        # ---------------------------------
        # Sort by depth
        # ---------------------------------
        order = np.argsort(zc)
        zc = zc[order]
        sh = sh[order]
    
        # ---------------------------------
        # Remove outlier window shifts using MAD
        # ---------------------------------
        if len(sh) >= 3:
            med = np.nanmedian(sh)
            mad = np.nanmedian(np.abs(sh - med))
    
            if np.isfinite(mad) and mad > 0:
                sigma_robust = 1.4826 * mad
                keep = np.abs(sh - med) <= 3.5 * sigma_robust
                zc = zc[keep]
                sh = sh[keep]
    
        if len(sh) < 2:
            QMessageBox.warning(
                self,
                "Too Few Points",
                "Too few auto-shift windows remain after outlier filtering."
            )
            return None
    
        # ---------------------------------
        # Smooth slightly to reduce jitter
        # ---------------------------------
        sh_s = pd.Series(sh, index=zc).sort_index()
        sh_s = sh_s.rolling(window=3, center=True, min_periods=1).median()
    
        # ---------------------------------
        # Interpolate to moving run depth
        # ---------------------------------
        z_run = df_mov.index.astype(float).to_numpy()
        shift_interp = np.interp(
            z_run,
            sh_s.index.to_numpy(dtype=float),
            sh_s.to_numpy(dtype=float),
            left=float(sh_s.iloc[0]),
            right=float(sh_s.iloc[-1]),
        )
    
        self.auto_shift_curve = pd.Series(
            data=shift_interp,
            index=df_mov.index.astype(float),
            name="shift_ft",
            dtype=float,
        )
    
        # Optional: write quick summary to results box
        lines = []
        lines.append("Built auto shift curve")
        lines.append(f"Raw windows: {n_raw}")
        lines.append(f"Accepted windows: {len(sh_s)}")
        lines.append(f"corr threshold: >= {corr_threshold}")
        lines.append(f"n threshold: >= {min_n}")
        lines.append(f"|shift_ft| threshold: <= {max_abs_shift}")
        lines.append("")
        lines.append("Accepted window centers and shifts:")
        lines.append(pd.DataFrame({"z_center": sh_s.index, "shift_ft": sh_s.values}).to_string(index=False))
        self.results_box.setPlainText("\n".join(lines))
    
        return self.auto_shift_curve      
     

          
    def apply_auto_shift_to_moving_run(self):
        sc = self.build_auto_shift_curve()
        if sc is None:
            return
    
        df_mov = self.runs[self.mov_idx].df.copy()
    
        # Same sign convention you were already using in your prototype:
        # warped_depth = original_depth - shift_ft
        warped_depth = df_mov.index.astype(float).to_numpy() - sc.to_numpy(dtype=float)
    
        df_aligned = df_mov.copy()
        df_aligned.index = warped_depth
        df_aligned = df_aligned.sort_index()
    
        if not hasattr(self, "aligned_runs"):
            self.aligned_runs = {}
    
        self.aligned_runs[self.mov_idx] = df_aligned
    
        lines = []
        lines.append("Applied auto shift curve")
        lines.append(f"Base run:   {self.runs[self.base_idx].name}")
        lines.append(f"Moving run: {self.runs[self.mov_idx].name}")
        lines.append(f"Aligned samples: {len(df_aligned)}")
        lines.append("")
        lines.append("Auto shift curve stats (ft):")
        lines.append(sc.describe().to_string())
    
        self.results_box.setPlainText("\n".join(lines))
        
        self.update_overlay()        
    
        QMessageBox.information(
            self,
            "Auto Shift Applied",
            f"Applied auto shift curve to:\n{self.runs[self.mov_idx].name}"
        )
    
    
    
    
    # -----------------------------
    # Overlay plotting (bulk shift)
    # -----------------------------
    def update_overlay(self):
        if len(self.runs) < 2:
            return
        if self.base_idx == self.mov_idx:
            return
        if self.base_curve_combo.count() == 0 or self.mov_curve_combo.count() == 0:
            return
    
        base_curve = self.base_curve_combo.currentText()
        mov_curve = self.mov_curve_combo.currentText()
    
        if not base_curve or not mov_curve:
            return
    
        df_base = self.runs[self.base_idx].df
    
            
            
            
                    
        mode = self.shift_view_combo.currentText()
        #mode = self.show_auto_checkbox.currentText()
        
        df_base = self.runs[self.base_idx].df
        df_mov_raw = self.runs[self.mov_idx].df
        
        if mode == "Original":
            df_mov = df_mov_raw
            use_auto_curve = False
            use_final_curve = False
        
        elif mode == "Auto-shifted":
            df_mov = df_mov_raw
            use_auto_curve = True
            use_final_curve = False
        
        elif mode == "Final edited":
            df_mov = df_mov_raw
            use_auto_curve = False
            use_final_curve = True           
            
            
            
    
        print("\n=== UPDATE OVERLAY DEBUG ===")
        print("Base curve:", base_curve, "| in base?", base_curve in df_base.columns)
        print("Moving curve:", mov_curve, "| in moving?", mov_curve in df_mov.columns)
    
        if base_curve not in df_base.columns or mov_curve not in df_mov.columns:
            return
    
        z_base = df_base.index.values.astype(float)
        x_base = pd.to_numeric(df_base[base_curve], errors="coerce").values.astype(float)
    
        win_top = float(self.win_top.value())
        win_len = float(self.win_len.value())
        win_base = win_top + win_len
    
        if self.only_window.isChecked():
            mwin = (z_base >= win_top) & (z_base <= win_base)
            if mwin.sum() < 5:
                return
            z_base = z_base[mwin]
            x_base = x_base[mwin]



            z_mov = df_mov.index.values.astype(float)
            x_mov = pd.to_numeric(df_mov[mov_curve], errors="coerce").values.astype(float)
            
            if mode == "Original":
                z_mov_warp = z_mov - self.shift_ft   # manual slider only
            
            elif mode == "Auto-shifted":
                if self.auto_shift_curve is not None:
                    auto_shift = np.interp(
                        z_mov,
                        self.auto_shift_curve.index.values.astype(float),
                        self.auto_shift_curve.values.astype(float),
                        left=self.auto_shift_curve.iloc[0],
                        right=self.auto_shift_curve.iloc[-1],
                    )
                    z_mov_warp = z_mov - auto_shift
                else:
                    z_mov_warp = z_mov - self.shift_ft
            
            elif mode == "Final edited":
                if self.final_shift_curve is not None:
                    final_shift = np.interp(
                        z_mov,
                        self.final_shift_curve.index.values.astype(float),
                        self.final_shift_curve.values.astype(float),
                        left=self.final_shift_curve.iloc[0],
                        right=self.final_shift_curve.iloc[-1],
                    )
                    z_mov_warp = z_mov - final_shift
                else:
                    z_mov_warp = z_mov - self.shift_ft
            

    
  
        m = np.isfinite(z_mov_warp) & np.isfinite(x_mov)
        if m.sum() < 5:
            return
    
        order = np.argsort(z_mov_warp[m])
        z_src = z_mov_warp[m][order]
        x_src = x_mov[m][order]
    
        x_mov_on_base = np.interp(z_base, z_src, x_src, left=np.nan, right=np.nan)
    
        base_name = f"{base_curve}_BASE({self.runs[self.base_idx].name})"
        mov_name = f"{mov_curve}_MOV_SHIFTED({self.runs[self.mov_idx].name})"
    
        df_plot = pd.DataFrame(
            {base_name: x_base, mov_name: x_mov_on_base},
            index=z_base,
        )
    
        import pyqtgraph as pg
        self.log_canvas.curve_pens = {
            base_name: pg.mkPen(color=(0, 0, 0), width=2),
            mov_name: pg.mkPen(color=(200, 0, 0), width=2, style=Qt.DashLine),
        }
    
        ds = Dataset(data=df_plot, families_map={"GR": list(df_plot.columns)})
    
        from petrocore.viz.log_canvas_pg import TrackSpec
        tracks = [
            TrackSpec(
                name=f"Overlay: {base_curve} vs {mov_curve}",
                curves=list(df_plot.columns),
                x_range=(0, 150),
            )
        ]
    
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
    
            if getattr(r, "well_name", ""):
                top.addChild(QTreeWidgetItem([f"WELL: {r.well_name}"]))
    
            #top.addChild(QTreeWidgetItem([f"Depth: {r.df.index.min():.2f} – {r.df.index.max():.2f}"]))
                        
            if isinstance(r.df, pd.DataFrame) and len(r.df.index) > 0:
                top.addChild(QTreeWidgetItem([f"Depth: {r.df.index.min():.2f} – {r.df.index.max():.2f}"]))
            else:
                top.addChild(QTreeWidgetItem([f"Depth: [invalid df type: {type(r.df)}]"]))           
                        
            
            curves = QTreeWidgetItem([f"Curves ({len(r.df.columns)})"])
    
            for c in list(r.df.columns)[:200]:
                unit = ""
                if getattr(r, "curve_units", None):
                    unit = r.curve_units.get(c, "")
                if unit:
                    curves.addChild(QTreeWidgetItem([f"{c} [{unit}]"]))
                else:
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
        self.base_curve_combo.blockSignals(True)
        self.mov_curve_combo.blockSignals(True)
    
        self.base_curve_combo.clear()
        self.mov_curve_combo.clear()
    
        if not self.runs or self.base_idx >= len(self.runs) or self.mov_idx >= len(self.runs):
            self.base_curve_combo.blockSignals(False)
            self.mov_curve_combo.blockSignals(False)
            return
    
        df_base = self.runs[self.base_idx].df
        df_mov = self.runs[self.mov_idx].df
    
        base_candidates = self._find_gr_candidates(df_base.columns)
        mov_candidates = self._find_gr_candidates(df_mov.columns)
    
        print("\n=== SHIFT CURVE DEBUG ===")
        print("Base run:", self.runs[self.base_idx].name)
        print("Moving run:", self.runs[self.mov_idx].name)
        print("Base GR candidates:", base_candidates)
        print("Moving GR candidates:", mov_candidates)
    
        self.base_curve_combo.addItems(base_candidates)
        self.mov_curve_combo.addItems(mov_candidates)
    
        self.base_curve_combo.blockSignals(False)
        self.mov_curve_combo.blockSignals(False)
    
        if self.base_curve_combo.count() > 0:
            self.base_curve_combo.setCurrentIndex(0)
    
        if self.mov_curve_combo.count() > 0:
            self.mov_curve_combo.setCurrentIndex(0)
    
    






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
