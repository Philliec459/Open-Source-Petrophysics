


# apps/merge_gui/ui_panels/tops_panel.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QAbstractItemView,
    QHeaderView,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QFileDialog,
    QMessageBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
)

print(">>> LOADING tops_panel.py from:", __file__)


@dataclass
class TopInterval:
    formation: str
    top: float
    base: float


class TopsPanel(QWidget):
    """
    Tops / Zone panel

    Responsibilities
    ----------------
    - Display tops for the CURRENT well only
    - Read filtered tops from:
          state.current_tops_df
      and fall back to:
          state.tops_df (only if needed)
    - Build visible top/base intervals
    - Let user select one or more intervals to define ZoI
    - Store selected ZoI in:
          state.zoi_top
          state.zoi_base

    Notes
    -----
    Preferred workflow:
    - Load tops through LoadDataPanel
    - That panel stores:
          state.tops_df          -> all wells
          state.current_tops_df  -> current well only
    - This panel only displays and applies ZoI

    A convenience "Load Tops" button is kept here, but it should follow the
    same state conventions.
    """

    IGNORE_FORMATIONS = {"TD", "TOTAL DEPTH"}

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self._visible_intervals: list[TopInterval] = []
        
        self.above_pad = QDoubleSpinBox()
        self.above_pad.setRange(0.0, 1000.0)
        self.above_pad.setDecimals(1)
        self.above_pad.setSingleStep(1.0)
        self.above_pad.setValue(0.0)
        
        self.below_pad = QDoubleSpinBox()
        self.below_pad.setRange(0.0, 1000.0)
        self.below_pad.setDecimals(1)
        self.below_pad.setSingleStep(1.0)
        self.below_pad.setValue(0.0)        

        self._build_ui()
        self.refresh()

    # ------------------------------------------------------------------
    # State helper
    # ------------------------------------------------------------------
    def _state(self):
        if hasattr(self.controller, "get_state"):
            return self.controller.get_state()
        return self.controller.state

    def _ensure_state_fields(self):
        state = self._state()

        if not hasattr(state, "tops_df"):
            state.tops_df = None

        if not hasattr(state, "current_tops_df"):
            state.current_tops_df = None

        if not hasattr(state, "well_name"):
            state.well_name = None

        if not hasattr(state, "zoi_top"):
            state.zoi_top = None

        if not hasattr(state, "zoi_base"):
            state.zoi_base = None

        if not hasattr(state, "depth_top"):
            state.depth_top = None

        if not hasattr(state, "depth_base"):
            state.depth_base = None

        if not hasattr(state, "plot_top"):
            state.plot_top = None

        if not hasattr(state, "plot_bottom"):
            state.plot_bottom = None

        if not hasattr(state, "zoi_depth_range"):
            state.zoi_depth_range = None

        if not hasattr(state, "zoi_top_raw"):
            state.zoi_top_raw = None
        
        if not hasattr(state, "zoi_base_raw"):
            state.zoi_base_raw = None
        
        if not hasattr(state, "zoi_above_pad"):
            state.zoi_above_pad = None
        
        if not hasattr(state, "zoi_below_pad"):
            state.zoi_below_pad = None

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        self.info = QLabel("No tops loaded.")
        self.info.setWordWrap(True)
    
        self.table = QTableWidget(0, 2)
        #self.table.setHorizontalHeaderLabels(["Formation", "Top (ft)", "Base (ft)"])
        self.table.setHorizontalHeaderLabels(["Formation", "Top"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setSortingEnabled(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.itemSelectionChanged.connect(self._on_top_selected)


    
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeToContents)
    
        #self.btn_load = QPushButton("Load Tops")
        self.btn_apply = QPushButton("Apply Zone")
        #self.btn_clear = QPushButton("Clear Zone")
    
        #self.btn_load.clicked.connect(self.load_tops_file)
        self.btn_apply.clicked.connect(self.apply_selected_zone)
        #self.btn_clear.clicked.connect(self.clear_zone)
    
        btn_row = QHBoxLayout()
        #btn_row.addWidget(self.btn_load)
        btn_row.addWidget(self.btn_apply)
        #btn_row.addWidget(self.btn_clear)
        btn_row.addStretch(1)
    
        pad_box = QGroupBox("ZoI Padding")
        pad_form = QFormLayout(pad_box)
        pad_form.addRow("Above top (ft):", self.above_pad)
        pad_form.addRow("Below base (ft):", self.below_pad)
    
        layout = QVBoxLayout(self)
        layout.addWidget(self.info)
        layout.addLayout(btn_row)
        layout.addWidget(pad_box)
        layout.addWidget(self.table)

    # ------------------------------------------------------------------
    # Public refresh
    # ------------------------------------------------------------------


        
        
    def clear_tops_display(self):
        state = self.controller.get_state()
    
        # clear state
        state.tops_df = None
        state.tops = None
        state.zoi_depth_range = None
        if hasattr(state, "tops_file"):
            state.tops_file = None
        if hasattr(state, "tops_path"):
            state.tops_path = None
        if hasattr(state, "filtered_tops_df"):
            state.filtered_tops_df = None
    
        # clear widget
        self.table.clearContents()
        self.table.setRowCount(0)
        self.table.clearSelection()
        self.info.setText("No tops loaded.")    
        
   
    
    def _on_top_selected(self):
        row = self.table.currentRow()
        if row < 0:
            return
    
        item0 = self.table.item(row, 0)
        item1 = self.table.item(row, 1)
    
        if item0 is None or item1 is None:
            return
    
        self.selected_formation = item0.text()
        try:
            self.selected_top = float(item1.text())
        except Exception:
            self.selected_top = None
    
  

    def refresh(self):
        self._ensure_state_fields()
        self.refresh_from_state()

    def refresh_from_state(self):
        state = self._state()

        # Preferred source: already-filtered tops for current well
        current_tops_df = getattr(state, "current_tops_df", None)

        if isinstance(current_tops_df, pd.DataFrame) and not current_tops_df.empty:
            rows = self._build_intervals_from_current_tops(current_tops_df)
            if rows:
                self._set_visible_intervals(rows)
                current_well_name = self._get_current_well_name() or "current well"
                self.info.setText(
                    f"Showing tops for current well: {current_well_name}   ({len(rows)} tops)"
                )
                return

        # Fallback: if current_tops_df is missing, try filtering all tops
        all_tops_df = getattr(state, "tops_df", None)
        if isinstance(all_tops_df, pd.DataFrame) and not all_tops_df.empty:
            filtered_df = self._fallback_filter_from_all_tops(all_tops_df)
            if filtered_df is not None and not filtered_df.empty:
                state.current_tops_df = filtered_df.copy()
                rows = self._build_intervals_from_current_tops(filtered_df)
                if rows:
                    self._set_visible_intervals(rows)
                    current_well_name = self._get_current_well_name() or "current well"
                    self.info.setText(
                        f"Showing tops for current well: {current_well_name}   ({len(rows)} tops)"
                    )
                    return

        current_well_name = self._get_current_well_name()
        if current_well_name:
            self.info.setText(f"No tops found for current well: {current_well_name}")
        else:
            self.info.setText("No tops loaded or no LAS well selected.")

        self._set_visible_intervals([])

    # ------------------------------------------------------------------
    # Table display
    # ------------------------------------------------------------------
    def _set_visible_intervals(self, rows: list[TopInterval]):
        self._visible_intervals = rows or []
        self.table.setRowCount(0)

        for row in self._visible_intervals:
            r = self.table.rowCount()
            self.table.insertRow(r)

            item0 = QTableWidgetItem(str(row.formation))
            item1 = QTableWidgetItem(f"{row.top:.2f}")
            ###item2 = QTableWidgetItem(f"{row.base:.2f}")

            item1.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            ###item2.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)

            self.table.setItem(r, 0, item0)
            self.table.setItem(r, 1, item1)
            ###self.table.setItem(r, 2, item2)

    # ------------------------------------------------------------------
    # Current well helpers
    # ------------------------------------------------------------------
    def _get_current_well_name(self) -> Optional[str]:
        state = self._state()

        for attr in ["well_name", "current_well_name", "selected_well_name"]:
            val = getattr(state, attr, None)
            if val is not None and str(val).strip():
                return str(val).strip()

        las_runs = getattr(state, "las_runs", None)
        if las_runs:
            try:
                run0 = las_runs[0]
                if isinstance(run0, dict):
                    for key in ["well_name", "well", "name", "uwi", "api"]:
                        val = run0.get(key, None)
                        if val is not None and str(val).strip():
                            return str(val).strip()
            except Exception:
                pass

        return None

    @staticmethod
    def _normalize_well_name(name) -> str:
        if name is None:
            return ""

        s = str(name).strip().upper()
        s = s.replace("_", "")
        s = s.replace("-", "")
        s = s.replace(" ", "")
        return s

    # ------------------------------------------------------------------
    # Build intervals
    # ------------------------------------------------------------------
    def _build_intervals_from_current_tops(self, df: pd.DataFrame) -> list[TopInterval]:
        """
        Expects current_tops_df in normalized format:
            WELL
            TOP_NAME
            TOP_DEPTH
        """
        if df is None or df.empty:
            return []

        work = df.copy()
        work.columns = [str(c).strip() for c in work.columns]

        required = {"TOP_NAME", "TOP_DEPTH"}
        if not required.issubset(set(work.columns)):
            return []

        work = work[["TOP_NAME", "TOP_DEPTH"]].copy()
        work["TOP_NAME"] = work["TOP_NAME"].astype(str).str.strip()
        work["TOP_DEPTH"] = pd.to_numeric(work["TOP_DEPTH"], errors="coerce")
        work = work.dropna(subset=["TOP_NAME", "TOP_DEPTH"]).copy()
        work = work.sort_values("TOP_DEPTH").reset_index(drop=True)

        rows: list[TopInterval] = []

        for i in range(len(work)):
            formation = str(work.loc[i, "TOP_NAME"]).strip()
            if not formation or formation.upper() in self.IGNORE_FORMATIONS:
                continue

            top_depth = float(work.loc[i, "TOP_DEPTH"])

            if i < len(work) - 1:
                base_depth = float(work.loc[i + 1, "TOP_DEPTH"])
            else:
                base_depth = top_depth

            if base_depth < top_depth:
                base_depth = top_depth

            rows.append(
                TopInterval(
                    formation=formation,
                    top=top_depth,
                    base=base_depth,
                )
            )

        return rows

    def _fallback_filter_from_all_tops(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fallback only. Prefer state.current_tops_df from LoadDataPanel.
        This tries to filter an arbitrary tops dataframe using common well columns.
        """
        current_well_name = self._get_current_well_name()
        if not current_well_name:
            return pd.DataFrame()

        dff = df.copy()
        dff.columns = [str(c).strip() for c in dff.columns]

        well_candidates = [
            "WELL", "Well", "Well Name", "WELL_NAME", "Lease",
            "WellName", "UWI", "API", "HOLE_NAME", "BOREHOLE"
        ]

        well_col = next((c for c in well_candidates if c in dff.columns), None)
        if well_col is None:
            return pd.DataFrame()

        cur_norm = self._normalize_well_name(current_well_name)
        work = dff.copy()
        work["_WELL_NORM"] = work[well_col].astype(str).map(self._normalize_well_name)

        # exact normalized match
        out = work.loc[work["_WELL_NORM"] == cur_norm].copy()

        # contains fallback
        if out.empty and cur_norm:
            contains_mask = work["_WELL_NORM"].str.contains(cur_norm, na=False)
            reverse_mask = work["_WELL_NORM"].apply(
                lambda x: isinstance(x, str) and x in cur_norm if cur_norm else False
            )
            out = work.loc[contains_mask | reverse_mask].copy()

        out = out.drop(columns=["_WELL_NORM"], errors="ignore")

        # Try to normalize if not already normalized
        rename_map = {}
        for col in out.columns:
            c = str(col).strip().lower()

            if c in ["well", "well name", "well_name", "welll name", "lease", "uwi", "api"]:
                rename_map[col] = "WELL"
            elif c in ["formation", "top name", "top_name", "marker", "horizon", "zone", "name"]:
                rename_map[col] = "TOP_NAME"
            elif c in ["top", "top (ft)", "top depth", "top_depth", "depth", "md"]:
                rename_map[col] = "TOP_DEPTH"

        out = out.rename(columns=rename_map)

        required = {"WELL", "TOP_NAME", "TOP_DEPTH"}
        if not required.issubset(set(out.columns)):
            return pd.DataFrame()

        out = out[["WELL", "TOP_NAME", "TOP_DEPTH"]].copy()
        out["WELL"] = out["WELL"].astype(str).str.strip()
        out["TOP_NAME"] = out["TOP_NAME"].astype(str).str.strip()
        out["TOP_DEPTH"] = pd.to_numeric(out["TOP_DEPTH"], errors="coerce")
        out = out.dropna(subset=["TOP_DEPTH"]).copy()
        out = out.sort_values("TOP_DEPTH").reset_index(drop=True)

        return out

    # ------------------------------------------------------------------
    # Convenience file loading
    # ------------------------------------------------------------------
    def load_tops_file(self):
        """
        Convenience loader.
        Preferred approach is still to load tops through LoadDataPanel,
        but this keeps the button working and stores data in the same format.
        """
        self._ensure_state_fields()
        state = self._state()

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Tops File",
            "",
            "Data Files (*.csv *.xlsx *.xls)"
        )
        if not file_path:
            return

        try:
            ext = Path(file_path).suffix.lower()

            if ext == ".csv":
                try:
                    df = pd.read_csv(file_path)
                except UnicodeDecodeError:
                    df = pd.read_csv(file_path, encoding="latin1")
            elif ext in [".xlsx", ".xls"]:
                df = pd.read_excel(file_path)
            else:
                QMessageBox.warning(
                    self,
                    "Unsupported File",
                    f"Unsupported file type: {ext}"
                )
                return

            df.columns = [str(c).strip() for c in df.columns]

            # Normalize like LoadDataPanel
            normalized = self._normalize_uploaded_tops(df)

            state.tops_df = normalized.copy()

            filtered = self._fallback_filter_from_all_tops(normalized)
            state.current_tops_df = filtered.copy() if filtered is not None else pd.DataFrame()

            state.tops_file = file_path

            self.refresh_from_state()
            self._safe_call("refresh_plots")
            self._safe_call("refresh_ui")

        except Exception as e:
            QMessageBox.critical(self, "Load Tops Failed", str(e))

    def _normalize_uploaded_tops(self, tops_df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize arbitrary uploaded tops file into:
            WELL
            TOP_NAME
            TOP_DEPTH
        """
        rename_map = {}

        for col in tops_df.columns:
            c = str(col).strip()
            c_low = c.lower()

            if c_low in [
                "well",
                "well name",
                "well_name",
                "welll name",
                "well no",
                "well number",
                "lease",
                "uwi",
                "api",
                "holename",
                "hole_name",
                "borehole",
            ]:
                rename_map[col] = "WELL"

            elif c_low in [
                "formation",
                "top name",
                "top_name",
                "marker",
                "horizon",
                "zone",
                "name",
            ]:
                rename_map[col] = "TOP_NAME"

            elif c_low in [
                "top",
                "top (ft)",
                "top depth",
                "top_depth",
                "depth",
                "md",
                "measured depth",
            ]:
                rename_map[col] = "TOP_DEPTH"

        tops_df = tops_df.rename(columns=rename_map)

        required = ["WELL", "TOP_NAME", "TOP_DEPTH"]
        missing = [c for c in required if c not in tops_df.columns]
        if missing:
            raise ValueError(
                f"Tops file must contain columns equivalent to WELL, TOP_NAME, TOP_DEPTH. Missing: {missing}"
            )

        tops_df = tops_df[required].copy()
        tops_df["WELL"] = tops_df["WELL"].astype(str).str.strip()
        tops_df["TOP_NAME"] = tops_df["TOP_NAME"].astype(str).str.strip()
        tops_df["TOP_DEPTH"] = pd.to_numeric(tops_df["TOP_DEPTH"], errors="coerce")

        tops_df = tops_df.dropna(subset=["TOP_DEPTH"]).copy()
        tops_df = tops_df[tops_df["WELL"].str.len() > 0].copy()
        tops_df = tops_df[tops_df["TOP_NAME"].str.len() > 0].copy()
        tops_df = tops_df.sort_values(["WELL", "TOP_DEPTH"]).reset_index(drop=True)

        return tops_df

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------




    
    
    
    
    def auto_apply_zone_from_state(self):
        """
        Apply saved ZoI directly from state without requiring table row selection.
        Used when opening a saved .petro well.
        """
        state = self._state()
    
        ztop = getattr(state, "zoi_top", None)
        zbase = getattr(state, "zoi_base", None)
    
        if ztop is None or zbase is None:
            zrange = getattr(state, "zoi_depth_range", None)
            if not isinstance(zrange, (list, tuple)) or len(zrange) != 2:
                return
            try:
                ztop = float(zrange[0])
                zbase = float(zrange[1])
            except Exception:
                return
    
        try:
            ztop = float(ztop)
            zbase = float(zbase)
        except Exception:
            return
    
        top_depth = min(ztop, zbase)
        base_depth = max(ztop, zbase)
    
        # restore padding if available
        try:
            above = getattr(state, "zoi_above_pad", None)
            below = getattr(state, "zoi_below_pad", None)
            if above is not None:
                self.above_pad.setValue(float(above))
            if below is not None:
                self.below_pad.setValue(float(below))
        except Exception:
            pass
    
        # write exactly the same state fields manual apply would affect
        state.zoi_top = top_depth
        state.zoi_base = base_depth
    
        state.depth_top = top_depth
        state.depth_base = base_depth
        state.plot_top = top_depth
        state.plot_bottom = base_depth
        state.zoi_depth_range = (top_depth, base_depth)
    
        self._safe_call("rebuild_view")
        self._safe_call("refresh_plots")
        self._safe_call("refresh_ui")
    
        self.info.setText(
            f"Zone auto-applied for current well: {top_depth:.2f} ft to {base_depth:.2f} ft"
        )
    
    
    
    
    
    
    




     



    def apply_selected_zone(self):
        sel = self.table.selectionModel().selectedRows()
        rows = sorted({ix.row() for ix in sel})
    
        if not rows:
            QMessageBox.information(self, "Apply Zone", "Select one or more tops first.")
            return
    
        picked = [
            self._visible_intervals[r]
            for r in rows
            if 0 <= r < len(self._visible_intervals)
        ]
    
        if not picked:
            QMessageBox.warning(self, "Apply Zone", "No valid visible tops selected.")
            return
    
        raw_top = min(x.top for x in picked)
        raw_base = max(x.base for x in picked)
    
        above_ft = float(self.above_pad.value())
        below_ft = float(self.below_pad.value())
    
        top_depth = raw_top - above_ft
        base_depth = raw_base + below_ft
    
        state = self._state()
    
        # store raw interval too
        state.zoi_top_raw = raw_top
        state.zoi_base_raw = raw_base
        state.zoi_above_pad = above_ft
        state.zoi_below_pad = below_ft
    
        # Preferred names
        state.zoi_top = top_depth
        state.zoi_base = base_depth
    
        # Compatibility with older plot code
        state.depth_top = top_depth
        state.depth_base = base_depth
        state.plot_top = top_depth
        state.plot_bottom = base_depth
        state.zoi_depth_range = (top_depth, base_depth)
    
        self._safe_call("rebuild_view")
        self._safe_call("refresh_plots")
        self._safe_call("refresh_ui")
    
        self.info.setText(
            f"Zone applied for current well: {top_depth:.2f} ft to {base_depth:.2f} ft "
            f"(raw tops: {raw_top:.2f} to {raw_base:.2f}; "
            f"pad: +{above_ft:.2f} ft above, +{below_ft:.2f} ft below)"
        )

    def clear_zone(self):
        state = self._state()
    
        state.zoi_top = None
        state.zoi_base = None
    
        state.depth_top = None
        state.depth_base = None
        state.plot_top = None
        state.plot_bottom = None
        state.zoi_depth_range = None
    
        state.zoi_top_raw = None
        state.zoi_base_raw = None
        state.zoi_above_pad = None
        state.zoi_below_pad = None
    
        self._safe_call("rebuild_view")
        self._safe_call("refresh_plots")
        self._safe_call("refresh_ui")
    
        self.info.setText("Zone cleared.")
        self.table.clearSelection()
    




    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _safe_call(self, method_name: str):
        fn = getattr(self.controller, method_name, None)
        if callable(fn):
            fn()






















