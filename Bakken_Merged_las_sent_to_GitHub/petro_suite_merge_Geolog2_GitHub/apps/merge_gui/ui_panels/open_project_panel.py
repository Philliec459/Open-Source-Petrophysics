from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QTextEdit,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QComboBox,
)

from apps.merge_gui.ui_panels.load_data_panel import LoadDataPanel



# Optional curve-set support
try:
    from petrocore.services.curve_set_service import (
        load_curve_set_config,
        classify_curves_by_set,
    )
except Exception:
    load_curve_set_config = None
    classify_curves_by_set = None





class OpenProjectPanel(QWidget):
    """
    Open a PetroSuite project folder and load one saved .petro well package.

    Expected structure:
        PetroSuite_Projects/
            MyProject/
                project.json
                wells/
                    Well_A.petro
                    Well_B.petro
    """

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller

        self.project_dir: Path | None = None
        self.project_json_path: Path | None = None


  

        self.project_meta: dict = {}
        self.well_file_map: dict[str, Path] = {}

        self._build_ui()

    # ------------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------------
    def _state(self):
        if hasattr(self.controller, "get_state"):
            return self.controller.get_state()
        return getattr(self.controller, "state", None)

    def _build_ui(self):
        layout = QVBoxLayout(self)

        title = QLabel("Open Project")
        title.setStyleSheet("font-size: 11pt; font-weight: bold;")
        layout.addWidget(title)

        box = QGroupBox("Project")
        form = QFormLayout(box)

        self.project_folder_edit = QLineEdit()
        self.project_folder_edit.setReadOnly(True)
        self.project_folder_edit.setPlaceholderText("Select a saved PetroSuite project folder")

        self.project_name_edit = QLineEdit()
        self.project_name_edit.setReadOnly(True)

        self.well_combo = QComboBox()
        self.well_combo.setEnabled(False)
        self.well_combo.currentIndexChanged.connect(self._well_changed)

        ########form.addRow("Project Folder:", self.project_folder_edit)
        form.addRow("Project Name:", self.project_name_edit)
        form.addRow("Well Name:", self.well_combo)

        layout.addWidget(box)

        self.btn_browse = QPushButton("1) Choose Project from PetroSuite_Projects")
        self.btn_browse.clicked.connect(self._browse_project_folder)
        layout.addWidget(self.btn_browse)


        self.btn_open = QPushButton("2) Select a Well from that Project in the box above and CLICK HERE to load it")
        self.btn_open.clicked.connect(self._open_project)
        layout.addWidget(self.btn_open)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        layout.addWidget(self.status_label)

        layout.addStretch()

    def _set_status(self, text: str):
        self.status_label.setText(text)

    def _safe_read_json(self, path: Path) -> dict:
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    def _read_csv_from_zip(self, z: zipfile.ZipFile, member_name: str) -> Optional[pd.DataFrame]:
        try:
            with z.open(member_name) as f:
                return pd.read_csv(f)
        except Exception:
            return None

    def _read_json_from_zip(self, z: zipfile.ZipFile, member_name: str) -> dict:
        try:
            with z.open(member_name) as f:
                obj = json.load(f)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    def _find_depth_column(self, df: pd.DataFrame) -> str | None:
        if df is None or df.empty:
            return None

        preferred = ["DEPT", "DEPTH", "MD", "TVD", "TVDSS"]
        cols_upper = {str(c).strip().upper(): c for c in df.columns}

        for name in preferred:
            if name in cols_upper:
                return cols_upper[name]

        for c in df.columns:
            cu = str(c).strip().upper()
            if "DEPTH" in cu or cu == "DEPT":
                return c

        return None

    def _find_top_base_columns(self, df: pd.DataFrame) -> tuple[str | None, str | None]:
        if df is None or df.empty:
            return None, None

        cols_upper = {str(c).strip().upper(): c for c in df.columns}

        top_candidates = ["TOP", "TOPDEPTH", "TOP_DEPTH", "TOP (FT)", "DEPTH"]
        base_candidates = ["BASE", "BASEDEPTH", "BASE_DEPTH", "BOTTOM", "BOTTOM (FT)"]

        top_col = None
        base_col = None


        for c in top_candidates:
            if c in cols_upper:
                top_col = cols_upper[c]
                break

        for c in base_candidates:
            if c in cols_upper:
                base_col = cols_upper[c]
                break

        return top_col, base_col

    def _rebuild_curve_sets_from_df(self, df: pd.DataFrame):
        state = self._state()
        if state is None:
            return

        cols = list(df.columns) if df is not None else []

        if load_curve_set_config is not None and classify_curves_by_set is not None:
            try:
                config = load_curve_set_config()
                sets_map = classify_curves_by_set(cols, config)
                state.curve_sets = sets_map
                return
            except Exception:
                pass

        state.curve_sets = {}

    def _curve_sets_df_to_map(self, curve_sets_df: pd.DataFrame) -> dict:
        if curve_sets_df is None or curve_sets_df.empty:
            return {}

        curve_col = None
        set_col = None

        for c in curve_sets_df.columns:
            cu = str(c).strip().lower()
            if cu in ("curve_name", "curve", "mnemonic"):
                curve_col = c
            elif cu in ("curve_set", "set", "set_name", "family"):
                set_col = c

        if curve_col is None or set_col is None:
            return {}

        out: dict[str, list[str]] = {}
        for _, row in curve_sets_df.iterrows():
            curve_name = str(row[curve_col]).strip()
            set_name = str(row[set_col]).strip() or "UNKNOWN"
            if not curve_name:
                continue
            out.setdefault(set_name, []).append(curve_name)

        return out

    def _load_petro_file(self, petro_path: Path) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None, dict, dict]:
        """
        Returns:
            well_df, curve_sets_df, tops_df, params, manifest
        """
        with zipfile.ZipFile(petro_path, "r") as z:
            names = set(z.namelist())

            manifest = self._read_json_from_zip(z, "manifest.json") if "manifest.json" in names else {}
            params = self._read_json_from_zip(z, "params.json") if "params.json" in names else {}

            well_df = self._read_csv_from_zip(z, "well_data.csv") if "well_data.csv" in names else None
            curve_sets_df = self._read_csv_from_zip(z, "curve_sets.csv") if "curve_sets.csv" in names else None
            tops_df = self._read_csv_from_zip(z, "tops.csv") if "tops.csv" in names else None

        return well_df, curve_sets_df, tops_df, params, manifest

    # ------------------------------------------------------------------
    # Browse / preload
    # ------------------------------------------------------------------
    def _browse_project_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Project Folder", "")
        if not folder:
            return

        self.project_dir = Path(folder)
        self.project_folder_edit.setText(str(self.project_dir))
        self.project_name_edit.setText(self.project_dir.name)

        self._preload_project_folder()

    def _preload_project_folder(self):
        self.project_json_path = None
        self.project_meta = {}
        self.well_file_map = {}
        self.well_combo.clear()
        self.well_combo.setEnabled(False)

        if self.project_dir is None:
            return

        self.project_json_path = self.project_dir / "project.json"
        if self.project_json_path.exists():
            self.project_meta = self._safe_read_json(self.project_json_path)
        else:
            self.project_meta = {}

        project_name = self.project_meta.get("project_name", self.project_dir.name)
        self.project_name_edit.setText(str(project_name))

        wells_dir_name = self.project_meta.get("wells_folder", "wells")
        wells_dir = self.project_dir / str(wells_dir_name)

        # First preference: use project.json entries
        entries = self.project_meta.get("well_files", [])
        if isinstance(entries, list):
            for item in entries:
                if not isinstance(item, dict):
                    continue
                well_name = str(item.get("well_name", "")).strip()
                rel_file = str(item.get("file", "")).strip()
                if not rel_file:
                    continue

                petro_path = self.project_dir / rel_file
                if petro_path.exists() and petro_path.is_file():
                    if not well_name:
                        well_name = petro_path.stem
                    self.well_file_map[well_name] = petro_path

        # Fallback: scan wells directory for *.petro
        if wells_dir.exists() and wells_dir.is_dir():
            for petro_path in sorted(wells_dir.glob("*.petro")):
                well_name = petro_path.stem
                if well_name not in self.well_file_map:
                    self.well_file_map[well_name] = petro_path

        well_names = sorted(self.well_file_map.keys())

        self.well_combo.clear()
        if well_names:
            self.well_combo.addItems(well_names)
            self.well_combo.setEnabled(True)
            self._set_status(
                f"Project found: {project_name}\n"
                f"Well packages found: {len(well_names)}\n"
                f"Select a well and click Open Project."
            )
        else:
            self.well_combo.addItem("")
            self.well_combo.setEnabled(False)
            self._set_status(
                "Project folder found, but no .petro well files were detected."
            )

    def _well_changed(self):
        selected = self.well_combo.currentText().strip()
        petro_path = self.well_file_map.get(selected)

        if selected and petro_path is not None:
            self._set_status(
                f"Project: {self.project_name_edit.text().strip()}\n"
                f"Selected well: {selected}\n"
                f"File: {petro_path.name}"
            )


    

                    

    
    def _on_clear(self):
        state = self._state()
        if state is None:
            return
    
        state.loaded_las_files = []
        state.las_runs = []
        state.df = None
        state.merged_df = None
        state.analysis_df = None
    
        state.tops_df = None
        state.current_tops_df = None
        state.tops = None
    
        state.well_name = None
        state.zoi_top = None
        state.zoi_base = None
        state.zoi_depth_range = None
        state.depth_limits = None
    
        state.curve_sets = {}
        state.curve_sets_df = None
        state.params = {}
    
    
 


    # ------------------------------------------------------------------
    # Open
    # ------------------------------------------------------------------
    def _open_project(self):
        if self.project_dir is None:
            QMessageBox.information(self, "No Project Folder", "Please select a project folder first.")
            return

        selected_well = self.well_combo.currentText().strip()
        if not selected_well:
            QMessageBox.warning(self, "No Well Selected", "Please select a well to open.")
            return

        petro_path = self.well_file_map.get(selected_well)
        if petro_path is None or not petro_path.exists():
            QMessageBox.warning(self, "Missing Well File", "The selected .petro file could not be found.")
            return

        state = self._state()
        if state is None:
            QMessageBox.critical(self, "State Error", "Controller state is not available.")
            return

        '''        
        try:
            self._on_clear()
  
        
            # force widgets (especially Tops) to repaint from empty state
            if hasattr(self.controller, "refresh_ui"):
                self.controller.refresh_ui()
            if hasattr(self.controller, "refresh_plots"):
                self.controller.refresh_plots()
        
            well_df, curve_sets_df, tops_df, params, manifest = self._load_petro_file(petro_path)      
            
        '''

    
            
        try:
            self._on_clear()
        
            if hasattr(self.controller, "refresh_ui"):
                self.controller.refresh_ui()
            if hasattr(self.controller, "refresh_plots"):
                self.controller.refresh_plots()
        
            well_df, curve_sets_df, tops_df, params, manifest = self._load_petro_file(petro_path)   
            
        
        








 
            if well_df is None or well_df.empty:
                QMessageBox.warning(
                    self,
                    "Open Project",
                    "The selected .petro file does not contain valid well data."
                )
                return

            project_name = str(self.project_meta.get("project_name", self.project_dir.name))
            manifest_well_name = str(manifest.get("well_name", "")).strip()
            well_name = manifest_well_name or selected_well

            # ----------------------------------------------------------
            # Push core data into state
            # ----------------------------------------------------------
            state.project_dir = str(self.project_dir)
            state.project_name = project_name
            state.well_name = well_name

            state.df = well_df.copy()
            state.merged_df = well_df.copy()
            state.analysis_df = well_df.copy()

            if tops_df is not None and not tops_df.empty:
                state.tops_df = tops_df.copy()
                state.tops = tops_df.copy()
            else:
                state.tops_df = None
                state.tops = None

            if not hasattr(state, "params") or state.params is None:
                state.params = {}

            if isinstance(params, dict):
                state.params = params.copy()
            else:
                state.params = {}

            # ----------------------------------------------------------
            # Restore depth limits / ZOI
            # ----------------------------------------------------------
            depth_limits = manifest.get("depth_limits")
            zoi_depth_range = manifest.get("zoi_depth_range")

            if isinstance(depth_limits, (list, tuple)) and len(depth_limits) == 2:
                try:
                    state.depth_limits = (float(depth_limits[0]), float(depth_limits[1]))
                except Exception:
                    pass
            else:
                depth_col = self._find_depth_column(well_df)
                if depth_col is not None:
                    d = pd.to_numeric(well_df[depth_col], errors="coerce").dropna()
                    if not d.empty:
                        state.depth_limits = (float(d.min()), float(d.max()))

            if isinstance(zoi_depth_range, (list, tuple)) and len(zoi_depth_range) == 2:
                try:
                    state.zoi_depth_range = (float(zoi_depth_range[0]), float(zoi_depth_range[1]))
                except Exception:
                    pass
            elif tops_df is not None and not tops_df.empty:
                top_col, base_col = self._find_top_base_columns(tops_df)
                if top_col is not None and base_col is not None:
                    try:
                        z_top = pd.to_numeric(tops_df[top_col], errors="coerce").dropna().min()
                        z_base = pd.to_numeric(tops_df[base_col], errors="coerce").dropna().max()
                        if pd.notna(z_top) and pd.notna(z_base):
                            state.zoi_depth_range = (float(z_top), float(z_base))
                    except Exception:
                        pass

            # ----------------------------------------------------------
            # Restore curve sets
            # ----------------------------------------------------------
            curve_sets_map = self._curve_sets_df_to_map(curve_sets_df) if curve_sets_df is not None else {}
            if curve_sets_map:
                state.curve_sets = curve_sets_map
            else:
                self._rebuild_curve_sets_from_df(well_df)

            # Keep original table too if useful elsewhere
            state.curve_sets_df = curve_sets_df.copy() if curve_sets_df is not None else None

            # ----------------------------------------------------------
            # Refresh app
            # ----------------------------------------------------------
            if hasattr(self.controller, "refresh_ui"):
                self.controller.refresh_ui()

            if hasattr(self.controller, "refresh_plots"):
                self.controller.refresh_plots()

            msg = (
                f"Project loaded successfully.\n\n"
                f"Project: {project_name}\n"
                f"Well: {well_name}\n"
                f"File: {petro_path.name}\n"
                f"Well rows: {len(well_df)}\n"
                f"Tops rows: {0 if tops_df is None else len(tops_df)}\n"
                f"Curve sets rows: {0 if curve_sets_df is None else len(curve_sets_df)}"
            )

            QMessageBox.information(self, "Open Project", msg)

        except Exception as e:
            QMessageBox.critical(self, "Open Project Error", f"Could not open project.\n\n{e}")