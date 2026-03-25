from __future__ import annotations

import io
import json
import re
import zipfile
from pathlib import Path

import pandas as pd

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QMessageBox,
    QFormLayout,
    QGroupBox,
)

try:
    from petrocore.services.curve_set_service import (
        load_curve_set_config,
        classify_curves_by_set,
    )
except Exception:
    load_curve_set_config = None
    classify_curves_by_set = None


class SaveWellPanel(QWidget):
    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self._build_ui()
        self._prefill_from_state()

    def _state(self):
        if hasattr(self.controller, "get_state"):
            return self.controller.get_state()
        return getattr(self.controller, "state", None)

    def _prefill_from_state(self):
        state = self._state()
        if state is None:
            return

        if not hasattr(self, "well_name"):
            return
        if not hasattr(self, "project_name"):
            return

        project_name = getattr(state, "project_name", "") or "Bakken"
        well_name = getattr(state, "well_name", "") or ""

        self.project_name.setText(str(project_name))
        if well_name:
            self.well_name.setText(str(well_name))

    def refresh_from_state(self):
        self._prefill_from_state()



    def _safe_name(self, text: str) -> str:
        text = str(text).strip()
        text = re.sub(r"[\\/:*?\"<>|]+", "_", text)
        text = re.sub(r"\s+", "_", text)
        return text.strip("._") or "Unnamed_Well"

              

    def _build_ui(self):
        layout = QVBoxLayout(self)
    
        title = QLabel("Save Well Project")
        title.setStyleSheet("font-size: 11pt; font-weight: bold;")
        layout.addWidget(title)
    
        box = QGroupBox("Project Settings")
        form = QFormLayout(box)
    
        self.project_name = QLineEdit()
        self.project_name.setText("Bakken")
        self.project_name.setPlaceholderText("Enter project name")
    
        self.well_name = QLineEdit()
        self.well_name.setPlaceholderText("Enter well name")
    
        form.addRow("Project Name:", self.project_name)
        ############form.addRow("Well Name:", self.well_name)
    
        layout.addWidget(box)
    
        self.save_btn = QPushButton("Assign Project Name and then just Save Well (auto)")
        self.save_btn.clicked.connect(self.save_well)
        layout.addWidget(self.save_btn)
    
        layout.addStretch(1)
 
    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _normalize_text(self, value) -> str:
        if value is None:
            return ""
        return str(value).strip().upper()


    def _find_well_column(self, df: pd.DataFrame) -> str | None:
        if df is None or df.empty:
            return None

        preferred = [
            "WELL",
            "WELL_NAME",
            "well_name",
            "WELLNAME",
            "WELL NAME",
            "BOREHOLE",
            "UWI",
            "API",
        ]

        cols_upper = {str(c).strip().upper(): c for c in df.columns}
        for name in preferred:
            if name in cols_upper:
                return cols_upper[name]

        for c in df.columns:
            cu = str(c).strip().upper()
            if "WELL" in cu or "UWI" in cu or "API" in cu:
                return c

        return None

    def _filter_df_to_well(self, df: pd.DataFrame, well_name: str) -> pd.DataFrame:
        if df is None or df.empty:
            return df

        well_col = self._find_well_column(df)
        if well_col is None:
            return df.copy()

        target = self._normalize_text(well_name)
        if not target:
            return df.copy()

        work = df.copy()
        work["_well_norm_"] = work[well_col].map(self._normalize_text)

        exact = work[work["_well_norm_"] == target].copy()
        if not exact.empty:
            return exact.drop(columns=["_well_norm_"], errors="ignore")

        relaxed = work[work["_well_norm_"].str.contains(target, na=False)].copy()
        if not relaxed.empty:
            return relaxed.drop(columns=["_well_norm_"], errors="ignore")

        return df.copy()

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

    def _build_curve_sets_table(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = list(df.columns) if df is not None else []
        records = []

        if load_curve_set_config is not None and classify_curves_by_set is not None:
            try:
                config = load_curve_set_config()
                sets_map = classify_curves_by_set(cols, config)

                assigned = set()
                if isinstance(sets_map, dict):
                    for set_name, curve_list in sets_map.items():
                        if curve_list is None:
                            continue
                        for curve_name in curve_list:
                            curve_name = str(curve_name)
                            records.append(
                                {
                                    "curve_name": curve_name,
                                    "curve_set": str(set_name),
                                }
                            )
                            assigned.add(curve_name)

                for c in cols:
                    c = str(c)
                    if c not in assigned:
                        records.append(
                            {
                                "curve_name": c,
                                "curve_set": "UNASSIGNED",
                            }
                        )

            except Exception:
                for c in cols:
                    records.append(
                        {
                            "curve_name": str(c),
                            "curve_set": "UNKNOWN",
                        }
                    )
        else:
            for c in cols:
                records.append(
                    {
                        "curve_name": str(c),
                        "curve_set": "UNKNOWN",
                    }
                )

        return pd.DataFrame(records)

    def _df_to_bytes(self, df: pd.DataFrame) -> bytes:
        return df.to_csv(index=False).encode("utf-8")

    def _json_to_bytes(self, obj: dict) -> bytes:
        return json.dumps(obj, indent=2).encode("utf-8")

    def _write_petro_file(
        self,
        petro_path: Path,
        project_name: str,
        well_name: str,
        well_df: pd.DataFrame,
        curve_sets_df: pd.DataFrame,
        tops_df: pd.DataFrame | None,
        params: dict,
        depth_limits,
        zoi_depth_range,
    ) -> None:
        manifest = {
            "format": "PetroSuite Well File",
            "version": "1.0",
            "project_name": project_name,
            "well_name": well_name,
            "well_data_file": "well_data.csv",
            "curve_sets_file": "curve_sets.csv",
            "tops_file": "tops.csv" if tops_df is not None and not tops_df.empty else None,
            "params_file": "params.json",
            "depth_limits": list(depth_limits) if depth_limits else None,
            "zoi_depth_range": list(zoi_depth_range) if zoi_depth_range else None,
        }

        with zipfile.ZipFile(petro_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr("well_data.csv", self._df_to_bytes(well_df))
            z.writestr("curve_sets.csv", self._df_to_bytes(curve_sets_df))
            z.writestr("params.json", self._json_to_bytes(params))
            z.writestr("manifest.json", self._json_to_bytes(manifest))

            if tops_df is not None and not tops_df.empty:
                z.writestr("tops.csv", self._df_to_bytes(tops_df))

    # ------------------------------------------------------------------
    # Save logic
    # ------------------------------------------------------------------
    def save_well(self):
        state = self._state()
        if state is None:
            QMessageBox.warning(self, "Save Well", "No state is available.")
            return

        source_df = getattr(state, "analysis_df", None)
        if source_df is None or source_df.empty:
            source_df = getattr(state, "df", None)

        if source_df is None or source_df.empty:
            QMessageBox.warning(self, "Save Well", "No well data is loaded.")
            return

        project = self.project_name.text().strip()
        well = self.well_name.text().strip()

        if not project:
            QMessageBox.warning(self, "Save Well", "Enter a project name.")
            return

        if not well:
            well = getattr(state, "well_name", "") or ""
            well = str(well).strip()

        if not well:
            QMessageBox.warning(self, "Save Well", "Enter a well name.")
            return

        safe_project = self._safe_name(project)
        safe_well = self._safe_name(well)

        project_dir = Path("./PetroSuite_Projects") / safe_project
        wells_dir = project_dir / "wells"
        wells_dir.mkdir(parents=True, exist_ok=True)

        petro_path = wells_dir / f"{safe_well}.petro"

        try:
            # ----------------------------------------------------------
            # Well data
            # ----------------------------------------------------------
            well_df = self._filter_df_to_well(source_df.copy(), well)

            # ----------------------------------------------------------
            # Tops
            # ----------------------------------------------------------
            tops_df = getattr(state, "tops_df", None)
            if tops_df is not None and not tops_df.empty:
                tops_save_df = self._filter_df_to_well(tops_df, well)
            else:
                tops_save_df = None

            # ----------------------------------------------------------
            # Curve set catalog
            # ----------------------------------------------------------
            curve_sets_df = self._build_curve_sets_table(well_df)

            # ----------------------------------------------------------
            # Params
            # ----------------------------------------------------------
            params = getattr(state, "params", {}) or {}

            # ----------------------------------------------------------
            # Depth metadata
            # ----------------------------------------------------------
            depth_limits = getattr(state, "depth_limits", None)
            zoi_depth_range = getattr(state, "zoi_depth_range", None)

            if not depth_limits:
                depth_col = self._find_depth_column(well_df)
                if depth_col is not None:
                    d = pd.to_numeric(well_df[depth_col], errors="coerce").dropna()
                    if not d.empty:
                        depth_limits = [float(d.min()), float(d.max())]

            # ----------------------------------------------------------
            # Write single .petro file
            # ----------------------------------------------------------
            self._write_petro_file(
                petro_path=petro_path,
                project_name=project,
                well_name=well,
                well_df=well_df,
                curve_sets_df=curve_sets_df,
                tops_df=tops_save_df,
                params=params,
                depth_limits=depth_limits,
                zoi_depth_range=zoi_depth_range,
            )

            # ----------------------------------------------------------
            # Update/create project.json at project level
            # ----------------------------------------------------------
            project_json_path = project_dir / "project.json"

            project_manifest = {
                "project_name": project,
                "project_folder": str(project_dir),
                "wells_folder": "wells",
                "well_files": [],
            }

            if project_json_path.exists():
                try:
                    with open(project_json_path, "r", encoding="utf-8") as f:
                        old_manifest = json.load(f)
                    if isinstance(old_manifest, dict):
                        project_manifest.update(old_manifest)
                except Exception:
                    pass

            well_entry = {
                "well_name": well,
                "file": f"wells/{safe_well}.petro",
            }

            existing = project_manifest.get("well_files", [])
            if not isinstance(existing, list):
                existing = []

            existing = [
                item for item in existing
                if not (
                    isinstance(item, dict)
                    and (
                        item.get("well_name") == well
                        or item.get("file") == f"wells/{safe_well}.petro"
                    )
                )
            ]
            existing.append(well_entry)

            project_manifest["project_name"] = project
            project_manifest["project_folder"] = str(project_dir)
            project_manifest["wells_folder"] = "wells"
            project_manifest["well_files"] = existing

            with open(project_json_path, "w", encoding="utf-8") as f:
                json.dump(project_manifest, f, indent=2)

            # keep state in sync
            setattr(state, "project_name", project)
            setattr(state, "well_name", well)

            msg = [
                "Saved well package:",
                str(petro_path),
                "",
                f"Project: {project}",
                f"Well: {well}",
                f"Well rows: {len(well_df)}",
                f"Tops rows: {0 if tops_save_df is None else len(tops_save_df)}",
                f"Curve set rows: {len(curve_sets_df)}",
            ]

            QMessageBox.information(self, "Save Well", "\n".join(msg))

        except Exception as e:
            QMessageBox.critical(
                self,
                "Save Well Error",
                f"Could not save well package.\n\n{e}",
            )    
    
    
            
          
    
  
    
    
    
