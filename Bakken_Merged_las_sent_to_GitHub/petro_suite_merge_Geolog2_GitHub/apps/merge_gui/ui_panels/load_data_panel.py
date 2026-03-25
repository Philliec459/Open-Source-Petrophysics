from __future__ import annotations

from pathlib import Path
from typing import List
import re

import lasio
import pandas as pd

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QTextEdit,
)

from petrocore.services.curve_set_service import (
    load_curve_set_config,
    classify_curves_by_set,
    auto_pick_hidden_curves,
)

print(">>> LOADING load_data_panel.py from:", __file__)


class LoadDataPanel(QWidget):
    """
    Merge-app load panel.

    Responsibilities
    ----------------
    - Load one or more LAS files
    - Extract / guess current well name from LAS
    - Load one tops file
    - Normalize tops columns
    - Store:
        state.tops_df          -> all tops from file
        state.current_tops_df  -> tops filtered to current LAS well
    - Show a simple list of loaded LAS files
    """

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self._build_ui()

    # -------------------------------------------------------------------------
    # Basic helpers
    # -------------------------------------------------------------------------
    def _state(self):
        return self.controller.get_state()

    def _ensure_state_fields(self):
        state = self._state()

        if not hasattr(state, "loaded_las_files"):
            state.loaded_las_files = []

        if not hasattr(state, "las_runs"):
            state.las_runs = []

        if not hasattr(state, "analysis_df"):
            state.analysis_df = None

        if not hasattr(state, "tops_df"):
            state.tops_df = None

        if not hasattr(state, "current_tops_df"):
            state.current_tops_df = None

        if not hasattr(state, "well_name"):
            state.well_name = None

        if not hasattr(state, "curve_sets"):
            state.curve_sets = {}

        if not hasattr(state, "params") or state.params is None:
            state.params = {}

        if not hasattr(state, "zoi_top"):
            state.zoi_top = None

        if not hasattr(state, "zoi_base"):
            state.zoi_base = None

    # -------------------------------------------------------------------------
    # UI
    # -------------------------------------------------------------------------
    def _build_ui(self):
        layout = QVBoxLayout(self)

        title = QLabel("Load LAS Data")
        layout.addWidget(title)

        subtitle = QLabel(
            "Load one or more LAS files for curve organization, depth QC, merging, and tops."
        )
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        btn_row = QHBoxLayout()

        self.btn_load_las = QPushButton("Load LAS Files")
        self.btn_load_las.clicked.connect(self._on_load_las)

        self.btn_load_tops = QPushButton("Load Tops File")
        self.btn_load_tops.clicked.connect(self._on_load_tops)

        self.btn_clear = QPushButton("Clear Loaded Files")
        self.btn_clear.clicked.connect(self._on_clear)

        btn_row.addWidget(self.btn_load_las)
        btn_row.addWidget(self.btn_load_tops)
        btn_row.addWidget(self.btn_clear)
        btn_row.addStretch()

        layout.addLayout(btn_row)

        layout.addWidget(QLabel("Loaded LAS Files:"))
        self.file_list = QListWidget()
        layout.addWidget(self.file_list, stretch=2)

        layout.addWidget(QLabel("Status / Details:"))
        self.info_box = QTextEdit()
        self.info_box.setReadOnly(True)
        self.info_box.setMinimumHeight(180)
        layout.addWidget(self.info_box, stretch=1)

        layout.addStretch()

    # -------------------------------------------------------------------------
    # Well-name logic
    # -------------------------------------------------------------------------
    @staticmethod
    def normalize_well_name(name) -> str:
        """
        Normalize well names for matching LAS headers to tops files.

        Examples:
        - Rollefstad_Federal_12-3H3
        - 24814 Rollefstad Federal 12-3H3
        -> both become comparable
        """
        if name is None:
            return ""

        s = str(name).strip().upper()

        # Remove leading numeric ID like "24814 "
        s = re.sub(r"^\d+\s*", "", s)

        # Remove common separators
        s = s.replace("_", "")
        s = s.replace("-", "")
        s = s.replace(" ", "")

        return s

    def _extract_well_name_from_las(self, las, file_path: str) -> str:
        """
        Try common LAS well-header fields first, then fall back to scanning
        the full well section, and finally use the file stem.
        """
        candidates = [
            "WELL",
            "WELLNAME",
            "WELL NAME",
            "LEASE",
            "LOC",
            "UWI",
            "API",
            "APIN",
            "UWID",
            "NAME",
        ]

        if las is not None and hasattr(las, "well"):
            for key in candidates:
                try:
                    item = las.well.get(key)
                    if item is not None:
                        value = getattr(item, "value", None)
                        if value is not None:
                            s = str(value).strip()
                            if s and s.lower() not in ("none", "nan"):
                                return s
                except Exception:
                    pass

            try:
                for item in las.well:
                    try:
                        mnemonic = str(getattr(item, "mnemonic", "")).strip().upper()
                        descr = str(getattr(item, "descr", "")).strip().upper()
                        value = getattr(item, "value", None)

                        if value is None:
                            continue

                        s = str(value).strip()
                        if not s or s.lower() in ("none", "nan"):
                            continue

                        if mnemonic in candidates:
                            return s

                        if any(tag in descr for tag in ["WELL", "LEASE", "NAME", "UWI", "API"]):
                            return s
                    except Exception:
                        pass
            except Exception:
                pass

        return Path(file_path).stem.strip()

    def _guess_current_well_name(self):
        state = self._state()

        if getattr(state, "well_name", None):
            return str(state.well_name).strip()

        las_runs = getattr(state, "las_runs", [])
        if not las_runs:
            return None

        run0 = las_runs[0]
        las = run0.get("las", None)
        file_name = run0.get("file_path", run0.get("file_name", ""))

        guessed = self._extract_well_name_from_las(las, file_name)
        return guessed.strip() if guessed else None

    # -------------------------------------------------------------------------
    # Tops normalization / filtering
    # -------------------------------------------------------------------------
    def _normalize_tops_df(self, tops_df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize user tops file into:
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
            ]:
                rename_map[col] = "TOP_DEPTH"

        tops_df = tops_df.rename(columns=rename_map)

        required = ["WELL", "TOP_NAME", "TOP_DEPTH"]
        missing = [c for c in required if c not in tops_df.columns]
        if missing:
            raise ValueError(
                "Tops file must contain columns equivalent to: "
                f"Well Name, Formation, Top. Missing: {missing}"
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

    def _filter_tops_to_current_well(self, tops_df: pd.DataFrame) -> pd.DataFrame:
        """
        Return tops matching the active LAS well name.

        First tries exact normalized match.
        Then uses a soft contains match as a fallback.
        """
        if tops_df is None or tops_df.empty:
            return pd.DataFrame(columns=["WELL", "TOP_NAME", "TOP_DEPTH"])

        current_well = self._guess_current_well_name()
        if not current_well:
            return pd.DataFrame(columns=tops_df.columns)

        las_well_norm = self.normalize_well_name(current_well)

        work_df = tops_df.copy()
        work_df["WELL_NORM"] = work_df["WELL"].apply(self.normalize_well_name)

        # Exact normalized match
        tops_this_well = work_df.loc[work_df["WELL_NORM"] == las_well_norm].copy()

        # Fallback partial match
        if tops_this_well.empty and las_well_norm:
            contains_mask = work_df["WELL_NORM"].str.contains(las_well_norm, na=False)
            reverse_mask = work_df["WELL_NORM"].apply(
                lambda x: isinstance(x, str) and x in las_well_norm if las_well_norm else False
            )
            tops_this_well = work_df.loc[contains_mask | reverse_mask].copy()

        tops_this_well = tops_this_well.drop(columns=["WELL_NORM"], errors="ignore")
        tops_this_well = tops_this_well.sort_values("TOP_DEPTH").reset_index(drop=True)

        return tops_this_well

    def _recompute_current_tops(self):
        """
        Recompute the filtered tops for the current well and store them in state.
        """
        state = self._state()

        all_tops = getattr(state, "tops_df", None)
        if not isinstance(all_tops, pd.DataFrame) or all_tops.empty:
            state.current_tops_df = pd.DataFrame(columns=["WELL", "TOP_NAME", "TOP_DEPTH"])
            return

        filtered = self._filter_tops_to_current_well(all_tops)
        state.current_tops_df = filtered

    # -------------------------------------------------------------------------
    # LAS loading
    # -------------------------------------------------------------------------
    def _on_load_las(self):
        self._ensure_state_fields()
        state = self._state()

        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select LAS files",
            "",
            "LAS Files (*.las *.LAS);;All Files (*)",
        )

        if not file_paths:
            return

        added = 0
        messages: List[str] = []

        for fp in file_paths:
            try:
                las = lasio.read(fp)
                df = las.df().reset_index()


















                # Standardize depth column name
                if len(df.columns) > 0:
                    first_col = str(df.columns[0]).strip().upper()
                    if first_col in ["DEPTH", "DEPT", "MD"]:
                        df = df.rename(columns={df.columns[0]: "DEPT"})

                well_name = self._extract_well_name_from_las(las, fp)

                run_record = {
                    "file_path": fp,
                    "file_name": Path(fp).name,
                    "las": las,
                    "df": df,
                    "curves": list(df.columns),
                    "well_name": well_name,
                }

                state.las_runs.append(run_record)
                state.loaded_las_files.append(fp)

                # Active well name follows first/active load model for now
                state.well_name = well_name

                added += 1
                messages.append(
                    f"Loaded: {Path(fp).name} | well={well_name} | rows={len(df):,} | curves={len(df.columns)}"
                )

            except Exception as e:
                messages.append(f"FAILED: {Path(fp).name} | {e}")

        if state.las_runs:
            state.analysis_df = state.las_runs[0]["df"].copy()

            # Build curve sets / auto-hidden picks
            try:
                yaml_path = (
                    Path(__file__).resolve().parents[3]
                    / "petrocore"
                    / "config"
                    / "curve_sets.yaml"
                )
                config = load_curve_set_config(yaml_path)

                curve_names = list(state.analysis_df.columns)
                state.curve_sets = classify_curves_by_set(curve_names, config)

                hidden_picks = auto_pick_hidden_curves(state.curve_sets, config)
                state.params.update(hidden_picks)

            except Exception as e:
                messages.append(f"Curve set classification warning: {e}")

            # If tops were already loaded, re-filter now that well name is known
            self._recompute_current_tops()

            current_tops = getattr(state, "current_tops_df", None)
            if isinstance(current_tops, pd.DataFrame):
                if current_tops.empty:
                    messages.append(f"No tops matched current LAS well: {state.well_name}")
                else:
                    messages.append(
                        f"Filtered existing tops to current LAS well: {state.well_name} | rows={len(current_tops)}"
                    )

        self._refresh_file_list()
        self._append_info("\n".join(messages))

        if added > 0:
            guessed_well = self._guess_current_well_name()
            self._append_info(f"\nSuccessfully added {added} LAS file(s).")
            self._append_info(f"Current LAS well guess: {guessed_well}")
            self.controller.refresh_ui()

    # -------------------------------------------------------------------------
    # Tops loading
    # -------------------------------------------------------------------------
    def _on_load_tops(self):
        self._ensure_state_fields()
        state = self._state()

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Tops File",
            "",
            "Excel Files (*.xlsx *.xls);;CSV Files (*.csv);;Text Files (*.txt);;All Files (*)",
        )

        if not file_path:
            return

        try:
            suffix = Path(file_path).suffix.lower()

            if suffix in [".xlsx", ".xls"]:
                tops_df = pd.read_excel(file_path)
            elif suffix == ".csv":
                tops_df = pd.read_csv(file_path)
            else:
                tops_df = pd.read_csv(file_path, sep=None, engine="python")

            tops_df = self._normalize_tops_df(tops_df)

            # Store all tops
            state.tops_df = tops_df.copy()

            nrows = len(tops_df)
            nwells = tops_df["WELL"].nunique()

            self._append_info(
                f"Loaded tops file: {Path(file_path).name} | rows={nrows:,} | wells={nwells}"
            )

            current_well = self._guess_current_well_name()
            self._append_info(f"Current LAS well guess: {current_well}")

            # Store filtered tops for current well
            self._recompute_current_tops()
            filtered_tops = state.current_tops_df

            if current_well:
                if isinstance(filtered_tops, pd.DataFrame) and not filtered_tops.empty:
                    self._append_info(
                        f"Filtered tops to LAS well: {current_well} | rows={len(filtered_tops)}"
                    )
                    self._append_info(filtered_tops.to_string(index=False))
                else:
                    self._append_info(f"No tops matched LAS well '{current_well}'.")
            else:
                self._append_info(
                    "No current LAS well available yet. Tops file loaded, waiting for LAS."
                )

            self.controller.refresh_ui()

        except Exception as e:
            QMessageBox.critical(self, "Load Tops Error", str(e))
            self._append_info(f"FAILED tops load: {Path(file_path).name} | {e}")

    # -------------------------------------------------------------------------
    # Clear
    # -------------------------------------------------------------------------
    def _on_clear(self):
        state = self._state()

        state.loaded_las_files = []
        state.las_runs = []
        state.analysis_df = None
        state.tops_df = None
        state.current_tops_df = None
        state.well_name = None
        state.zoi_top = None
        state.zoi_base = None

        self.file_list.clear()
        self.info_box.clear()

        self._append_info("Cleared loaded LAS files and tops.")
        self.controller.refresh_ui()

    # -------------------------------------------------------------------------
    # Display helpers
    # -------------------------------------------------------------------------
    def _refresh_file_list(self):
        state = self._state()
        self.file_list.clear()

        for i, run in enumerate(getattr(state, "las_runs", []), start=1):
            name = run.get("file_name", f"Run {i}")
            df = run.get("df", None)
            well_name = run.get("well_name", "")

            nrows = len(df) if isinstance(df, pd.DataFrame) else 0
            ncols = len(df.columns) if isinstance(df, pd.DataFrame) else 0

            text = f"{i}. {name}   |   well={well_name}   |   rows={nrows:,}   curves={ncols}"
            item = QListWidgetItem(text)
            self.file_list.addItem(item)

    def _append_info(self, text: str):
        if text:
            self.info_box.append(text)

    # -------------------------------------------------------------------------
    # Panel refresh
    # -------------------------------------------------------------------------
    def refresh(self):
        self._ensure_state_fields()
        self._refresh_file_list()