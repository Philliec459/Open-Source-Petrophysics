# packages/petrocore/petrocore/models/dataset.py

from __future__ import annotations

from dataclasses import dataclass, asdict, field
#from typing import Dict, List, Optional, Any, Iterable
from typing import Dict, List, Optional, Any, Iterable, Tuple


import json
import pandas as pd


@dataclass
class CurveMeta:
    name: str
    units: str = ""
    family: str = ""                 # e.g. "GR", "RHOB", "TNPH", "RT", "NMR"
    source: str = ""                 # e.g. "run0", "merged", filename, etc.
    description: str = ""
    role: str = ""                   # e.g. "key", "aux"
    display: Dict[str, Any] = field(default_factory=dict)  # future: scale, etc.


@dataclass
class Dataset:
    """
    Core data contract shared by both GUIs.
    - data: DataFrame indexed by depth (float), columns are curve mnemonics
    - meta: per-curve metadata
    - families_map: family -> candidate curves in preference order
    - history: list of steps / audit records
    """
    data: pd.DataFrame
    meta: Dict[str, CurveMeta] = field(default_factory=dict)
    families_map: Dict[str, List[str]] = field(default_factory=dict)
    units_map: Dict[str, str] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    name: str = "dataset"





    # -------------------------
    # ZOI / analysis state
    # -------------------------
    zoi_depth_range: Optional[Tuple[float, float]] = None

    @property
    def zoi(self) -> Optional["Dataset"]:
        """Return a sliced Dataset for the current ZOI, or None if not set."""
        if self.zoi_depth_range is None:
            return None
        top, base = self.zoi_depth_range
        return self.slice_depth(top, base)

    def set_zoi(self, top: float, base: float) -> "Dataset":
        """Set the ZOI depth range and return the sliced ZOI Dataset."""
        top, base = float(top), float(base)
        if top > base:
            top, base = base, top
        self.zoi_depth_range = (top, base)
        # return the sliced dataset
        return self.slice_depth(top, base)

    def clear_zoi(self) -> None:
        """Clear ZOI selection."""
        self.zoi_depth_range = None


    def __post_init__(self):
        # Ensure depth index is float and sorted ascending
        self.data = self.data.copy()
        self.data.index = pd.to_numeric(self.data.index, errors="coerce").astype(float)
        self.data = self.data[~self.data.index.isna()].sort_index()

        # If units_map exists, apply to meta if missing
        for c in self.data.columns:
            if c not in self.meta:
                self.meta[c] = CurveMeta(name=c)
            if c in self.units_map and not self.meta[c].units:
                self.meta[c].units = self.units_map[c]

    @property
    def depth(self) -> pd.Index:
        return self.data.index

    def curves(self) -> List[str]:
        return list(self.data.columns)

    def has_curve(self, curve: str) -> bool:
        return curve in self.data.columns

    def add_curve(self, name: str, series: pd.Series, meta: Optional[CurveMeta] = None, overwrite: bool = True):
        if (not overwrite) and (name in self.data.columns):
            raise ValueError(f"Curve already exists: {name}")
        s = pd.to_numeric(series, errors="coerce")
        s = s.reindex(self.data.index)  # align to dataset depth
        self.data[name] = s
        if meta is None:
            meta = CurveMeta(name=name)
        self.meta[name] = meta
        if meta.units:
            self.units_map[name] = meta.units

    def get_family_candidates(self, family: str) -> List[str]:
        return list(self.families_map.get(family, []))

    def first_present(self, candidates: Iterable[str]) -> Optional[str]:
        s = set(self.data.columns)
        for c in candidates:
            if c in s:
                return c
        return None

    def best_curve_for_family(self, family: str) -> Optional[str]:
        return self.first_present(self.get_family_candidates(family))

    def slice_depth(self, top: float, base: float) -> Dataset:
        top, base = (top, base) if top <= base else (base, top)
        df = self.data.loc[(self.data.index >= top) & (self.data.index <= base)].copy()
        out = Dataset(
            data=df,
            meta={k: self.meta[k] for k in df.columns if k in self.meta},
            families_map=self.families_map.copy(),
            units_map=self.units_map.copy(),
            history=self.history.copy(),
            name=self.name,
        )
        return out

    # -------------------------
    # Parquet I/O
    # -------------------------
    def to_parquet(self, path: str):
        """
        Writes:
          - {path} as parquet for curve data
          - {path}.meta.json sidecar for metadata (simple, robust)
        """
        self.data.to_parquet(path, index=True)

        meta_dict = {
            "name": self.name,
            "curves": {k: asdict(v) for k, v in self.meta.items()},
            "families_map": self.families_map,
            "units_map": self.units_map,
            "history": self.history,
        }
        with open(path + ".meta.json", "w", encoding="utf-8") as f:
            json.dump(meta_dict, f, indent=2)

    @staticmethod
    def from_parquet(path: str) -> Dataset:
        df = pd.read_parquet(path)
        # Depth index expected
        if df.index.name is None:
            df.index.name = "DEPT"

        meta_path = path + ".meta.json"
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta_dict = json.load(f)
        except FileNotFoundError:
            meta_dict = {}

        curve_meta = {}
        for k, v in (meta_dict.get("curves", {}) or {}).items():
            curve_meta[k] = CurveMeta(**v)

        return Dataset(
            data=df,
            meta=curve_meta,
            families_map=meta_dict.get("families_map", {}) or {},
            units_map=meta_dict.get("units_map", {}) or {},
            history=meta_dict.get("history", []) or [],
            name=meta_dict.get("name", "dataset"),
        )
