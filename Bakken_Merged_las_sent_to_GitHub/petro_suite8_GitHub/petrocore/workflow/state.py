from __future__ import annotations

print(">>> LOADING state.py from:", __file__)


from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple

import pandas as pd
from petrocore.models.dataset import Dataset


@dataclass
class WorkflowState:
    # -------------------------
    # Data sources (one or both may be used)
    # -------------------------
    dataset: Optional[Dataset] = None               # preferred (for workflow)
    merged_df: Optional[pd.DataFrame] = None        # optional (merge/QC legacy)

    # -------------------------
    # Active working frame
    # -------------------------
    analysis_df: Optional[pd.DataFrame] = None      # what PlotsPanel reads

    # -------------------------
    # Depth / ZOI
    # -------------------------
    depth_limits: Optional[Tuple[float, float]] = None
    zoi_depth_range: Optional[Tuple[float, float]] = None

    # -------------------------
    # Workflow config + execution
    # -------------------------
    params: Dict[str, Any] = field(default_factory=dict)
    enabled_steps: Dict[str, bool] = field(default_factory=dict)
    registry: Any = None

    # -------------------------
    # Tops overlays (optional)
    # -------------------------
    tops_df: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=["Name", "Depth", "Color"])
    )

    # -------------------------
    # UI helpers
    # -------------------------
    plot_title: str = ""
