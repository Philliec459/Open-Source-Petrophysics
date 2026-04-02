# petrocore/workflow/state.py

from dataclasses import dataclass, field
import pandas as pd

@dataclass
class WorkflowState:
    raw_runs: list = field(default_factory=list)
    merged_df: pd.DataFrame | None = None
    full_df: pd.DataFrame | None = None
    analysis_df: pd.DataFrame | None = None
    tops_df: pd.DataFrame | None = None
    params: dict = field(default_factory=dict)

    zoi_top: float | None = None
    zoi_base: float | None = None
    zoi_name: str = ""