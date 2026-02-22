from dataclasses import dataclass
from typing import Callable, Dict, List
import pandas as pd

StepFn = Callable[[pd.DataFrame, Dict], pd.DataFrame]

@dataclass
class Step:
    key: str
    title: str
    fn: StepFn
    enabled_by_default: bool = True

class WorkflowRegistry:
    def __init__(self):
        self.steps: List[Step] = []

    def add(self, key: str, title: str, fn: StepFn, enabled_by_default: bool = True):
        self.steps.append(Step(key, title, fn, enabled_by_default))

def run_pipeline(df: pd.DataFrame, params: Dict, enabled: Dict[str, bool], registry: WorkflowRegistry) -> pd.DataFrame:
    out = df.copy()
    for step in registry.steps:
        if not enabled.get(step.key, step.enabled_by_default):
            continue
        out = step.fn(out, params)
    return out
