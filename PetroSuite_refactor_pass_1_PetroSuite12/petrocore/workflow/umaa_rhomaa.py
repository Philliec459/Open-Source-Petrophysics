from __future__ import annotations

from typing import Any
import pandas as pd

from petrocore.workflow.porosity_engine import WorkflowResult


def compute_umaa_rhomaa(df: pd.DataFrame, curve_map: dict[str, str], params: dict[str, Any]) -> WorkflowResult:
    """Core UMAA/RHOMAA engine. No Qt code belongs here."""
    out = df.copy()
    fluid_density = float(params.get("fluid_density", 1.10))

    rhob_curve = curve_map.get("RHOB", "")
    pef_curve = curve_map.get("PEF", "")
    phit_curve = curve_map.get("PHIT", "")

    if not rhob_curve or rhob_curve not in out.columns:
        raise ValueError("No density curve selected.")
    if not pef_curve or pef_curve not in out.columns:
        raise ValueError("No PEF curve selected.")
    if not phit_curve or phit_curve not in out.columns:
        raise ValueError("No PHIT curve selected.")

    rhob = pd.to_numeric(out[rhob_curve], errors="coerce")
    pef = pd.to_numeric(out[pef_curve], errors="coerce")
    phit = pd.to_numeric(out[phit_curve], errors="coerce")

    out["UMAA"] = (rhob * pef).clip(lower=0.0, upper=30)
    out["RHOMAA"] = ((rhob - phit * fluid_density) / (1 - phit)).clip(lower=1, upper=3.2)
    return WorkflowResult(out, ["UMAA", "RHOMAA"], ["UMAA and RHOMAA computed."])
