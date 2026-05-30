from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class WorkflowResult:
    df: pd.DataFrame
    outputs: list[str]
    messages: list[str] = field(default_factory=list)


def compute_porosity(df: pd.DataFrame, curve_map: dict[str, str], params: dict[str, Any]) -> WorkflowResult:
    """Core porosity engine. No Qt code belongs here."""
    out = df.copy()
    method = params.get("method", "density_neutron")
    matrix_density = float(params.get("matrix_density", 2.71))
    fluid_density = float(params.get("fluid_density", 1.10))

    rhob_curve = curve_map.get("RHOB", "")
    nphi_curve = curve_map.get("NPHI", "") or curve_map.get("TNPH", "")

    if method == "density_only":
        if not rhob_curve or rhob_curve not in out.columns:
            raise ValueError("No density curve selected.")
        rhob = pd.to_numeric(out[rhob_curve], errors="coerce")
        out["POR_DEN"] = ((matrix_density - rhob) / (matrix_density - fluid_density)).clip(0.0, 0.60)
        return WorkflowResult(out, ["POR_DEN"], ["Density porosity computed."])

    if method == "neutron_only":
        if not nphi_curve or nphi_curve not in out.columns:
            raise ValueError("No neutron curve selected.")
        out["PHIN"] = pd.to_numeric(out[nphi_curve], errors="coerce").clip(0.0, 0.60)
        return WorkflowResult(out, ["PHIN"], ["Neutron porosity computed."])

    if not rhob_curve or rhob_curve not in out.columns:
        raise ValueError("No density curve selected.")
    if not nphi_curve or nphi_curve not in out.columns:
        raise ValueError("No neutron curve selected.")

    rhob = pd.to_numeric(out[rhob_curve], errors="coerce")
    nphi = pd.to_numeric(out[nphi_curve], errors="coerce")

    out["POR_DEN"] = ((matrix_density - rhob) / (matrix_density - fluid_density)).clip(0.0, 0.60)
    out["PHIT_DN"] = np.sqrt((out["POR_DEN"] ** 2 + nphi ** 2) / 2.0).clip(0.0, 0.60)
    return WorkflowResult(out, ["POR_DEN", "PHIT_DN"], ["Density-neutron porosity computed."])
