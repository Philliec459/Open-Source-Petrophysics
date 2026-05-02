"""
Single source of truth for curve family aliases and priorities.

Keep all mnemonic preferences here. UI files should import helper functions
instead of defining TNPH/RHOB/GR lists locally.
"""
from __future__ import annotations

from typing import Iterable, Mapping

CURVE_FAMILIES: dict[str, list[str]] = {
    "DEPTH": ["DEPT", "DEPTH", "MD", "TVD", "TVDSS"],
    "GR": ["GR_EDTC", "HSGR", "HCGR", "CGR", "ECGR", "SGR", "GR", "GRC", "HGR", "EGR"],
    "CGR": ["HCGR", "CGR", "ECGR", "GR_EDTC"],
    "SP": ["SP"],
    "CALI": ["HCAL", "CALI", "CALI1", "CAL", "C1", "CALS"],
    "BS": ["BS", "BIT", "BITSIZE"],
    "RHOB": ["RHOZ", "RHOB", "DEN", "DENS", "RHO8", "ZDEN", "RHO"],
    "TNPH": ["TNPH", "NPHI", "NPOR", "CNL", "CNC", "CNCF", "NEUT", "HTNP", "HTNP_SAN", "HNPO"],
    "NPHI": ["TNPH", "NPHI", "NPOR", "CNL", "CNC", "CNCF", "NEUT", "HTNP", "HTNP_SAN", "HNPO"],
    "DTCO": ["DTCO", "DTC", "DT", "AC"],
    "DTSM": ["DTSM", "DTS"],
    "PEF": ["PEF", "PEFZ", "PEF8", "PE"],
    "RT": ["AT90", "AF90", "ILD", "LLD", "RT", "RESD", "RDEP", "RLA5", "AORT", "AORX", "RD"],
    "PHIT": ["PHIT", "PHIE", "PHIT_NMR", "NPHI", "TNPH", "POR", "PHIT_DN", "POR_DEN"],
    "NMRPHIT": ["PHIT_NMR", "CMR_PHI_CONV", "TCMR", "MPHS"],
    "NMRPHIE": ["PHIE_NMR", "CMRP_3MS", "CMRP_3ms", "MPHI"],
    "CBW": ["CBW"],
    "BVIE": ["BVIE", "MBVI"],
    "FFI": ["FFI", "CMFF", "BFV", "BFFV"],
    "KSDR": ["KSDR"],
    "KTIM": ["KTIM"],
    "HFK": ["HFK"],
    "HTHO": ["HTHO"],
    "HURA": ["HURA", "HTUR"],
    "HSGR": ["HSGR"],
    "HCGR": ["HCGR"],
    "HCAL": ["HCAL"],
    "DWAL_WALK2": ["DWAL_WALK2"],
    "DWCA_WALK2": ["DWCA_WALK2"],
    "DWSI_WALK2": ["DWSI_WALK2"],
    "DWSU_WALK2": ["DWSU_WALK2"],
    "DWTI_WALK2": ["DWTI_WALK2"],
}

# Compatibility aliases so older UI keys keep working.
FAMILY_ALIASES: dict[str, str] = {
    "GAMMA": "GR",
    "DENSITY": "RHOB",
    "NEUTRON": "TNPH",
    "SONIC": "DTCO",
    "RESISTIVITY": "RT",
    "POROSITY": "PHIT",
    "NMR": "NMRPHIT",
    "ECS": "DWAL_WALK2",
    "NGS": "HFK",
}


def canonical_family(family: str) -> str:
    key = str(family or "").strip().upper()
    return FAMILY_ALIASES.get(key, key)


def family_candidates(family: str) -> list[str]:
    return list(CURVE_FAMILIES.get(canonical_family(family), []))


def _columns_upper_map(columns: Iterable[str]) -> dict[str, str]:
    return {str(c).strip().upper(): c for c in columns}


def first_present(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    cols = list(columns or [])
    upper = _columns_upper_map(cols)
    for cand in candidates:
        match = upper.get(str(cand).strip().upper())
        if match is not None:
            return match
    return None


def best_curve(columns: Iterable[str], family_or_candidates: str | Iterable[str]) -> str | None:
    """Pick best available curve by exact case-insensitive match, then substring fallback."""
    if isinstance(family_or_candidates, str):
        candidates = family_candidates(family_or_candidates)
    else:
        candidates = list(family_or_candidates or [])

    cols = list(columns or [])
    exact = first_present(cols, candidates)
    if exact is not None:
        return exact

    for cand in candidates:
        cu = str(cand).strip().upper()
        for col in cols:
            if cu and cu in str(col).strip().upper():
                return col
    return None


def family_matches(columns: Iterable[str], family_or_candidates: str | Iterable[str]) -> list[str]:
    """Return all matching curves in priority order, preserving actual column names."""
    if isinstance(family_or_candidates, str):
        candidates = family_candidates(family_or_candidates)
    else:
        candidates = list(family_or_candidates or [])

    cols = list(columns or [])
    upper = _columns_upper_map(cols)
    out: list[str] = []

    for cand in candidates:
        match = upper.get(str(cand).strip().upper())
        if match is not None and match not in out:
            out.append(match)

    for cand in candidates:
        cu = str(cand).strip().upper()
        for col in cols:
            if cu and cu in str(col).strip().upper() and col not in out:
                out.append(col)
    return out


def pick_many(columns: Iterable[str], families: Iterable[str]) -> dict[str, str | None]:
    return {canonical_family(f): best_curve(columns, f) for f in families}
