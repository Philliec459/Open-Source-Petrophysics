from __future__ import annotations

from typing import Iterable
import re

from petrocore.config.curve_aliases import (
    CURVE_FAMILIES,
    canonical_family,
    family_candidates,
    family_matches,
    best_curve,
    first_present,
)


def classify_curve_families(df_or_columns):
    """
    Classify curves using the central curve alias table.

    Returns a legacy-friendly dictionary with lowercase family labels, while also
    ensuring the matching logic comes from one source of truth.
    """



    cols_obj = getattr(df_or_columns, "columns", df_or_columns)

    if cols_obj is None:
        cols = []
    else:
        cols = list(cols_obj)





    out = {
        "gamma": family_matches(cols, "GR"),
        "resistivity": family_matches(cols, "RT"),
        "density": family_matches(cols, "RHOB"),
        "neutron": family_matches(cols, "TNPH"),
        "sonic": family_matches(cols, "DTCO"),
        "porosity": family_matches(cols, "PHIT"),
        "nmr"     : family_matches(cols, "NMRPHIT"),
        "other": [],
    }

    assigned = {c for vals in out.values() for c in vals}
    for c in cols:
        cu = str(c).upper().strip()
        if c in assigned:
            continue
        if "NMR" in cu or cu in {"PHIT_NMR", "PHIE_NMR", "T2LM", "BVIE", "CBW"}:
            out["nmr"].append(c)
            assigned.add(c)
        else:
            out["other"].append(c)
    return out


# Explicit exports used by UI code during the transition.
__all__ = [
    "CURVE_FAMILIES",
    "canonical_family",
    "family_candidates",
    "family_matches",
    "best_curve",
    "first_present",
    "classify_curve_families",
]
