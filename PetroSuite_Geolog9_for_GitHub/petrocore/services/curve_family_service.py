from __future__ import annotations

from typing import Dict, List
import pandas as pd
import re

import re

def classify_curve_families(df):
    families = {
        "gamma": [],
        "resistivity": [],
        "density": [],
        "neutron": [],
        "sonic": [],
        "porosity": [],
        "nmr": [],
        "other": [],
    }

    for col in df.columns:
        c = str(col).upper().strip()

        if re.search(r'(^|_)(GR_EDTC|GR|SGR|CGR|HCGR|ECGR)($|_)', c):
            families["gamma"].append(col)

        elif re.search(r'(^|_)(RT|ILD|LLD|AT90|AF90|RESD|RDEP|RLA5)($|_)', c):
            families["resistivity"].append(col)

        elif re.search(r'(^|_)(RHOB|RHOZ|DEN|DRHO)($|_)', c):
            families["density"].append(col)

        elif re.search(r'(^|_)(TNPH|NPHI|NPOR)($|_)', c):
            families["neutron"].append(col)

        elif re.search(r'(^|_)(DT|DTC|DTCO|AC)($|_)', c):
            families["sonic"].append(col)

        elif re.search(r'(^|_)(PHIT|PHIE|POR)($|_)', c):
            families["porosity"].append(col)

        elif "NMR" in c:
            families["nmr"].append(col)

        else:
            families["other"].append(col)

    return families