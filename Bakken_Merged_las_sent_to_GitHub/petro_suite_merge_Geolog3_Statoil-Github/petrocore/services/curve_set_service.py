import re
import yaml


def load_curve_set_config(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def classify_curves_by_set(cols, config):
    """
    Returns:
        {
            "WIRE": [...],
            "NMR": [...],
            ...
        }
    """
    sets_cfg = config.get("sets", {})
    out = {}

    assigned = set()

    for set_name, set_def in sets_cfg.items():
        patterns = set_def.get("patterns", [])
        matched = []

        for col in cols:
            if col in assigned:
                continue

            for pat in patterns:
                if re.search(pat, str(col), flags=re.IGNORECASE):
                    matched.append(col)
                    assigned.add(col)
                    break

        if matched:
            out[set_name] = matched

    return out


def classify_curves_with_winners(cols, config):
    """
    Returns:
        sets_map: grouped curves by set
        winners_map: preferred default curve per set if found
    """
    sets_cfg = config.get("sets", {})
    sets_map = classify_curves_by_set(cols, config)
    winners_map = {}

    for set_name, matched_cols in sets_map.items():
        set_def = sets_cfg.get(set_name, {})
        winners = set_def.get("winners", [])

        matched_upper = {str(c).upper(): c for c in matched_cols}
        chosen = None

        # exact match first
        for w in winners:
            wu = str(w).upper()
            if wu in matched_upper:
                chosen = matched_upper[wu]
                break

        # fallback: startswith match
        if chosen is None:
            for w in winners:
                wu = str(w).upper()
                for c in matched_cols:
                    if str(c).upper().startswith(wu):
                        chosen = c
                        break
                if chosen is not None:
                    break

        # fallback: first curve in set
        if chosen is None and matched_cols:
            chosen = matched_cols[0]

        if chosen is not None:
            winners_map[set_name] = chosen

    return sets_map, winners_map


def pick_winner_curve(cols, preferred_names):
    """
    Pick the best curve from available columns based on preferred names.
    """
    if not cols:
        return None

    cols_upper = {str(c).upper(): c for c in cols}

    # exact match first
    for name in preferred_names:
        nu = str(name).upper()
        if nu in cols_upper:
            return cols_upper[nu]

    # startswith fallback
    for name in preferred_names:
        nu = str(name).upper()
        for c in cols:
            if str(c).upper().startswith(nu):
                return c

    return None


def auto_pick_hidden_curves(cols):
    """
    Return a list of curves that are usually hidden from the main display
    because they are metadata, QC, duplicates, or less commonly used
    auxiliary curves.
    """
    if not cols:
        return []

    hidden = []
    for c in cols:
        cu = str(c).upper()

        if (
            cu.startswith("AF") or
            cu.startswith("AO") or
            cu.startswith("AT") or
            cu.startswith("CBP") or
            cu in {
                "BHF", "BMF", "CDF", "CFTC", "CNTC", "CTEM",
                "DCAL", "DRHO", "DSOZ", "MUDT", "TT1", "TT2"
            } or
            cu.endswith("_SIG") or
            "QUALITY" in cu or
            "QC" in cu
        ):
            hidden.append(c)

    return hidden