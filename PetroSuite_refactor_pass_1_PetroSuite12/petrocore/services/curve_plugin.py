from __future__ import annotations

from petrocore.services.curve_config_service import (
    load_curve_families_config,
    load_curve_priorities_config,
    load_curve_sets_config,
)
from petrocore.services.curve_family_service import classify_curve_families
from petrocore.services.curve_winner_service import pick_family_winners
from petrocore.services.curve_set_service import build_curve_sets

class CurvesPlugin:
    def run(self, state):
        df = getattr(state, "analysis_df", None)
        if df is None or df.empty:
            state.curve_families = {}
            state.curve_winners = {}
            state.curve_sets = {}
            return state

        family_map = load_curve_families_config()
        priorities_map = load_curve_priorities_config()
        set_config = load_curve_sets_config()

        families = classify_curve_families(df, family_map)
        winners = pick_family_winners(df, families, priorities_map)
        curve_sets = build_curve_sets(df, winners, set_config)

        state.curve_families = families
        state.curve_winners = winners
        state.curve_sets = curve_sets
        return state