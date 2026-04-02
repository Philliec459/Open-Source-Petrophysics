from __future__ import annotations

from typing import Dict, List


def pick_best_curve(
    candidates: List[str],
    priority_list: List[str],
) -> str:
    """
    Pick the best curve from candidates based on the priority list.
    If none match, return the first candidate.
    """
    if not candidates:
        return ""

    candidates_upper = {str(c).upper(): c for c in candidates}

    for preferred in priority_list:
        match = candidates_upper.get(str(preferred).upper())
        if match:
            return match

    return candidates[0]


def pick_family_winners(
    classified_families: Dict[str, List[str]],
    priorities_map: Dict[str, List[str]],
) -> Dict[str, str]:
    winners: Dict[str, str] = {}

    for family_name, candidates in classified_families.items():
        priority_list = priorities_map.get(family_name, [])
        winners[family_name] = pick_best_curve(candidates, priority_list)

    return winners