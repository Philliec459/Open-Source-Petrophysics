from petrocore.config.curve_aliases import best_curve, family_matches


def test_best_curve_prefers_family_priority_order():
    columns = ["DEPTH", "GR", "RHOZ", "RHOB", "TNPH"]

    assert best_curve(columns, "RHOB") == "RHOZ"
    assert best_curve(columns, "NEUTRON") == "TNPH"


def test_family_matches_returns_priority_order_with_actual_names():
    columns = ["depth", "NPHI", "TNPH_SAN", "TNPH", "RHOB"]

    assert family_matches(columns, "TNPH")[:2] == ["TNPH", "NPHI"]
