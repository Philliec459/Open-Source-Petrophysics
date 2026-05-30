import numpy as np

from petrocore.workflow.sw import compute_sw_waxman_smits


def test_compute_sw_waxman_smits_returns_bounded_values():
    rt = np.array([5.0, 10.0, 25.0])
    phi = np.array([0.12, 0.18, 0.25])
    qv = np.array([0.1, 0.2, 0.3])

    sw = compute_sw_waxman_smits(rt, phi, Rw=0.08, B=0.5, Qv=qv)

    assert sw.shape == rt.shape
    assert np.all(np.isfinite(sw))
    assert np.nanmin(sw) >= 0.0
    assert np.nanmax(sw) <= 1.0


def test_compute_sw_waxman_smits_marks_invalid_inputs_nan():
    rt = np.array([10.0, np.nan, 20.0])
    phi = np.array([0.2, 0.2, 0.0])
    qv = np.array([0.1, 0.1, 0.1])

    sw = compute_sw_waxman_smits(rt, phi, Rw=0.08, B=0.5, Qv=qv)

    assert np.isfinite(sw[0])
    assert np.isnan(sw[1])
    assert np.isnan(sw[2])
