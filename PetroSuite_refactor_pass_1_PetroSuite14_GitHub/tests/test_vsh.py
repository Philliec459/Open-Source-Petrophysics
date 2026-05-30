import numpy as np

from petrocore.workflow.vsh import compute_vsh_gr, compute_vsh_hl


def test_compute_vsh_gr_scales_clips_and_preserves_nan():
    gr = np.array([25.0, 50.0, 100.0, 150.0, 175.0, np.nan])

    vsh = compute_vsh_gr(gr, gr_clean=50.0, gr_shale=150.0)

    np.testing.assert_allclose(vsh[:5], [0.0, 0.0, 0.5, 1.0, 1.0])
    assert np.isnan(vsh[5])


def test_compute_vsh_hl_returns_same_shape_and_bounded_values():
    gr = np.array([50.0, 70.0, 90.0, 110.0, 150.0])

    vsh = compute_vsh_hl(gr, gr_clean=50.0, gr_shale=150.0, window=3)

    assert vsh.shape == gr.shape
    assert np.nanmin(vsh) >= 0.0
    assert np.nanmax(vsh) <= 1.0
