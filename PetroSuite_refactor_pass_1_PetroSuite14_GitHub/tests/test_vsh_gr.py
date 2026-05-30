import numpy as np

from petrocore.workflow.vsh import compute_vsh_gr


def test_compute_vsh_gr():
    gr = np.array([50.0, 100.0, 150.0])

    vsh = compute_vsh_gr(gr, gr_clean=50.0, gr_shale=150.0)

    np.testing.assert_allclose(vsh, [0.0, 0.5, 1.0])