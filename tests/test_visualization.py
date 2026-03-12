import os
import torch
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for headless testing

import pytorch_volumetric as pv

TEST_DIR = os.path.dirname(__file__)


def test_draw_sdf_slice_smoke():
    """Smoke test that draw_sdf_slice runs without error (no display)."""
    sdf = pv.SphereSDF(1.0)

    # Slice along z=0
    query_range = [(-1.5, 1.5), (-1.5, 1.5), (0.0, 0.0)]
    result = pv.draw_sdf_slice(sdf, query_range, resolution=0.1, do_plot=False)

    sdf_val, sdf_grad, pts, ax, cset1, cset2, v = result
    assert sdf_val is not None
    assert sdf_grad is not None
    assert pts.shape[-1] == 3
    # ax, cset1, cset2 should be None when do_plot=False
    assert ax is None


def test_draw_sdf_slice_with_plot():
    """Test draw_sdf_slice with plotting enabled (Agg backend, no display)."""
    import matplotlib.pyplot as plt
    plt.figure()

    sdf = pv.SphereSDF(1.0)
    query_range = [(-1.5, 1.5), (-1.5, 1.5), (0.0, 0.0)]
    result = pv.draw_sdf_slice(sdf, query_range, resolution=0.1, do_plot=True)

    sdf_val, sdf_grad, pts, ax, cset1, cset2, v = result
    assert ax is not None
    assert cset1 is not None
    assert cset2 is not None
    plt.close('all')


def test_draw_sdf_slice_requires_single_dim():
    """draw_sdf_slice should raise if no dimension is sliced."""
    sdf = pv.SphereSDF(1.0)
    query_range = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]
    try:
        pv.draw_sdf_slice(sdf, query_range, resolution=0.5)
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        pass


def test_draw_sdf_slice_gradient_field():
    """Test draw_sdf_slice with gradient plotting."""
    import matplotlib.pyplot as plt
    plt.figure()

    sdf = pv.SphereSDF(1.0)
    query_range = [(-1.5, 1.5), (-1.5, 1.5), (0.0, 0.0)]
    result = pv.draw_sdf_slice(sdf, query_range, resolution=0.2, do_plot=True, plot_grad=True)
    sdf_val, sdf_grad, pts, ax, cset1, cset2, v = result
    assert ax is not None
    plt.close('all')
