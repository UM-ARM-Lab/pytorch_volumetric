import os
import torch
import pytorch_volumetric as pv

TEST_DIR = os.path.dirname(__file__)


def test_cached_sdf_lookup_gt_strategy():
    """Test CachedSDF with LOOKUP_GT_SDF out-of-bounds strategy."""
    d = "cuda" if torch.cuda.is_available() else "cpu"
    obj = pv.MeshObjectFactory(os.path.join(TEST_DIR, "probe.obj"))
    gt_sdf = pv.MeshSDF(obj)
    bb = obj.bounding_box(padding=0.01)

    cached = pv.CachedSDF(
        'probe_gt', resolution=0.005, range_per_dim=bb, gt_sdf=gt_sdf,
        out_of_bounds_strategy=pv.OutOfBoundsStrategy.LOOKUP_GT_SDF,
        device=d, clean_cache=True, cache_path="test_gt_cache.pkl"
    )

    # Query points well outside the bounding box - should fall back to GT
    oob_pts = torch.tensor([[10.0, 10.0, 10.0], [20.0, 20.0, 20.0]], device=d)
    val, grad = cached(oob_pts)

    # Compare against ground truth
    gt_val, gt_grad = gt_sdf(oob_pts)
    assert torch.allclose(val, gt_val, atol=1e-3)

    # Clean up cache file
    os.remove("test_gt_cache.pkl")


def test_cached_sdf_bounding_box_strategy():
    """Test CachedSDF with BOUNDING_BOX out-of-bounds strategy."""
    d = "cuda" if torch.cuda.is_available() else "cpu"
    obj = pv.MeshObjectFactory(os.path.join(TEST_DIR, "probe.obj"))
    gt_sdf = pv.MeshSDF(obj)
    bb = obj.bounding_box(padding=0.01)

    cached = pv.CachedSDF(
        'probe_bb', resolution=0.005, range_per_dim=bb, gt_sdf=gt_sdf,
        out_of_bounds_strategy=pv.OutOfBoundsStrategy.BOUNDING_BOX,
        device=d, clean_cache=True, cache_path="test_bb_cache.pkl"
    )

    # Query points outside the bounding box
    oob_pts = torch.tensor([[10.0, 10.0, 10.0]], device=d)
    val, grad = cached(oob_pts)

    # Bounding box strategy should return positive distance for outside points
    assert val.item() > 0

    # Gradient should point away from bounding box (unit norm)
    grad_norm = torch.linalg.norm(grad, dim=-1)
    assert torch.allclose(grad_norm, torch.ones_like(grad_norm), atol=1e-5)

    # BB strategy should under-approximate the true SDF
    gt_val, _ = gt_sdf(oob_pts)
    assert val.item() <= gt_val.item() + 1e-3

    # Clean up cache file
    os.remove("test_bb_cache.pkl")


def test_cached_sdf_inbound_matches_gt():
    """Test that in-bounds cached queries are close to ground truth."""
    d = "cuda" if torch.cuda.is_available() else "cpu"
    obj = pv.MeshObjectFactory(os.path.join(TEST_DIR, "probe.obj"))
    gt_sdf = pv.MeshSDF(obj)
    bb = obj.bounding_box(padding=0.05)

    resolution = 0.005
    cached = pv.CachedSDF(
        'probe_inbound', resolution=resolution, range_per_dim=bb, gt_sdf=gt_sdf,
        device=d, clean_cache=True, cache_path="test_inbound_cache.pkl"
    )

    # Sample points inside the bounding box
    coords, pts = pv.get_coordinates_and_points_in_grid(0.01, bb, device=d)
    pts = pts[torch.randperm(len(pts))[:200]]

    cached_val, cached_grad = cached(pts)
    gt_val, gt_grad = gt_sdf(pts)

    # Cached values should be close to ground truth (within resolution)
    assert torch.allclose(cached_val, gt_val, atol=resolution * 2)

    os.remove("test_inbound_cache.pkl")
