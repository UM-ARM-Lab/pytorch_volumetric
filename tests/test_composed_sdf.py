"""
Tests for ComposedSDF correctness. Uses SphereSDF primitives to avoid
mesh I/O dependencies, making tests fast and deterministic.
"""
import torch
import pytorch_kinematics as pk
import pytorch_volumetric as pv


def _make_composed_spheres(n_spheres, offsets, device="cpu"):
    """Create a ComposedSDF of spheres at given offsets along the x-axis."""
    sdfs = [pv.SphereSDF(0.5) for _ in range(n_spheres)]
    tsfs = []
    for offset in offsets:
        tsfs.append(pk.Translate(offset, 0, 0, device=device))
    tsf = tsfs[0].stack(*tsfs[1:]) if len(tsfs) > 1 else tsfs[0]
    return pv.ComposedSDF(sdfs, tsf)


# ── Basic value correctness ──────────────────────────────────────────────────

def test_composed_single_sdf_matches_raw():
    """ComposedSDF with one sphere (identity transform) should match SphereSDF directly."""
    d = "cuda" if torch.cuda.is_available() else "cpu"
    sphere = pv.SphereSDF(0.5)
    # Use None transform with set_transforms (as done in test_differentiability)
    composed = pv.ComposedSDF([pv.SphereSDF(0.5)], None)
    composed.set_transforms(pk.Translate(0, 0, 0, device=d), batch_dim=(1,))

    pts = torch.randn(100, 3, device=d)
    v_raw, g_raw = sphere(pts)
    v_comp, g_comp = composed(pts)

    assert torch.allclose(v_raw, v_comp.squeeze(0), atol=1e-5)
    assert torch.allclose(g_raw, g_comp.squeeze(0), atol=1e-5)


def test_composed_takes_min_across_sdfs():
    """ComposedSDF should return the minimum SDF value across all SDFs."""
    d = "cuda" if torch.cuda.is_available() else "cpu"
    # Two spheres at x=-2 and x=+2
    composed = _make_composed_spheres(2, [-2.0, 2.0], device=d)

    # Point near left sphere: should have small SDF from left, large from right
    pt_left = torch.tensor([[-1.6, 0.0, 0.0]], device=d)
    v, g = composed(pt_left)
    v_left, _ = pv.SphereSDF(0.5)(pt_left - torch.tensor([[-2.0, 0, 0]], device=d))
    assert torch.allclose(v.squeeze(), v_left.squeeze(), atol=1e-5)

    # Point near right sphere
    pt_right = torch.tensor([[2.4, 0.0, 0.0]], device=d)
    v, g = composed(pt_right)
    v_right, _ = pv.SphereSDF(0.5)(pt_right - torch.tensor([[2.0, 0, 0]], device=d))
    assert torch.allclose(v.squeeze(), v_right.squeeze(), atol=1e-5)


def test_composed_gradient_matches_closest():
    """Gradient should come from the closest SDF (the one with min value)."""
    d = "cuda" if torch.cuda.is_available() else "cpu"
    # Two spheres far apart
    composed = _make_composed_spheres(2, [-5.0, 5.0], device=d)

    # Point clearly closer to left sphere
    pts = torch.tensor([[-4.0, 0.0, 0.0]], device=d)
    v_comp, g_comp = composed(pts)

    # Should match left sphere's gradient
    pts_in_left_frame = pts - torch.tensor([[-5.0, 0, 0]], device=d)
    _, g_left = pv.SphereSDF(0.5)(pts_in_left_frame)
    assert torch.allclose(g_comp.squeeze(), g_left.squeeze(), atol=1e-5)


# ── Transform correctness ────────────────────────────────────────────────────

def test_composed_with_translation():
    """Test ComposedSDF with a translated sphere."""
    d = "cuda" if torch.cuda.is_available() else "cpu"

    composed = _make_composed_spheres(2, [3.0, -3.0], device=d)

    # Point at (3, 0, 0) should be at the first sphere center -> distance = -0.5
    pts = torch.tensor([[3.0, 0.0, 0.0]], device=d)
    v, g = composed(pts)
    assert torch.allclose(v.squeeze(), torch.tensor(-0.5, device=d), atol=1e-5)

    # Point at origin: equidistant from both spheres, distance = 3.0 - 0.5 = 2.5
    pts = torch.tensor([[0.0, 0.0, 0.0]], device=d)
    v, g = composed(pts)
    assert torch.allclose(v.squeeze(), torch.tensor(2.5, device=d), atol=1e-5)


def test_composed_set_transforms():
    """Test that set_transforms updates correctly."""
    d = "cuda" if torch.cuda.is_available() else "cpu"
    sdfs = [pv.SphereSDF(1.0), pv.SphereSDF(1.0)]
    composed = pv.ComposedSDF(sdfs, None)
    composed.set_transforms(pk.Translate(0, 0, 0, device=d).stack(pk.Translate(0, 0, 0, device=d)))

    pts = torch.tensor([[0.0, 0.0, 0.0]], device=d)
    v1, _ = composed(pts)
    assert v1.squeeze().item() < 0  # inside sphere at origin

    # Move both spheres far away
    composed.set_transforms(pk.Translate(100, 0, 0, device=d).stack(pk.Translate(-100, 0, 0, device=d)))
    v2, _ = composed(pts)
    assert v2.squeeze().item() > 0  # now outside


# ── Batch dimensions ─────────────────────────────────────────────────────────

def test_composed_flat_query_values():
    """ComposedSDF with flat N x 3 queries should return N values."""
    d = "cuda" if torch.cuda.is_available() else "cpu"
    composed = _make_composed_spheres(2, [-2.0, 2.0], device=d)

    pts = torch.randn(100, 3, device=d)
    v, g = composed(pts)
    # Without tsf_batch, ComposedSDF flattens all dimensions
    assert v.shape == (100,)
    assert g.shape == (100, 3)


def _make_batched_sphere_transforms(S, B, device):
    """Helper: create B configs for S spheres, with SDF i at x = (i+1)*(b+1)."""
    matrices = torch.eye(4, device=device).unsqueeze(0).repeat(B * S, 1, 1)
    for b in range(B):
        matrices[0 * B + b, 0, 3] = -(b + 1.0)
        matrices[1 * B + b, 0, 3] = (b + 1.0)
    return matrices


def test_composed_batch_transforms():
    """Test ComposedSDF with batched transforms (multiple configurations)."""
    d = "cuda" if torch.cuda.is_available() else "cpu"
    sdfs = [pv.SphereSDF(0.5), pv.SphereSDF(0.5)]

    # 3 configurations, 2 SDFs -> 6 transforms total
    # Layout: S interleaved with B, i.e. [sdf0_cfg0, sdf0_cfg1, sdf0_cfg2, sdf1_cfg0, sdf1_cfg1, sdf1_cfg2]
    B = 3
    S = 2
    matrices = _make_batched_sphere_transforms(S, B, d)

    tsf = pk.Transform3d(matrix=matrices, device=d)
    composed = pv.ComposedSDF(sdfs, None)
    composed.set_transforms(tsf, batch_dim=(B,))

    # Query at origin
    pts = torch.tensor([[0.0, 0.0, 0.0]], device=d)
    v, g = composed(pts)

    # For each config, the closest sphere surface distance from origin is (offset - 0.5)
    assert v.shape == (B, 1)
    for b in range(B):
        expected_dist = (b + 1.0) - 0.5
        assert torch.allclose(v[b, 0], torch.tensor(expected_dist, device=d), atol=1e-5)


def test_composed_batch_transforms_no_grad():
    """Batched transforms with compute_grad=False should match with-grad values."""
    d = "cuda" if torch.cuda.is_available() else "cpu"
    sdfs = [pv.SphereSDF(0.5), pv.SphereSDF(0.5)]
    B = 3
    S = 2
    matrices = _make_batched_sphere_transforms(S, B, d)

    tsf = pk.Transform3d(matrix=matrices, device=d)
    composed = pv.ComposedSDF(sdfs, None)
    composed.set_transforms(tsf, batch_dim=(B,))

    pts = torch.randn(50, 3, device=d)
    v_grad, g_grad = composed(pts, compute_grad=True)
    v_no_grad, g_no_grad = composed(pts, compute_grad=False)

    assert v_no_grad.shape == v_grad.shape
    assert torch.allclose(v_no_grad, v_grad, atol=1e-5)
    assert g_no_grad is None


def test_composed_batch_transforms_matches_sequential():
    """Batched-config query should match running each config individually."""
    d = "cuda" if torch.cuda.is_available() else "cpu"
    sdfs_batched = [pv.SphereSDF(0.5), pv.SphereSDF(0.5)]
    B = 4
    S = 2
    matrices = _make_batched_sphere_transforms(S, B, d)

    tsf = pk.Transform3d(matrix=matrices, device=d)
    composed = pv.ComposedSDF(sdfs_batched, None)
    composed.set_transforms(tsf, batch_dim=(B,))

    pts = torch.randn(100, 3, device=d)

    # Batched result
    v_batched, g_batched = composed(pts)
    assert v_batched.shape == (B, 100)

    # Sequential: run each config individually and compare
    for b in range(B):
        sdfs_single = [pv.SphereSDF(0.5), pv.SphereSDF(0.5)]
        single_matrices = torch.stack([matrices[0 * B + b], matrices[1 * B + b]])
        single_tsf = pk.Transform3d(matrix=single_matrices, device=d)
        single_composed = pv.ComposedSDF(sdfs_single, None)
        single_composed.set_transforms(single_tsf)

        v_single, g_single = single_composed(pts)
        assert torch.allclose(v_batched[b], v_single.squeeze(), atol=1e-5), f"Config {b} mismatch"
        assert torch.allclose(g_batched[b], g_single.squeeze(), atol=1e-5), f"Config {b} grad mismatch"


# ── Differentiability ─────────────────────────────────────────────────────────

def test_composed_differentiability():
    """d(sdf_val)/d(pts) should equal sdf_grad through autograd."""
    d = "cuda" if torch.cuda.is_available() else "cpu"
    composed = _make_composed_spheres(3, [-3.0, 0.0, 3.0], device=d)

    pts = torch.randn(200, 3, device=d, requires_grad=True)
    v, g = composed(pts)
    v.sum().backward()
    assert pts.grad is not None
    assert torch.allclose(pts.grad, g, atol=1e-4)


def test_composed_differentiability_batched_transforms():
    """Autograd with batched transforms (as used by RobotSDF)."""
    d = "cuda" if torch.cuda.is_available() else "cpu"
    sdfs = [pv.SphereSDF(0.5), pv.SphereSDF(0.5)]
    B = 2
    S = 2
    matrices = torch.eye(4, device=d).unsqueeze(0).repeat(B * S, 1, 1)
    for b in range(B):
        matrices[0 * B + b, 0, 3] = -(b + 1.0)
        matrices[1 * B + b, 0, 3] = (b + 1.0)

    composed = pv.ComposedSDF(sdfs, None)
    composed.set_transforms(pk.Transform3d(matrix=matrices, device=d), batch_dim=(B,))

    pts = torch.randn(50, 3, device=d, requires_grad=True)
    v, g = composed(pts)
    assert v.shape == (B, 50)
    v.sum().backward()
    assert pts.grad is not None


# ── Many SDFs ─────────────────────────────────────────────────────────────────

def test_composed_many_sdfs():
    """Test with a larger number of SDFs (similar to a robot with many links)."""
    d = "cuda" if torch.cuda.is_available() else "cpu"
    n = 10
    offsets = [float(i) * 3.0 for i in range(n)]
    composed = _make_composed_spheres(n, offsets, device=d)

    pts = torch.randn(500, 3, device=d)
    v, g = composed(pts)
    assert v.shape == (500,)
    assert g.shape == (500, 3)

    # Verify min-selection: manually check a point near the first sphere
    pt = torch.tensor([[0.0, 0.0, 0.0]], device=d)
    v_comp, _ = composed(pt)
    v_direct, _ = pv.SphereSDF(0.5)(pt)  # sphere 0 is at origin
    assert torch.allclose(v_comp.squeeze(), v_direct.squeeze(), atol=1e-5)


# ── Surface bounding box ─────────────────────────────────────────────────────

def test_composed_surface_bounding_box():
    """Bounding box should encompass all SDFs."""
    d = "cuda" if torch.cuda.is_available() else "cpu"
    composed = _make_composed_spheres(2, [-5.0, 5.0], device=d)
    bb = composed.surface_bounding_box()

    # Should span from about -5.5 to 5.5 in x (sphere centers +/- radius)
    assert bb[0, 0] <= -5.0  # x min
    assert bb[0, 1] >= 5.0   # x max
