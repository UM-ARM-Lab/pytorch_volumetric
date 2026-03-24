"""
Tests for ComposedSDF and CachedSDF correctness.
Uses a mix of:
  - SphereSDF primitives (fast, analytic)
  - CachedSDF wrapping SphereSDF (exercises voxel lookup paths)
  - CachedSDF from real meshes (L-shape, capsule — asymmetric, non-uniform gradients)
"""
import os
import torch
import pytorch_kinematics as pk
import pytorch_volumetric as pv
from pytorch_volumetric.sdf import OutOfBoundsStrategy

TEST_DIR = os.path.dirname(__file__)
MESH_DIR = os.path.join(TEST_DIR, "test_meshes")


def _make_composed_spheres(n_spheres, offsets, device="cpu"):
    """Create a ComposedSDF of spheres at given offsets along the x-axis."""
    sdfs = [pv.SphereSDF(0.5) for _ in range(n_spheres)]
    tsfs = []
    for offset in offsets:
        tsfs.append(pk.Translate(offset, 0, 0, device=device))
    tsf = tsfs[0].stack(*tsfs[1:]) if len(tsfs) > 1 else tsfs[0]
    return pv.ComposedSDF(sdfs, tsf)


def _make_cached_sphere(radius=0.5, resolution=0.02, padding=0.3,
                         out_of_bounds_strategy=OutOfBoundsStrategy.BOUNDING_BOX,
                         method='nearest', device="cpu"):
    """Create a CachedSDF wrapping a SphereSDF (no mesh files needed)."""
    gt = pv.SphereSDF(radius)
    bb = gt.surface_bounding_box(padding=padding).numpy().tolist()
    cache_path = os.path.join(TEST_DIR, f"test_sphere_cache_{radius}_{resolution}_{padding}.pkl")
    cached = pv.CachedSDF(f"test_sphere_{radius}", resolution, bb, gt,
                           out_of_bounds_strategy=out_of_bounds_strategy,
                           device=device, clean_cache=True, cache_path=cache_path,
                           method=method)
    if os.path.exists(cache_path):
        os.remove(cache_path)
    return cached


def _make_cached_mesh(mesh_name, resolution=0.005, padding=0.05,
                       out_of_bounds_strategy=OutOfBoundsStrategy.BOUNDING_BOX,
                       method='nearest', device="cpu"):
    """Create a CachedSDF from a test mesh file (L_shape.obj or capsule.obj)."""
    mesh_path = os.path.join(MESH_DIR, mesh_name)
    factory = pv.MeshObjectFactory(mesh_path)
    gt = pv.MeshSDF(factory)
    bb = factory.bounding_box(padding=padding)
    cache_path = os.path.join(TEST_DIR, f"test_mesh_cache_{mesh_name}_{resolution}.pkl")
    cached = pv.CachedSDF(mesh_name, resolution, bb, gt,
                           out_of_bounds_strategy=out_of_bounds_strategy,
                           device=device, clean_cache=True, cache_path=cache_path,
                           method=method)
    if os.path.exists(cache_path):
        os.remove(cache_path)
    return cached


def _make_composed_cached_spheres(n_spheres, offsets, device="cpu", method='nearest'):
    """Create a ComposedSDF of CachedSDFs wrapping SphereSDF primitives."""
    sdfs = [_make_cached_sphere(method=method, device=device) for _ in range(n_spheres)]
    tsfs = []
    for offset in offsets:
        tsfs.append(pk.Translate(offset, 0, 0, device=device))
    tsf = tsfs[0].stack(*tsfs[1:]) if len(tsfs) > 1 else tsfs[0]
    return pv.ComposedSDF(sdfs, tsf)


def _make_composed_mixed_meshes(device="cpu"):
    """ComposedSDF with heterogeneous CachedSDFs: L-shape + capsule at different positions.
    Exercises non-uniform gradients, asymmetric geometry, and real mesh lookups."""
    sdf_l = _make_cached_mesh("L_shape.obj", device=device)
    sdf_c = _make_cached_mesh("capsule.obj", device=device)
    # Place L-shape at x=-0.3, capsule at x=+0.3
    tsf = pk.Translate(-0.3, 0, 0, device=device).stack(pk.Translate(0.3, 0, 0, device=device))
    return pv.ComposedSDF([sdf_l, sdf_c], tsf)


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


# ── CachedSDF correctness (exercises optimized fused lookup) ────────────────

def test_cached_sdf_no_grad_matches_gt():
    """CachedSDF no-grad fused lookup should approximate ground truth SphereSDF."""
    d = "cuda" if torch.cuda.is_available() else "cpu"
    gt = pv.SphereSDF(0.5)
    cached = _make_cached_sphere(device=d)

    # Points inside the voxel grid (keep within grid to avoid OOB)
    pts = torch.randn(500, 3, device=d) * 0.2
    v_gt, _ = gt(pts)
    v_cached, _ = cached(pts, compute_grad=False)
    # Nearest-neighbor discretization error: mean should be small, max bounded by resolution
    diff = (v_gt - v_cached).abs()
    assert diff.mean() < 0.02, f"Mean error {diff.mean():.4f} too large"
    assert diff.max() < 0.05, f"Max error {diff.max():.4f} too large"


def test_cached_sdf_no_grad_matches_with_grad():
    """CachedSDF no-grad path should produce same values as with-grad path."""
    d = "cuda" if torch.cuda.is_available() else "cpu"
    cached = _make_cached_sphere(device=d)

    pts = torch.randn(500, 3, device=d) * 0.3
    v_grad, g_grad = cached(pts, compute_grad=True)
    v_no_grad, g_no_grad = cached(pts, compute_grad=False)

    assert torch.allclose(v_grad, v_no_grad, atol=1e-6)
    assert g_no_grad is None


def test_cached_sdf_oob_bounding_box():
    """CachedSDF BOUNDING_BOX strategy: OOB points get distance to bounding box."""
    d = "cuda" if torch.cuda.is_available() else "cpu"
    cached = _make_cached_sphere(padding=0.1, out_of_bounds_strategy=OutOfBoundsStrategy.BOUNDING_BOX, device=d)

    # Points far outside the grid — distance should be positive and large
    pts_far = torch.tensor([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]], device=d)
    v, _ = cached(pts_far, compute_grad=False)
    assert (v > 0).all()
    assert (v > 3.0).all()  # well outside the grid


def test_cached_sdf_oob_lookup_gt():
    """CachedSDF LOOKUP_GT_SDF strategy: OOB points fall back to ground truth."""
    d = "cuda" if torch.cuda.is_available() else "cpu"
    gt = pv.SphereSDF(0.5)
    cached = _make_cached_sphere(padding=0.1, out_of_bounds_strategy=OutOfBoundsStrategy.LOOKUP_GT_SDF, device=d)

    # Points outside the grid but where GT is well-defined
    pts_far = torch.tensor([[3.0, 0.0, 0.0], [-3.0, 0.0, 0.0]], device=d)
    v_cached, _ = cached(pts_far, compute_grad=False)
    v_gt, _ = gt(pts_far, compute_grad=False)
    assert torch.allclose(v_cached, v_gt, atol=1e-5)


def test_cached_sdf_mixed_inbound_oob():
    """CachedSDF with a mix of in-bound and OOB points."""
    d = "cuda" if torch.cuda.is_available() else "cpu"
    cached = _make_cached_sphere(padding=0.1, device=d)

    pts = torch.tensor([
        [0.0, 0.0, 0.0],   # in-bound (inside sphere)
        [0.4, 0.0, 0.0],   # in-bound (near surface)
        [5.0, 0.0, 0.0],   # OOB
        [-5.0, 0.0, 0.0],  # OOB
    ], device=d)
    v, _ = cached(pts, compute_grad=False)

    assert v[0] < 0  # inside sphere
    assert v[1] < 0.15  # near surface
    assert v[2] > 0  # OOB, positive
    assert v[3] > 0  # OOB, positive


# ── ComposedSDF with CachedSDF (exercises BatchedViewLookup path) ───────────

def test_composed_cached_single_config_no_grad():
    """ComposedSDF with CachedSDFs: single-config no-grad uses BatchedViewLookup."""
    d = "cuda" if torch.cuda.is_available() else "cpu"
    composed = _make_composed_cached_spheres(3, [-2.0, 0.0, 2.0], device=d)

    pts = torch.randn(200, 3, device=d) * 0.5
    v_grad, g_grad = composed(pts, compute_grad=True)
    v_no_grad, _ = composed(pts, compute_grad=False)

    assert torch.allclose(v_grad, v_no_grad, atol=1e-6)


def test_composed_cached_single_config_values():
    """ComposedSDF with CachedSDFs: min selection matches individual lookups."""
    d = "cuda" if torch.cuda.is_available() else "cpu"
    offsets = [-3.0, 3.0]
    composed = _make_composed_cached_spheres(2, offsets, device=d)

    # Point near left sphere
    pt = torch.tensor([[-2.6, 0.0, 0.0]], device=d)
    v, _ = composed(pt, compute_grad=False)
    v_direct, _ = _make_cached_sphere(device=d)(pt - torch.tensor([[-3.0, 0, 0]], device=d))
    assert torch.allclose(v.squeeze(), v_direct.squeeze(), atol=0.02)


def test_composed_cached_oob_handling():
    """ComposedSDF with CachedSDFs: OOB points handled correctly in batched lookup."""
    d = "cuda" if torch.cuda.is_available() else "cpu"
    composed = _make_composed_cached_spheres(2, [-2.0, 2.0], device=d)

    # Mix of in-bound and OOB points
    pts = torch.tensor([
        [0.0, 0.0, 0.0],   # between both spheres, in-bound for both grids
        [10.0, 0.0, 0.0],  # far OOB
    ], device=d)
    v, _ = composed(pts, compute_grad=False)

    assert v[0] > 0  # between spheres, outside both
    assert v[1] > 0  # far away
    assert v[1] > v[0]  # farther point has larger distance


# ── ComposedSDF with CachedSDF: batched configs (per-link fused path) ──────

def test_composed_cached_batched_config_no_grad():
    """Batched configs with CachedSDF: no-grad matches with-grad values."""
    d = "cuda" if torch.cuda.is_available() else "cpu"
    sdfs = [_make_cached_sphere(device=d) for _ in range(2)]
    B, S = 4, 2

    matrices = torch.eye(4, device=d).unsqueeze(0).repeat(B * S, 1, 1)
    for b in range(B):
        matrices[0 * B + b, 0, 3] = -(b + 1.0)
        matrices[1 * B + b, 0, 3] = (b + 1.0)

    composed = pv.ComposedSDF(sdfs, None)
    composed.set_transforms(pk.Transform3d(matrix=matrices, device=d), batch_dim=(B,))

    pts = torch.randn(50, 3, device=d) * 0.3
    v_grad, _ = composed(pts, compute_grad=True)
    v_no_grad, _ = composed(pts, compute_grad=False)

    assert v_no_grad.shape == v_grad.shape == (B, 50)
    assert torch.allclose(v_grad, v_no_grad, atol=0.02)


def test_composed_cached_batched_config_matches_sequential():
    """Batched CachedSDF configs match running each config individually."""
    d = "cuda" if torch.cuda.is_available() else "cpu"
    B, S = 4, 2

    matrices = torch.eye(4, device=d).unsqueeze(0).repeat(B * S, 1, 1)
    for b in range(B):
        matrices[0 * B + b, 0, 3] = -(b + 1.0)
        matrices[1 * B + b, 0, 3] = (b + 1.0)

    # Batched query
    sdfs_batched = [_make_cached_sphere(device=d) for _ in range(S)]
    composed = pv.ComposedSDF(sdfs_batched, None)
    composed.set_transforms(pk.Transform3d(matrix=matrices, device=d), batch_dim=(B,))

    pts = torch.randn(100, 3, device=d) * 0.3
    v_batched, _ = composed(pts, compute_grad=False)

    # Sequential: run each config individually
    for b in range(B):
        sdfs_single = [_make_cached_sphere(device=d) for _ in range(S)]
        single_matrices = torch.stack([matrices[0 * B + b], matrices[1 * B + b]])
        single_composed = pv.ComposedSDF(sdfs_single, None)
        single_composed.set_transforms(pk.Transform3d(matrix=single_matrices, device=d))

        v_single, _ = single_composed(pts, compute_grad=False)
        assert torch.allclose(v_batched[b], v_single.squeeze(), atol=1e-5), f"Config {b} mismatch"


# ── Rotation tests ──────────────────────────────────────────────────────────

def _rotation_matrix_z(angle, device="cpu"):
    """Create a 4x4 rotation matrix around the z-axis."""
    c, s = torch.cos(angle), torch.sin(angle)
    m = torch.eye(4, device=device)
    m[0, 0] = c
    m[0, 1] = -s
    m[1, 0] = s
    m[1, 1] = c
    return m


def test_composed_with_rotation():
    """ComposedSDF with rotated transforms: values and gradients correct."""
    d = "cuda" if torch.cuda.is_available() else "cpu"
    sdfs = [pv.SphereSDF(0.5), pv.SphereSDF(0.5)]

    # obj_frame_to_link_frame: transforms points FROM object frame TO link frame.
    # To place sphere 0 at (2,0,0) in object frame, the transform must map (2,0,0) → (0,0,0).
    # Sphere 0: translate by (-2, 0, 0)
    m0 = torch.eye(4, device=d)
    m0[0, 3] = -2.0
    # Sphere 1: rotate 90° around z, then translate so (0,2,0) maps to origin.
    # Rotation maps (0,2,0) → (-2,0,0), so add translation (2,0,0) to get to origin.
    m1 = _rotation_matrix_z(torch.tensor(torch.pi / 2, device=d), device=d)
    m1[0, 3] = 2.0
    matrices = torch.stack([m0, m1])

    composed = pv.ComposedSDF(sdfs, pk.Transform3d(matrix=matrices, device=d))

    # Point at (2, 0, 0) — should be at sphere 0's center in link frame
    pt = torch.tensor([[2.0, 0.0, 0.0]], device=d)
    v, g = composed(pt)
    assert torch.allclose(v.squeeze(), torch.tensor(-0.5, device=d), atol=1e-5)

    # Point at (0, 2, 0) — should be at sphere 1's center in link frame
    pt = torch.tensor([[0.0, 2.0, 0.0]], device=d)
    v, g = composed(pt)
    assert torch.allclose(v.squeeze(), torch.tensor(-0.5, device=d), atol=1e-5)


def test_composed_cached_with_rotation():
    """ComposedSDF with CachedSDF + rotation: no-grad matches with-grad."""
    d = "cuda" if torch.cuda.is_available() else "cpu"
    sdfs = [_make_cached_sphere(device=d), _make_cached_sphere(device=d)]

    # obj_frame_to_link_frame: places spheres at x=±2 in object frame
    m0 = torch.eye(4, device=d)
    m0[0, 3] = -2.0
    m1 = _rotation_matrix_z(torch.tensor(torch.pi / 4, device=d), device=d)
    m1[0, 3] = 2.0
    matrices = torch.stack([m0, m1])

    composed = pv.ComposedSDF(sdfs, pk.Transform3d(matrix=matrices, device=d))

    pts = torch.randn(200, 3, device=d) * 0.5
    v_grad, g_grad = composed(pts, compute_grad=True)
    v_no_grad, _ = composed(pts, compute_grad=False)
    assert torch.allclose(v_grad, v_no_grad, atol=1e-6)

    # Gradient should be unit-length (SDF gradient property)
    g_norm = g_grad.norm(dim=-1)
    # Inside/outside the sphere the gradient should be ~1 (for well-behaved SDF)
    assert (g_norm > 0.5).all()  # not zero
    assert (g_norm < 1.5).all()  # not exploding


# ── Lazy transform tests ────────────────────────────────────────────────────

def test_lazy_link_frame_to_obj_frame():
    """link_frame_to_obj_frame should be None after set_transforms, populated after with-grad query."""
    d = "cuda" if torch.cuda.is_available() else "cpu"
    composed = _make_composed_spheres(2, [-2.0, 2.0], device=d)

    # After construction, should be None (lazy)
    assert composed.link_frame_to_obj_frame is None

    # No-grad query should NOT populate it
    pts = torch.randn(10, 3, device=d)
    composed(pts, compute_grad=False)
    assert composed.link_frame_to_obj_frame is None

    # With-grad query should populate it
    composed(pts, compute_grad=True)
    assert composed.link_frame_to_obj_frame is not None
    assert len(composed.link_frame_to_obj_frame) == 2

    # Verify it's actually the inverse: compose forward and inverse should give identity
    for i, inv_tsf in enumerate(composed.link_frame_to_obj_frame):
        sl = composed.ith_transform_slice(i)
        fwd_mat = composed.obj_frame_to_link_frame.get_matrix()[sl]
        inv_mat = inv_tsf.get_matrix()
        product = fwd_mat @ inv_mat
        identity = torch.eye(4, device=d).unsqueeze(0).expand_as(product)
        assert torch.allclose(product, identity, atol=1e-5)


def test_lazy_transforms_reset_on_set_transforms():
    """set_transforms should reset lazy transforms so they're recomputed for new config."""
    d = "cuda" if torch.cuda.is_available() else "cpu"
    composed = _make_composed_spheres(2, [-2.0, 2.0], device=d)

    pts = torch.randn(10, 3, device=d)
    composed(pts, compute_grad=True)
    assert composed.link_frame_to_obj_frame is not None

    # set_transforms should reset
    composed.set_transforms(pk.Translate(0, 0, 0, device=d).stack(pk.Translate(1, 0, 0, device=d)))
    assert composed.link_frame_to_obj_frame is None
    assert composed._grad_rotation_mats is None


def test_batched_view_persists_across_set_transforms():
    """BatchedViewLookup should be created once in __init__ and survive set_transforms."""
    d = "cuda" if torch.cuda.is_available() else "cpu"
    sdfs = [_make_cached_sphere(device=d), _make_cached_sphere(device=d)]
    composed = pv.ComposedSDF(sdfs, None)

    bv = composed._batched_view
    assert bv is not None

    # set_transforms should NOT recreate it
    composed.set_transforms(pk.Translate(0, 0, 0, device=d).stack(pk.Translate(1, 0, 0, device=d)))
    assert composed._batched_view is bv  # same object


# ── Mesh-based tests (asymmetric geometry, non-uniform gradients) ───────────

def test_mesh_cached_sdf_no_grad_matches_with_grad():
    """CachedSDF from L-shape mesh: no-grad fused lookup matches with-grad path."""
    d = "cuda" if torch.cuda.is_available() else "cpu"
    cached = _make_cached_mesh("L_shape.obj", device=d)

    pts = torch.randn(300, 3, device=d) * 0.05
    v_grad, g_grad = cached(pts, compute_grad=True)
    v_no_grad, _ = cached(pts, compute_grad=False)
    assert torch.allclose(v_grad, v_no_grad, atol=1e-6)


def test_mesh_cached_sdf_oob():
    """CachedSDF from capsule mesh: OOB points handled correctly."""
    d = "cuda" if torch.cuda.is_available() else "cpu"
    cached = _make_cached_mesh("capsule.obj", padding=0.02, device=d)

    # Far OOB points
    pts = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=d)
    v, _ = cached(pts, compute_grad=False)
    assert (v > 0).all()


def test_composed_mixed_meshes_no_grad_matches_with_grad():
    """ComposedSDF with L-shape + capsule: no-grad matches with-grad."""
    d = "cuda" if torch.cuda.is_available() else "cpu"
    composed = _make_composed_mixed_meshes(device=d)

    pts = torch.randn(200, 3, device=d) * 0.1
    v_grad, g_grad = composed(pts, compute_grad=True)
    v_no_grad, _ = composed(pts, compute_grad=False)
    assert torch.allclose(v_grad, v_no_grad, atol=1e-6)


def test_composed_mixed_meshes_min_selection():
    """ComposedSDF with L-shape + capsule: closer object determines SDF value."""
    d = "cuda" if torch.cuda.is_available() else "cpu"
    composed = _make_composed_mixed_meshes(device=d)

    # Point near L-shape (at x=-0.3): should have small distance
    pt_near_l = torch.tensor([[-0.3, 0.0, 0.0]], device=d)
    v_near, _ = composed(pt_near_l, compute_grad=False)

    # Point far from both objects
    pt_far = torch.tensor([[2.0, 0.0, 0.0]], device=d)
    v_far, _ = composed(pt_far, compute_grad=False)

    assert v_near < v_far


def test_composed_mixed_meshes_batched_config():
    """ComposedSDF with mesh CachedSDFs: batched configs match sequential."""
    d = "cuda" if torch.cuda.is_available() else "cpu"
    B, S = 3, 2

    matrices = torch.eye(4, device=d).unsqueeze(0).repeat(B * S, 1, 1)
    for b in range(B):
        matrices[0 * B + b, 0, 3] = -(b + 1) * 0.1  # L-shape shifts left
        matrices[1 * B + b, 0, 3] = (b + 1) * 0.1   # capsule shifts right

    # Batched
    sdfs_batched = [_make_cached_mesh("L_shape.obj", device=d),
                    _make_cached_mesh("capsule.obj", device=d)]
    composed = pv.ComposedSDF(sdfs_batched, None)
    composed.set_transforms(pk.Transform3d(matrix=matrices, device=d), batch_dim=(B,))

    pts = torch.randn(50, 3, device=d) * 0.05
    v_batched, _ = composed(pts, compute_grad=False)
    assert v_batched.shape == (B, 50)

    # Sequential
    for b in range(B):
        sdfs_single = [_make_cached_mesh("L_shape.obj", device=d),
                       _make_cached_mesh("capsule.obj", device=d)]
        single_matrices = torch.stack([matrices[0 * B + b], matrices[1 * B + b]])
        single_composed = pv.ComposedSDF(sdfs_single, None)
        single_composed.set_transforms(pk.Transform3d(matrix=single_matrices, device=d))

        v_single, _ = single_composed(pts, compute_grad=False)
        # Tolerance accounts for nearest-neighbor boundary differences between CachedSDF instances
        assert torch.allclose(v_batched[b], v_single.squeeze(), atol=0.01), \
            f"Config {b}: max diff {(v_batched[b] - v_single.squeeze()).abs().max():.4f}"


def test_composed_mixed_meshes_differentiability():
    """ComposedSDF with mesh CachedSDFs: autograd backward works."""
    d = "cuda" if torch.cuda.is_available() else "cpu"
    composed = _make_composed_mixed_meshes(device=d)

    pts = torch.randn(100, 3, device=d) * 0.1
    pts.requires_grad_(True)
    pts.retain_grad()
    v, g = composed(pts)
    v.sum().backward()
    assert pts.grad is not None
    assert torch.allclose(pts.grad, g, atol=1e-4)
