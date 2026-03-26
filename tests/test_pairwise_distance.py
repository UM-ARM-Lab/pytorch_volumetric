import os
import torch
import pytorch_kinematics as pk
import pytorch_volumetric as pv

TEST_DIR = os.path.dirname(__file__)


def test_pairwise_distance_identity():
    """Pairwise distance of identical transforms should be zero on diagonal."""
    d = "cuda" if torch.cuda.is_available() else "cpu"
    tf = pk.Transform3d(pos=torch.randn(5, 3, device=d), device=d)
    dist = pv.pairwise_distance(tf)
    assert dist.shape == (5, 5)
    # Diagonal should be zero
    assert torch.allclose(dist.diag(), torch.zeros(5, device=d), atol=1e-5)
    # Should be symmetric
    assert torch.allclose(dist, dist.T, atol=1e-5)
    # Off-diagonal should be non-negative
    assert (dist >= -1e-6).all()


def test_pairwise_distance_distinct():
    """Distinct transforms should have positive pairwise distance."""
    d = "cuda" if torch.cuda.is_available() else "cpu"
    tf1 = pk.Transform3d(pos=torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], device=d), device=d)
    dist = pv.pairwise_distance(tf1)
    assert dist[0, 1] > 0
    assert dist[1, 0] > 0


def test_pairwise_distance_chamfer():
    """Test pairwise_distance_chamfer returns correct shape and zero on diagonal."""
    d = "cuda" if torch.cuda.is_available() else "cpu"
    obj = pv.MeshObjectFactory(os.path.join(TEST_DIR, "probe.obj"))

    B = 3
    gt_tf = pk.Transform3d(pos=torch.randn(B, 3, device=d), device=d)

    dist = pv.pairwise_distance_chamfer(gt_tf, obj_factory=obj, scale=1)
    assert dist.shape == (B, B)
    # Diagonal should be near zero (same transform applied and inverted)
    assert torch.allclose(dist.diag(), torch.zeros(B, device=d), atol=1e-3)


def test_pairwise_distance_chamfer_with_sdf():
    """Test pairwise_distance_chamfer using obj_sdf for acceleration."""
    d = "cuda" if torch.cuda.is_available() else "cpu"
    obj = pv.MeshObjectFactory(os.path.join(TEST_DIR, "probe.obj"))
    mesh_sdf = pv.MeshSDF(obj)

    B = 3
    gt_tf = pk.Transform3d(pos=torch.randn(B, 3, device=d) * 0.01, device=d)

    dist = pv.pairwise_distance_chamfer(gt_tf, obj_factory=obj, obj_sdf=mesh_sdf, scale=1)
    assert dist.shape == (B, B)
    assert torch.allclose(dist.diag(), torch.zeros(B, device=d), atol=1e-3)


def test_pairwise_distance_chamfer_asymmetric():
    """Test with different A and B transform sets."""
    d = "cuda" if torch.cuda.is_available() else "cpu"
    obj = pv.MeshObjectFactory(os.path.join(TEST_DIR, "probe.obj"))

    A = pk.Transform3d(pos=torch.randn(2, 3, device=d) * 0.01, device=d)
    B_inv = pk.Transform3d(pos=torch.randn(4, 3, device=d) * 0.01, device=d)

    dist = pv.pairwise_distance_chamfer(A, B_world_to_link_tfs=B_inv, obj_factory=obj, scale=1)
    assert dist.shape == (2, 4)
