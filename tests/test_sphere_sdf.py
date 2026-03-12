import torch
import pytorch_volumetric as pv


def test_sphere_sdf_values():
    """Test that SphereSDF returns correct signed distances."""
    radius = 0.5
    sdf = pv.SphereSDF(radius)

    # Points at known distances from sphere surface
    # Point at origin: distance should be -radius (inside)
    pts = torch.tensor([[0.0, 0.0, 0.0]])
    dist, grad = sdf(pts)
    assert torch.allclose(dist, torch.tensor([-radius]), atol=1e-6)

    # Point on surface: distance should be 0
    pts = torch.tensor([[radius, 0.0, 0.0]])
    dist, grad = sdf(pts)
    assert torch.allclose(dist, torch.tensor([0.0]), atol=1e-6)

    # Point outside: distance should be positive
    pts = torch.tensor([[1.0, 0.0, 0.0]])
    dist, grad = sdf(pts)
    assert torch.allclose(dist, torch.tensor([1.0 - radius]), atol=1e-6)

    # Point inside: distance should be negative
    pts = torch.tensor([[0.25, 0.0, 0.0]])
    dist, grad = sdf(pts)
    assert dist.item() < 0


def test_sphere_sdf_gradient():
    """Test that SphereSDF gradient points radially outward."""
    sdf = pv.SphereSDF(1.0)

    # Gradient at point on x-axis should be [1, 0, 0]
    pts = torch.tensor([[2.0, 0.0, 0.0]])
    dist, grad = sdf(pts)
    assert torch.allclose(grad, torch.tensor([[1.0, 0.0, 0.0]]), atol=1e-6)

    # Gradient at point on y-axis should be [0, 1, 0]
    pts = torch.tensor([[0.0, 3.0, 0.0]])
    dist, grad = sdf(pts)
    assert torch.allclose(grad, torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-6)

    # Gradient should be unit length everywhere (except origin)
    pts = torch.randn(100, 3) + 0.1  # offset to avoid origin
    dist, grad = sdf(pts)
    grad_norms = torch.linalg.norm(grad, dim=-1)
    assert torch.allclose(grad_norms, torch.ones_like(grad_norms), atol=1e-4)


def test_sphere_sdf_batch():
    """Test SphereSDF with batch dimensions."""
    sdf = pv.SphereSDF(1.0)

    # 2D batch: (B1, B2, 3)
    pts = torch.randn(5, 10, 3)
    dist, grad = sdf(pts)
    assert dist.shape == (5, 10)
    assert grad.shape == (5, 10, 3)


def test_sphere_sdf_differentiability():
    """Test that SphereSDF supports autograd."""
    sdf = pv.SphereSDF(1.0)

    pts = torch.randn(50, 3, requires_grad=True)
    dist, grad = sdf(pts)
    dist.sum().backward()
    assert pts.grad is not None
    assert torch.allclose(pts.grad, grad, atol=1e-5)


def test_sphere_sdf_bounding_box():
    """Test surface_bounding_box returns correct bounds."""
    radius = 2.0
    sdf = pv.SphereSDF(radius)

    bb = sdf.surface_bounding_box()
    assert bb.shape == (3, 2)
    assert torch.allclose(bb[:, 0], torch.tensor([-radius]))
    assert torch.allclose(bb[:, 1], torch.tensor([radius]))

    # With padding
    padding = 0.5
    bb = sdf.surface_bounding_box(padding=padding)
    expected = radius + padding
    assert torch.allclose(bb[:, 0], torch.tensor([-expected]))
    assert torch.allclose(bb[:, 1], torch.tensor([expected]))
