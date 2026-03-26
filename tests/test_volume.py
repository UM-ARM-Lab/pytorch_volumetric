import torch
from pytorch_volumetric import is_inside


def test_is_inside_basic():
    range_per_dim = torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])

    # Points inside
    pts_in = torch.tensor([[0.5, 0.5, 0.5], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    result = is_inside(pts_in, range_per_dim)
    assert result.all()

    # Points outside
    pts_out = torch.tensor([[-0.1, 0.5, 0.5], [0.5, 1.1, 0.5], [0.5, 0.5, -0.1]])
    result = is_inside(pts_out, range_per_dim)
    assert not result.any()


def test_is_inside_mixed():
    range_per_dim = torch.tensor([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]])
    pts = torch.tensor([
        [0.0, 0.0, 0.0],   # inside
        [2.0, 0.0, 0.0],   # outside
        [-1.0, -1.0, -1.0], # on boundary (inside)
    ])
    result = is_inside(pts, range_per_dim)
    assert result[0].item() is True
    assert result[1].item() is False
    assert result[2].item() is True


def test_is_inside_2d():
    """Test with 2D points."""
    range_per_dim = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    pts = torch.tensor([[0.5, 0.5], [1.5, 0.5]])
    result = is_inside(pts, range_per_dim)
    assert result[0].item() is True
    assert result[1].item() is False
