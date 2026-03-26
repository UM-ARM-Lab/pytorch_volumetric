import numpy as np
import torch
import pytorch_volumetric as pv
from pytorch_volumetric.voxel import get_divisible_range_by_resolution, VoxelGrid, ExpandingVoxelGrid, VoxelSet, \
    bounds_contain_another_bounds


def test_get_divisible_range_by_resolution():
    """Ensure range is snapped to be evenly divisible by resolution."""
    resolution = 0.1
    range_per_dim = [(-0.55, 0.57), (0.0, 1.03)]
    result = get_divisible_range_by_resolution(resolution, range_per_dim)
    for low, high in result:
        span = high - low
        # span / resolution should be an integer
        assert abs(round(span / resolution) - span / resolution) < 1e-10


def test_get_coordinates_and_points_in_grid():
    resolution = 0.5
    range_per_dim = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    coords, pts = pv.get_coordinates_and_points_in_grid(resolution, range_per_dim)

    assert len(coords) == 3
    for c in coords:
        assert c[0] >= 0.0
        assert c[-1] <= 1.0 + resolution

    # pts should be cartesian product
    expected_n = 1
    for c in coords:
        expected_n *= len(c)
    assert pts.shape == (expected_n, 3)


def test_get_coordinates_and_points_in_grid_no_points():
    resolution = 0.5
    range_per_dim = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    coords, pts = pv.get_coordinates_and_points_in_grid(resolution, range_per_dim, get_points=False)
    assert pts is None
    assert len(coords) == 3


def test_voxel_grid_set_get():
    """Test basic set/get on VoxelGrid."""
    resolution = 0.1
    range_per_dim = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    vg = VoxelGrid(resolution, range_per_dim)

    # Set some values
    pts = torch.tensor([[0.15, 0.15, 0.15], [0.55, 0.55, 0.55]])
    values = torch.tensor([1.0, 2.0])
    vg[pts] = values

    # Get them back - should be close (quantized to grid)
    retrieved = vg[pts]
    assert torch.allclose(retrieved, values, atol=1e-5)


def test_voxel_grid_get_known_pos_and_values():
    resolution = 0.1
    range_per_dim = [(0.0, 0.5), (0.0, 0.5), (0.0, 0.5)]
    vg = VoxelGrid(resolution, range_per_dim)

    # Initially no known values (all zeros = invalid_val)
    pos, val = vg.get_known_pos_and_values()
    assert pos.shape[0] == 0

    # Set a value
    pts = torch.tensor([[0.15, 0.15, 0.15]])
    vg[pts] = torch.tensor([5.0])
    pos, val = vg.get_known_pos_and_values()
    assert pos.shape[0] >= 1
    assert (val == 5.0).any()


def test_voxel_grid_resize_to_fit():
    resolution = 0.1
    range_per_dim = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    vg = VoxelGrid(resolution, range_per_dim)

    # Set values in a small region (spaced > resolution apart so they land in different cells)
    pts = torch.tensor([[0.15, 0.15, 0.15], [0.55, 0.55, 0.55]])
    values = torch.tensor([1.0, 2.0])
    vg[pts] = values

    original_data_shape = vg._data.shape
    vg.resize_to_fit()
    # After resize, the grid should be smaller
    new_data_shape = vg._data.shape
    for orig, new in zip(original_data_shape, new_data_shape):
        assert new <= orig

    # Values should be preserved (check via get_known_pos_and_values)
    pos, val = vg.get_known_pos_and_values()
    assert pos.shape[0] == 2
    assert torch.sort(val.flatten())[0].allclose(torch.sort(values)[0])


def test_voxel_grid_get_voxel_values_and_center_points():
    resolution = 0.5
    range_per_dim = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    vg = VoxelGrid(resolution, range_per_dim)

    data = vg.get_voxel_values()
    assert data.shape == vg._data.shape
    assert (data == 0).all()

    center_pts = vg.get_voxel_center_points()
    assert center_pts.shape[-1] == 3


def test_expanding_voxel_grid():
    """Test that ExpandingVoxelGrid auto-expands when setting out-of-bounds points."""
    resolution = 0.1
    range_per_dim = [(0.0, 0.5), (0.0, 0.5), (0.0, 0.5)]
    vg = ExpandingVoxelGrid(resolution, range_per_dim)

    original_range = vg.range_per_dim.copy()

    # Set a point inside bounds
    pts_inside = torch.tensor([[0.15, 0.15, 0.15]])
    vg[pts_inside] = torch.tensor([1.0])

    # Set a point outside bounds - should trigger expansion
    pts_outside = torch.tensor([[0.8, 0.8, 0.8]])
    vg[pts_outside] = torch.tensor([2.0])

    # Range should have expanded
    for dim in range(3):
        assert vg.range_per_dim[dim][1] >= 0.8

    # Both values should still be retrievable
    assert vg[pts_inside].item() != 0
    assert vg[pts_outside].item() != 0


def test_voxel_set():
    """Test VoxelSet stores and appends positions/values."""
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    values = torch.tensor([1.0, 2.0])
    vs = VoxelSet(positions, values)

    pos, val = vs.get_known_pos_and_values()
    assert torch.equal(pos, positions)
    assert torch.equal(val, values)

    # Append more
    new_pts = torch.tensor([[2.0, 2.0, 2.0]])
    new_vals = torch.tensor([3.0])
    vs[new_pts] = new_vals

    pos, val = vs.get_known_pos_and_values()
    assert pos.shape[0] == 3
    assert val.shape[0] == 3


def test_voxel_set_getitem_raises():
    """VoxelSet.__getitem__ should raise RuntimeError."""
    vs = VoxelSet(torch.zeros(1, 3), torch.zeros(1))
    try:
        vs[torch.zeros(1, 3)]
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        pass


def test_bounds_contain_another_bounds():
    outer = np.array([[0.0, 10.0], [0.0, 10.0]])
    inner = np.array([[1.0, 9.0], [2.0, 8.0]])
    assert bounds_contain_another_bounds(outer, inner)

    # Not contained
    inner_bad = np.array([[-1.0, 9.0], [2.0, 8.0]])
    assert not bounds_contain_another_bounds(outer, inner_bad)
