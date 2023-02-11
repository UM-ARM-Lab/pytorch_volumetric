import abc
import copy
import math

import numpy as np
import torch
from multidim_indexing import torch_view


def get_divisible_range_by_resolution(resolution, range_per_dim):
    # ensure value range divides resolution evenly
    temp_range = []
    for low, high in range_per_dim:
        span = high - low
        span = round(span / resolution)
        temp_range.append((low, low + span * resolution))
    return temp_range


def get_coordinates_and_points_in_grid(resolution, range_per_dim, dtype=torch.float, device='cpu', get_points=True):
    # create points along the value ranges
    coords = [torch.arange(low, high + resolution, resolution, dtype=dtype, device=device) for low, high in
              range_per_dim]
    pts = torch.cartesian_prod(*coords) if get_points else None
    return coords, pts


class Voxels(abc.ABC):
    @abc.abstractmethod
    def get_known_pos_and_values(self):
        """Return the position (N x 3) and values (N) of known voxels"""

    @abc.abstractmethod
    def __getitem__(self, pts):
        """Return the values (N) at the positions (N x 3)"""

    @abc.abstractmethod
    def __setitem__(self, pts, value):
        """Set the values (N) at the positions (N x 3)"""


class VoxelGrid(Voxels):
    def __init__(self, resolution, range_per_dim, dtype=torch.float, device='cpu'):
        self.resolution = resolution
        self.invalid_val = 0
        self.dtype = dtype
        self.device = device
        self._create_voxels(resolution, range_per_dim)

    def _create_voxels(self, resolution, range_per_dim):
        self.range_per_dim = get_divisible_range_by_resolution(resolution, range_per_dim)
        self.coords, self.pts = get_coordinates_and_points_in_grid(resolution, self.range_per_dim, dtype=self.dtype,
                                                                   device=self.device)
        # underlying data
        self._data = torch.zeros([len(coord) for coord in self.coords], dtype=self.dtype, device=self.device)
        self.voxels = torch_view.TorchMultidimView(self._data, self.range_per_dim, invalid_value=self.invalid_val)
        self.range_per_dim = np.array(self.range_per_dim)

    def get_known_pos_and_values(self):
        known = self.voxels.raw_data != self.invalid_val
        indices = known.nonzero()
        # these points are in object frame
        pos = self.voxels.ensure_value_key(indices)
        val = self.voxels.raw_data[indices]
        return pos, val

    def get_voxel_values(self):
        """Get the raw value of the voxels without any coordinate information"""
        return self._data

    def get_voxel_center_points(self):
        return self.pts

    def __getitem__(self, pts):
        return self.voxels[pts]

    def __setitem__(self, pts, value):
        self.voxels[pts] = value


class ExpandingVoxelGrid(VoxelGrid):
    def __setitem__(self, pts, value):
        # if this query goes outside the range, expand the range in increments of the resolution
        min = pts.min(dim=0).values
        max = pts.max(dim=0).values
        range_per_dim = copy.deepcopy(self.range_per_dim)
        for dim in range(len(min)):
            over = (max[dim] - self.range_per_dim[dim][1]).item()
            under = (self.range_per_dim[dim][0] - min[dim]).item()
            # adjust in increments of resolution
            if over > 0:
                range_per_dim[dim][1] += math.ceil(over / self.resolution) * self.resolution
            if under > 0:
                range_per_dim[dim][0] -= math.ceil(under / self.resolution) * self.resolution
        if not np.allclose(range_per_dim, self.range_per_dim):
            # transfer over values
            known_pos, known_values = self.get_known_pos_and_values()
            self._create_voxels(self.resolution, range_per_dim)
            super().__setitem__(known_pos, known_values)

        return super().__setitem__(pts, value)


class VoxelSet(Voxels):
    def __init__(self, positions, values):
        self.positions = positions
        self.values = values

    def __getitem__(self, pts):
        raise RuntimeError("Cannot get arbitrary points on a voxel set")

    def __setitem__(self, pts, value):
        self.positions = torch.cat((self.positions, pts.view(-1, self.positions.shape[-1])), dim=0)
        self.values = torch.cat((self.values, value))

    def get_known_pos_and_values(self):
        return self.positions, self.values
