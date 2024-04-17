import torch


def is_inside(points: torch.tensor, range_per_dim: torch.tensor):
    """Return whether the points are inside the range
    :param points: N x d geometric points; d is the number of dimensions
    :param range_per_dim: d x 2 range of values per dimension, each row is (min, max)
    :return: N boolean tensor
    """
    return torch.all((range_per_dim[:, 0] <= points) & (points <= range_per_dim[:, 1]), dim=-1)
