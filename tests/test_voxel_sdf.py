import matplotlib.pyplot as plt
import torch
import pytorch_volumetric as pv

to_plot = True


def test_voxel_down_sample():
    N = 100

    def f(x, y):
        return torch.sin(x) + 2 * torch.cos(y)

    x = torch.linspace(-2, 2, N)
    y = torch.linspace(-2, 2, N)
    xx, yy = torch.meshgrid(x, y)
    values = f(xx, yy).flatten()
    pts = torch.stack((xx.flatten(), yy.flatten(), values), dim=-1)

    bounds = 4
    prev_resolution = bounds / N
    new_resolution = 0.2
    reduce_factor = prev_resolution / new_resolution
    pts_reduced = pv.voxel_down_sample(pts, new_resolution)

    values_reduced = f(pts_reduced[:, 0], pts_reduced[:, 1])
    assert pts_reduced.shape[0] < pts.shape[0] * reduce_factor
    # expect an error of around the new resolution
    assert torch.allclose(values_reduced, pts_reduced[:, 2], atol=new_resolution * 2)
    # plt.scatter(pts[:, 0], pts[:, 1], c=values)
    # plot 3d
    if to_plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # plot 3d surface
        ax.plot_surface(xx, yy, values.reshape(xx.shape))
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2])
        ax.scatter(pts_reduced[:, 0], pts_reduced[:, 1], pts_reduced[:, 2])
        plt.show()


if __name__ == "__main__":
    test_voxel_down_sample()
