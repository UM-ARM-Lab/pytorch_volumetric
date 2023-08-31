import copy

import torch

from pytorch_volumetric import voxel
from pytorch_volumetric import sdf
from pytorch_volumetric import model_to_sdf
from matplotlib import pyplot as plt
import matplotlib.colors


def fmt(x):
    s = f"{x:.1f}"
    if s.endswith("0"):
        s = f"{x:.0f}"
    if x == 0:
        return "surface"
    return rf"{s}" if plt.rcParams["text.usetex"] else f"{s}"


def draw_sdf_slice(s: sdf.ObjectFrameSDF, query_range, resolution=0.01, interior_padding=0.2,
                   cmap="Greys_r", device="cpu", plot_grad=False):
    """

    :param s: SDF to query on
    :param query_range: (min, max) for each dimension x,y,z. One dimension must have min=max to be sliced along, with
    the other dimensions shown. Note that this should be given in the SDF's frame.
    :param resolution:
    :param interior_padding:
    :param cmap: matplotlib compatible colormap
    :param device: pytorch device
    :param plot_grad: whether to plot the gradient field
    :return:
    """
    coords, pts = voxel.get_coordinates_and_points_in_grid(resolution, query_range, device=device)
    # add a small amount of noise to avoid querying regular grid
    pts += torch.randn_like(pts) * 1e-6
    dim_labels = ['x', 'y', 'z']
    slice_dim = None
    for i in range(len(dim_labels)):
        if len(coords[i]) == 1:
            slice_dim = i
            break

    # to properly draw a slice, the coords for that dimension must have only 1 element
    if slice_dim is None:
        raise RuntimeError(f"Sliced SDF requires a single query value for the sliced, but all query dimensions > 1")

    shown_dims = [i for i in range(3) if i != slice_dim]

    sdf_val, sdf_grad = s(pts)
    norm = matplotlib.colors.Normalize(vmin=sdf_val.min().cpu() - interior_padding, vmax=sdf_val.max().cpu())

    ax = plt.gca()
    ax.set_xlabel(dim_labels[shown_dims[0]])
    ax.set_ylabel(dim_labels[shown_dims[1]])
    x = coords[shown_dims[0]].cpu()
    z = coords[shown_dims[1]].cpu()
    v = sdf_val.reshape(len(x), len(z)).transpose(0, 1).cpu()
    cset1 = ax.contourf(x, z, v, norm=norm, cmap=cmap)
    cset2 = ax.contour(x, z, v, colors='k', levels=[0], linestyles='dashed')
    if plot_grad:
        sdf_grad_uv = sdf_grad.reshape(len(x), len(z), 3).permute(1, 0, 2).cpu()
        # subsample arrows
        subsample_n = 5
        ax.quiver(x[::subsample_n],
                  z[::subsample_n],
               sdf_grad_uv[::subsample_n, ::subsample_n, shown_dims[0]],
               sdf_grad_uv[::subsample_n, ::subsample_n, shown_dims[1]], color='g')
    ax.clabel(cset2, cset2.levels, inline=True, fontsize=13, fmt=fmt)
    # fig = plt.gcf()
    # fig.canvas.draw()
    plt.draw()
    plt.pause(0.005)
    return sdf_val, sdf_grad, pts, ax, cset1, cset2, v


def get_transformed_meshes(robot_sdf: model_to_sdf.RobotSDF, obj_to_world_tsf=None):
    """Get the meshes of each link of the robot, transformed to the world frame.
    Each link is assumed to be a MeshSDF.
    You can use this like:

    import open3d as o3d
    meshes = get_transformed_meshes(robot_sdf)
    o3d.visualization.draw_geometries(meshes)
    """

    meshes = []
    # link to obj in the form of (object) H (link)
    tsfs = robot_sdf.sdf.obj_frame_to_link_frame.inverse()
    # given a transform in the form of (world) H (object)
    if obj_to_world_tsf is not None:
        # compose the transform to get (world) H (link)
        tsfs = obj_to_world_tsf.compose(tsfs)
    tsfs = tsfs.get_matrix()
    for i in range(len(robot_sdf.sdf_to_link_name)):
        # assuming they are individually MeshSDFs
        mesh = copy.deepcopy(robot_sdf.sdf.sdfs[i].obj_factory._mesh)
        mesh = mesh.transform(tsfs[i].cpu().numpy())
        meshes.append(mesh)
    return meshes
