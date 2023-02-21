import os
import math
import torch
import matplotlib.colors
from matplotlib import pyplot as plt
import numpy as np
from timeit import default_timer as timer

import pytorch_kinematics as pk
from pytorch_volumetric.model_to_sdf import RobotSDF
from pytorch_volumetric import voxel

import pybullet as p
import pybullet_data

import logging

plt.switch_backend('Qt5Agg')

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO, force=True,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')


def test_urdf_to_sdf():
    urdf = "kuka_iiwa/model.urdf"
    search_path = pybullet_data.getDataPath()
    full_urdf = os.path.join(search_path, urdf)
    chain = pk.build_serial_chain_from_urdf(open(full_urdf).read(), "lbr_iiwa_link_7")
    s = RobotSDF(chain, path_prefix=os.path.join(search_path, "kuka_iiwa"))
    th = torch.tensor([0.0, -math.pi / 4.0, 0.0, math.pi / 2.0, 0.0, math.pi / 4.0, 0.0])

    s.set_joint_configuration(th)

    y = 0.02
    query_range = np.array([
        [-1, 0.5],
        [y, y],
        [-0.2, 0.8],
    ])

    # toggles - g:GUI w:wireframe j:joint axis a:AABB i:interrupt
    p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.setAdditionalSearchPath(search_path)
    armId = p.loadURDF(urdf, [0, 0, 0], useFixedBase=True)
    # p.resetBasePositionAndOrientation(armId, [0, 0, 0], [0, 0, 0, 1])
    for i, q in enumerate(th):
        p.resetJointState(armId, i, q.item())

    vis = None
    try:
        from base_experiments.env.env import draw_AABB
        from base_experiments.env.pybullet_env import DebugDrawer
        vis = DebugDrawer(1., 1.5)
        vis.toggle_3d(True)
        vis.set_camera_position([-0.1, 0, 0], yaw=-30, pitch=-20)
        # draw_AABB(vis, query_range)
    except:
        pass

    coords, pts = voxel.get_coordinates_and_points_in_grid(0.01, query_range, device=s.device)
    sdf_val, sdf_grad = s(pts)

    norm = matplotlib.colors.Normalize(vmin=sdf_val.min().cpu() - 0.2, vmax=sdf_val.max().cpu())
    # color_map = matplotlib.cm.ScalarMappable(norm=norm)
    # rgb = color_map.to_rgba(sdf_val.reshape(-1).cpu())
    # rgb = rgb[:, :-1]

    surface = sdf_val.abs() < 0.005
    if vis is not None:
        vis.draw_points("surface", pts[surface])

    plt.ion()
    plt.show()

    def fmt(x):
        s = f"{x:.1f}"
        if s.endswith("0"):
            s = f"{x:.0f}"
        if x == 0:
            return "surface"
        return rf"{s}" if plt.rcParams["text.usetex"] else f"{s}"

    ax = plt.gca()
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    x = coords[0]
    z = coords[2]
    v = sdf_val.reshape(len(x), len(z)).transpose(0, 1)
    cset1 = ax.contourf(x, z, v, norm=norm, cmap='Greys_r')
    cset2 = ax.contour(x, z, v, colors='k', levels=[0], linestyles='dashed')
    ax.clabel(cset2, cset2.levels, inline=True, fontsize=13, fmt=fmt)
    plt.draw()


def test_batch_over_configurations():
    urdf = "kuka_iiwa/model.urdf"
    search_path = pybullet_data.getDataPath()
    full_urdf = os.path.join(search_path, urdf)
    chain = pk.build_serial_chain_from_urdf(open(full_urdf).read(), "lbr_iiwa_link_7")
    # d = "cuda" if torch.cuda.is_available() else "cpu"
    d = "cuda"

    chain = chain.to(device=d)
    s = RobotSDF(chain, path_prefix=os.path.join(search_path, "kuka_iiwa"))

    th = torch.tensor([0.0, -math.pi / 4.0, 0.0, math.pi / 2.0, 0.0, math.pi / 4.0, 0.0], device=d)
    N = 200
    th_perturbation = torch.randn(N - 1, 7, device=d) * 0.1
    th = torch.cat((th.view(1, -1), th_perturbation + th))

    s.set_joint_configuration(th)

    y = 0.02
    query_range = np.array([
        [-1, 0.5],
        [y, y],
        [-0.2, 0.8],
    ])

    coords, pts = voxel.get_coordinates_and_points_in_grid(0.01, query_range, device=s.device)

    start = timer()
    all_sdf_val, all_sdf_grad = s(pts)
    elapsed = timer() - start
    logger.info("configurations: %d points: %d elapsed: %fms time per config and point: %fms", N, len(pts),
                elapsed * 1000, elapsed * 1000 / N / len(pts))

    for i in range(N):
        th_i = th[i]
        s.set_joint_configuration(th_i)
        sdf_val, sdf_grad = s(pts)
        assert torch.allclose(sdf_val, all_sdf_val[i])
        assert torch.allclose(sdf_grad, all_sdf_grad[i])


if __name__ == "__main__":
    # test_urdf_to_sdf()
    test_batch_over_configurations()
