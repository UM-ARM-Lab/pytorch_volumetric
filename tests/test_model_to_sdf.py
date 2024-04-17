import os
import math
import torch
import time
from matplotlib import pyplot as plt
import numpy as np
from timeit import default_timer as timer
import open3d as o3d
import pytorch_kinematics as pk
import pytorch_volumetric as pv

import pybullet as p
import pybullet_data

import logging

plt.switch_backend('Qt5Agg')

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO, force=True,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')

TEST_DIR = os.path.dirname(__file__)

visualize = True


def test_urdf_to_sdf():
    visualization = "open3d"
    urdf = "kuka_iiwa/model.urdf"
    search_path = pybullet_data.getDataPath()
    full_urdf = os.path.join(search_path, urdf)
    chain = pk.build_serial_chain_from_urdf(open(full_urdf).read(), "lbr_iiwa_link_7")
    d = "cuda" if torch.cuda.is_available() else "cpu"

    chain = chain.to(device=d)
    # use MeshSDF or CachedSDF for much faster lookup
    s = pv.RobotSDF(chain, path_prefix=os.path.join(search_path, "kuka_iiwa"), )
    # link_sdf_cls=pv.cache_link_sdf_factory(resolution=0.02, padding=0.1, device=d))
    th = torch.tensor([0.0, -math.pi / 4.0, 0.0, math.pi / 2.0, 0.0, math.pi / 4.0, 0.0], device=d)

    s.set_joint_configuration(th)

    y = 0.02
    query_range = np.array([
        [-1, 0.5],
        [y, y],
        [-0.2, 0.8],
    ])

    plt.ion()
    plt.show()

    if visualize:
        ret = pv.draw_sdf_slice(s, query_range, resolution=0.01, device=s.device)
        sdf_val = ret[0]
        pts = ret[2]

        surface = sdf_val.abs() < 0.005

        if visualization == "pybullet":
            # toggles - g:GUI w:wireframe j:joint axis a:AABB i:interrupt
            p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.setAdditionalSearchPath(search_path)
            armId = p.loadURDF(urdf, [0, 0, 0], useFixedBase=True)
            # p.resetBasePositionAndOrientation(armId, [0, 0, 0], [0, 0, 0, 1])
            for i, q in enumerate(th):
                p.resetJointState(armId, i, q.item())

            try:
                from base_experiments.env.env import draw_ordered_end_points
                from base_experiments.env.pybullet_env import DebugDrawer
                vis = DebugDrawer(1., 1.5)
                vis.toggle_3d(True)
                vis.set_camera_position([-0.1, 0, 0], yaw=-30, pitch=-20)
                # draw bounding box for each link (set breakpoints here to better see the link frame bounding box)
                tfs = s.sdf.obj_frame_to_link_frame.inverse()
                for i in range(len(th)):
                    sdf = s.sdf.sdfs[i]
                    aabb = pv.aabb_to_ordered_end_points(np.array(sdf.ranges))
                    aabb = tfs.transform_points(torch.tensor(aabb, device=tfs.device, dtype=tfs.dtype))[i]
                    draw_ordered_end_points(vis, aabb)
                    time.sleep(0.2)

                vis.draw_points("surface", pts[surface])
            except:
                pass
            finally:
                p.disconnect()
        elif visualization == "open3d":
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts[surface].cpu().numpy())
            if visualize:
                o3d.visualization.draw_geometries(pv.get_transformed_meshes(s) + [pcd])


def test_batch_over_configurations():
    urdf = "kuka_iiwa/model.urdf"
    search_path = pybullet_data.getDataPath()
    full_urdf = os.path.join(search_path, urdf)
    chain = pk.build_serial_chain_from_urdf(open(full_urdf).read(), "lbr_iiwa_link_7")
    d = "cuda" if torch.cuda.is_available() else "cpu"

    chain = chain.to(device=d)
    s = pv.RobotSDF(chain, path_prefix=os.path.join(search_path, "kuka_iiwa"),
                    link_sdf_cls=pv.cache_link_sdf_factory(resolution=0.02, padding=1.0, device=d))

    th = torch.tensor([0.0, -math.pi / 4.0, 0.0, math.pi / 2.0, 0.0, math.pi / 4.0, 0.0], device=d)
    N = 20
    th_perturbation = torch.randn(N - 1, 7, device=d) * 0.1
    th = torch.cat((th.view(1, -1), th_perturbation + th))

    s.set_joint_configuration(th)

    y = 0.02
    query_range = np.array([
        [-1, 0.5],
        [y, y],
        [-0.2, 0.8],
    ])

    coords, pts = pv.get_coordinates_and_points_in_grid(0.01, query_range, device=s.device)

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
        assert torch.allclose(sdf_grad, all_sdf_grad[i], atol=1e-6)


def test_bounding_box():
    urdf = "kuka_iiwa/model.urdf"
    search_path = pybullet_data.getDataPath()
    full_urdf = os.path.join(search_path, urdf)
    chain = pk.build_serial_chain_from_urdf(open(full_urdf).read(), "lbr_iiwa_link_7")
    d = "cuda" if torch.cuda.is_available() else "cpu"

    chain = chain.to(device=d)
    # use MeshSDF or CachedSDF for much faster lookup
    s = pv.RobotSDF(chain, path_prefix=os.path.join(search_path, "kuka_iiwa"), )
    th = torch.tensor([0.0, -math.pi / 4.0, 0.0, math.pi / 2.0, 0.0, math.pi / 4.0, 0.0], device=d)

    s.set_joint_configuration(th)

    # toggles - g:GUI w:wireframe j:joint axis a:AABB i:interrupt
    p.connect(p.GUI if visualize else p.DIRECT)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.setAdditionalSearchPath(search_path)
    armId = p.loadURDF(urdf, [0, 0, 0], useFixedBase=True)
    # p.resetBasePositionAndOrientation(armId, [0, 0, 0], [0, 0, 0, 1])
    for i, q in enumerate(th):
        p.resetJointState(armId, i, q.item())

    if visualize:
        try:
            from base_experiments.env.env import draw_ordered_end_points, draw_AABB
            from base_experiments.env.pybullet_env import DebugDrawer
            delay = 0.2
            vis = DebugDrawer(1., 1.5)
            vis.toggle_3d(True)
            vis.set_camera_position([-0.1, 0, 0], yaw=-30, pitch=-20)
            # draw bounding box for each link (set breakpoints here to better see the link frame bounding box)
            bbs = s.link_bounding_boxes()
            for i in range(len(s.sdf.sdfs)):
                bb = bbs[i]
                draw_ordered_end_points(vis, bb)
                time.sleep(delay)
            # total aabb
            aabb = s.surface_bounding_box(padding=0)
            draw_AABB(vis, aabb.cpu().numpy())
            time.sleep(delay)
        except ImportError as e:
            print(e)

        time.sleep(1)
    p.disconnect()


def test_single_link_robot():
    full_urdf = os.path.join(TEST_DIR, 'offset_wrench.urdf')
    chain = pk.build_serial_chain_from_urdf(open(full_urdf).read(), "offset_wrench")
    d = "cuda" if torch.cuda.is_available() else "cpu"

    chain = chain.to(device=d)
    # paths to the link meshes are specified with their relative path inside the URDF
    # we need to give them the path prefix as we need their absolute path to load
    sdf = pv.RobotSDF(chain, path_prefix=TEST_DIR,
                      link_sdf_cls=pv.cache_link_sdf_factory(resolution=0.001, padding=0.05, device=d))
    trans_x, trans_y, trans_z = 0.0, 0.0, 0.0
    rot_x, rot_y, rot_z = 0.0, 0.0, 0.0
    trans = torch.tensor([trans_x, trans_y, trans_z], device=d)
    rot = torch.tensor([rot_x, rot_y, rot_z], device=d)
    H = torch.eye(4, device=d)
    H[:-1, -1] = trans
    H[:-1, :-1] = pk.euler_angles_to_matrix(rot, 'XYZ')

    th = torch.cat((trans, rot), dim=0)
    sdf.set_joint_configuration(th.view(1, -1))
    query_range = sdf.surface_bounding_box(padding=0.05)[0]
    # M x 3 points
    coords, pts = pv.get_coordinates_and_points_in_grid(0.001, query_range, device=sdf.device)

    sdf_val, sdf_grad = sdf(pts)
    # because we passed in th with size (1, 6), the output is also (1, M) and (1, M, 3) meaning we have a batch of 1
    sdf_val = sdf_val[0]
    sdf_grad = sdf_grad[0]
    near_surface = sdf_val.abs() < 0.001
    surf_pts = pts[near_surface]
    surf_norms = sdf_grad[near_surface]
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(surf_pts.cpu())
    pc.normals = o3d.utility.Vector3dVector(surf_norms.cpu())
    if visualize:
        o3d.visualization.draw_geometries([pc])

    # test multiple joint configurations
    B = 5
    th = th.view(1, -1).repeat(B, 1)
    sdf.set_joint_configuration(th)
    query_range = sdf.surface_bounding_box(padding=0.05)
    assert query_range.shape == (B, 3, 2)
    for i in range(1, B):
        assert torch.allclose(query_range[0], query_range[i])

    # test non-batch query when we have a batch of configurations
    BB = 10
    N = 100
    assert surf_pts.shape[0] > BB * N
    test_pts = surf_pts[:BB * N]
    sdf_vals, sdf_grads = sdf(test_pts)

    assert sdf_vals.shape == (B, BB * N)
    assert sdf_grads.shape == (B, BB * N, 3)
    assert torch.allclose(sdf_vals.abs(), torch.zeros_like(sdf_vals), atol=1e-3)

    # test batch query when we have a batch of configurations
    batch_pts = test_pts.view(BB, N, 3)
    # will return with batch order Configuration x Point Query Batch x Num data point
    batch_sdf_vals, batch_sdf_grads = sdf(batch_pts)
    assert batch_sdf_vals.shape == (B, BB, N)
    assert batch_sdf_grads.shape == (B, BB, N, 3)
    assert torch.allclose(batch_sdf_vals, sdf_vals.view(B, BB, N))


if __name__ == "__main__":
    test_urdf_to_sdf()
    test_batch_over_configurations()
    test_bounding_box()
    test_single_link_robot()
