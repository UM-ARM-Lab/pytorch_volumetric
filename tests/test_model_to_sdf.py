import pymeshlab
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

import matplotlib
from matplotlib import cm

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


def _make_robot_translucent(robot_id, alpha=0.4):
    def make_transparent(link):
        link_id = link[1]
        rgba = list(link[7])
        rgba[3] = alpha
        p.changeVisualShape(robot_id, link_id, rgbaColor=rgba)

    visual_data = p.getVisualShapeData(robot_id)
    for link in visual_data:
        make_transparent(link)


def test_urdf_to_sdf():
    # visualization = "open3d"
    # visualization = "pybullet"
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

    # plt.ion()
    # plt.show()

    if visualize:
        # toggles - g:GUI w:wireframe j:joint axis a:AABB i:interrupt
        p.connect(p.GUI)
        # record video
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setAdditionalSearchPath(search_path)
        armId = p.loadURDF(urdf, [0, 0, 0], useFixedBase=True)
        # p.resetBasePositionAndOrientation(armId, [0, 0, 0], [0, 0, 0, 1])
        for i, q in enumerate(th):
            p.resetJointState(armId, i, q.item())

        _make_robot_translucent(armId, alpha=0.5)

        from base_experiments.env.pybullet_env import DebugDrawer
        vis = DebugDrawer(0.5, 0.8)
        vis.toggle_3d(True)
        vis.set_camera_position([-0.2, 0, 0.3], yaw=-70, pitch=-15)

        p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "video.mp4")

        # expected SDF value range
        norm = matplotlib.colors.Normalize(vmin=-0.15, vmax=0.35)
        m = cm.ScalarMappable(norm=norm, cmap=cm.jet)
        sdf_id = None

        # input("position the camera and press enter to start the animation")

        # sweep across to animate it
        for x in np.linspace(-0.8, 0.5, 100):
            query_range = np.array([
                [x, x],
                [-0.3, 0.3],
                [-0.2, 0.8],
            ])

            resolution = 0.01
            # ret = pv.draw_sdf_slice(s, query_range, resolution=resolution, device=s.device, do_plot=False)
            # sdf_val = ret[0]
            # pts = ret[2]
            coords, pts = pv.get_coordinates_and_points_in_grid(resolution, query_range, device=s.device)

            sdf_val, sdf_grad = s(pts + torch.randn_like(pts) * 0e-6)
            sdf_color = m.to_rgba(sdf_val.cpu())

            # Assume grid dimensions (rows and cols). You need to know or compute these.
            rows = len(coords[2])
            cols = len(coords[1])

            # Generating the faces (triangulating the grid cells)
            faces = []
            # face_colors = []
            for c in range(cols - 1):
                for r in range(rows - 1):
                    # Compute indices of the vertices of the quadrilateral cell in column-major order
                    idx0 = c * rows + r
                    idx1 = c * rows + (r + 1)
                    idx2 = (c + 1) * rows + (r + 1)
                    idx3 = (c + 1) * rows + r

                    # Two triangles per grid cell
                    faces.append([idx0, idx1, idx2])
                    faces.append([idx0, idx2, idx3])
                    # color is average of the 3 vertices
                    # face_colors.append(sdf_color[[idx0, idx1, idx2]].mean(axis=0))
                    # face_colors.append(sdf_color[[idx0, idx2, idx3]].mean(axis=0))
            faces = np.array(faces)

            # surface = sdf_val.abs() < 0.005
            # set alpha
            # sdf_color[:, -1] = 0.5

            # TODO draw mesh of level set
            # Create the mesh
            this_mesh = pymeshlab.Mesh(vertex_matrix=pts.cpu().numpy(), face_matrix=faces,
                                       v_color_matrix=sdf_color,
                                       # f_color_matrix=face_colors
                                       )
            # create and save mesh
            ms = pymeshlab.MeshSet()
            ms.add_mesh(this_mesh, "sdf")

            # UV map and turn vertex coloring into a texture
            base_name = f"sdf_{x}"
            ms.compute_texcoord_parametrization_triangle_trivial_per_wedge()
            ms.compute_texmap_from_color(textname=f"tex_{base_name}")

            print(f"has vertex colors: {this_mesh.has_vertex_color()}")
            print(f"has face colors: {this_mesh.has_face_color()}")
            print(f"has vertex tex coords: {this_mesh.has_vertex_tex_coord()}")

            # Check vertex colors are set correctly
            assert ms.current_mesh().vertex_number() == len(sdf_color), "Mismatch in vertex counts"
            # check vertex colors
            mvc = ms.current_mesh().vertex_color_matrix()
            print(mvc.shape)
            print(mvc)
            print(sdf_color.shape)
            print(sdf_color)
            # assert np.allclose(mvc, sdf_color), "Mismatch in vertex colors"

            # fn = os.path.join(cfg.DATA_DIR, "shape_explore", f"mesh_{base_name}.obj")
            fn = os.path.join(".", f"mesh_{base_name}.obj")
            ms.save_current_mesh(fn, save_vertex_color=True)

            prev_sdf_id = sdf_id
            visId = p.createVisualShape(p.GEOM_MESH, fileName=fn)
            sdf_id = p.createMultiBody(0, baseVisualShapeIndex=visId, basePosition=[0, 0, 0])
            if prev_sdf_id is not None:
                p.removeBody(prev_sdf_id)

            # time.sleep(0.1)

            # # first plot the points
            # from base_experiments.env.pybullet_env import DebugDrawer
            # vis = DebugDrawer(1., 1.5)
            # vis.toggle_3d(True)
            #
            # vis.draw_points("sdf", pts.cpu().numpy(), color=sdf_color[:, :3])


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
    if visualize:
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
    # test_batch_over_configurations()
    # test_bounding_box()
    # test_single_link_robot()
