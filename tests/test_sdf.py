import open3d as o3d
import torch
import os
import pytorch_volumetric as pv
from pytorch_volumetric import sample_mesh_points

TEST_DIR = os.path.dirname(__file__)


def do_test_gradients_at_surface_pts(mesh):
    d = "cuda" if torch.cuda.is_available() else "cpu"

    # press n to visualize the normals / gradients
    obj = pv.MeshObjectFactory(os.path.join(TEST_DIR, mesh))
    sdf = pv.MeshSDF(obj)

    # sample points on the obj mesh surface uniformly
    pts, normals, _ = sample_mesh_points(obj, name=mesh, num_points=1000)

    # query the sdf value and gradient at the sampled points
    sdf_vals, sdf_grads = sdf(pts)

    assert torch.allclose(sdf_vals.abs(), torch.zeros_like(sdf_vals), atol=1e-4)

    # test batch query
    batch_pts = pts.view(10, 100, -1)
    batch_sdf_vals, batch_sdf_grads = sdf(batch_pts)
    assert batch_sdf_vals.shape == (10, 100)
    assert torch.allclose(batch_sdf_vals.abs(), torch.zeros_like(batch_sdf_vals), atol=1e-4)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.cpu())
    pcd.normals = o3d.utility.Vector3dVector(normals.cpu())
    pcd.paint_uniform_color([0, 1, 0])
    o3d.visualization.draw_geometries([sdf.obj_factory._mesh, pcd])

    # compare the gradient against the surface normals plotted before - should be the same
    pcd.normals = o3d.utility.Vector3dVector(sdf_grads.cpu())
    # color based on sdf value
    colors = torch.zeros_like(sdf_vals).view(-1, 1).repeat(1, 3)
    colors[:, 0] = (sdf_vals - sdf_vals.min()) / (sdf_vals.max() - sdf_vals.min())
    pcd.colors = o3d.utility.Vector3dVector(colors.cpu())
    o3d.visualization.draw_geometries([sdf.obj_factory._mesh, pcd])

    # sample points in the bounding box
    coords, pts = pv.get_coordinates_and_points_in_grid(0.002, obj.bounding_box(0.01), device=d)
    # randomly downsample to some number of points
    pts = pts[torch.randperm(len(pts))[:1000]]
    # query the sdf value and gradient at the sampled points
    sdf_vals, sdf_grads = sdf(pts)
    # visualize the sdf value and gradient at the sampled points
    pcd.points = o3d.utility.Vector3dVector(pts.cpu())
    pcd.normals = o3d.utility.Vector3dVector(sdf_grads.cpu())
    colors = torch.zeros_like(sdf_vals).view(-1, 1).repeat(1, 3)
    colors[:, 0] = (sdf_vals - sdf_vals.min()) / (sdf_vals.max() - sdf_vals.min())
    colors[:, 1] = 1
    pcd.colors = o3d.utility.Vector3dVector(colors.cpu())
    o3d.visualization.draw_geometries([sdf.obj_factory._mesh, pcd])


def test_compose_sdf():
    import pytorch_kinematics as pk
    d = "cuda" if torch.cuda.is_available() else "cpu"

    obj = pv.MeshObjectFactory("YcbPowerDrill/textured_simple_reoriented.obj")

    # 2 drills in the world
    sdf1 = pv.MeshSDF(obj)
    sdf2 = pv.MeshSDF(obj)
    # need to specify the transform of each SDF frame
    tsf1 = pk.Translate(0.1, 0, 0, device=d)
    tsf2 = pk.Translate(-0.2, 0, 0.2, device=d)
    sdf = pv.ComposedSDF([sdf1, sdf2], tsf1.stack(tsf2))
    # sample points in the bounding box

    coords, pts = pv.get_coordinates_and_points_in_grid(0.002, obj.bounding_box(0.01), device=d)
    # randomly downsample to some number of points
    pts = pts[torch.randperm(len(pts))[:1000]]
    # query the sdf value and gradient at the sampled points
    sdf_vals, sdf_grads = sdf(pts)
    # visualize the sdf value and gradient at the sampled points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.cpu())
    pcd.normals = o3d.utility.Vector3dVector(sdf_grads.cpu())
    colors = torch.zeros_like(sdf_vals).view(-1, 1).repeat(1, 3)
    colors[:, 0] = (sdf_vals - sdf_vals.min()) / (sdf_vals.max() - sdf_vals.min())
    colors[:, 1] = 1
    pcd.colors = o3d.utility.Vector3dVector(colors.cpu())
    o3d.visualization.draw_geometries([sdf1.obj_factory._mesh, pcd])


def test_gradients_at_surface_pts():
    do_test_gradients_at_surface_pts("probe.obj")
    do_test_gradients_at_surface_pts("offset_wrench_nogrip.obj")


if __name__ == "__main__":
    test_gradients_at_surface_pts()
    test_compose_sdf()
