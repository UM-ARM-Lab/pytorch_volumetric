import torch
import os
import pytorch_volumetric as pv
import pytorch_kinematics as pk
from pytorch_seed import seed
from pytorch_volumetric import sample_mesh_points
import matplotlib.pyplot as plt
import seaborn as sns

plt.switch_backend('Qt5Agg')
TEST_DIR = os.path.dirname(__file__)

visualize = True


def test_chamfer_distance(mesh):
    d = "cuda" if torch.cuda.is_available() else "cpu"
    device = d
    dtype = torch.float
    B = 300
    seed(3)

    # press n to visualize the normals / gradients
    obj = pv.MeshObjectFactory(os.path.join(TEST_DIR, mesh))

    # sample points on the obj mesh surface uniformly
    pts, normals, _ = sample_mesh_points(obj, name=mesh, num_points=1000, device=d, dtype=dtype)

    # sample a random transform
    gt_tf = pk.Transform3d(pos=torch.randn(3, device=d), rot=pk.random_rotation(device=d), device=d)
    pts_world = gt_tf.transform_points(pts)

    # giving it the transform should have 0 distance
    world_to_object = gt_tf.inverse().get_matrix().repeat(B, 1, 1)

    err = pv.batch_chamfer_dist(world_to_object, pts_world, obj)
    assert err.shape == (B,)
    assert torch.allclose(err, torch.zeros_like(err), atol=1e-4)

    # randomly pertrub the transform
    radian_sigma = 0.1
    translation_sigma = 0.1
    world_to_object_perturbed = torch.eye(4, dtype=dtype, device=device).repeat(B, 1, 1)

    delta_R = torch.randn((B, 3), dtype=dtype, device=device) * radian_sigma
    delta_R = pk.axis_angle_to_matrix(delta_R)
    world_to_object_perturbed[:, :3, :3] = delta_R @ world_to_object[:, :3, :3]
    world_to_object_perturbed[:, :3, 3] = world_to_object[:, :3, 3]

    delta_t = torch.randn((B, 3), dtype=dtype, device=device) * translation_sigma
    world_to_object_perturbed[:, :3, 3] += delta_t

    # compare the chamfer distance to the matrix distance of the transform (norm of their difference)
    err = pv.batch_chamfer_dist(world_to_object_perturbed, pts_world, obj)
    # use the p=2 induced vector norm (spectral norm)
    mat_diff = world_to_object_perturbed - world_to_object
    # get the max singular value of the matrix
    _, singular_values, _ = torch.svd(mat_diff, compute_uv=False)
    mat_norm = singular_values[:, 0]

    if visualize:
        sns.displot(mat_norm.cpu().numpy(), kind="kde", fill=True)
        # plot the mat_norm with respect to the chamfer distance
        plt.figure()
        plt.scatter(mat_norm.cpu().numpy(), err.cpu().numpy())
        plt.xlabel("Matrix Norm")
        plt.ylabel("Chamfer Distance (mm^2)")
        # set x and y min limit to 0
        plt.xlim(0)
        plt.ylim(0)
        plt.show()


if __name__ == "__main__":
    test_chamfer_distance("probe.obj")
    test_chamfer_distance("offset_wrench_nogrip.obj")
