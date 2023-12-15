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


def do_test_chamfer_distance(mesh):
    d = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float
    B = 300
    N = 1000
    seed(3)

    # press n to visualize the normals / gradients
    obj = pv.MeshObjectFactory(os.path.join(TEST_DIR, mesh))

    # sample points on the obj mesh surface uniformly
    pts, normals, _ = sample_mesh_points(obj, name=mesh, num_points=N, device=d, dtype=dtype)

    # sample a random transform
    gt_tf = pk.Transform3d(pos=torch.randn(3, device=d), rot=pk.random_rotation(device=d), device=d)
    pts_world = gt_tf.transform_points(pts)

    # giving it the transform should have 0 distance
    world_to_object = gt_tf.inverse().get_matrix().repeat(B, 1, 1)

    err = pv.batch_chamfer_dist(world_to_object, pts_world, obj)
    assert err.shape == (B,)
    assert torch.allclose(err, torch.zeros_like(err), atol=1e-4)

    # randomly pertrub the transform
    perturbed_tf = gt_tf.sample_perturbations(B, radian_sigma=0.1, translation_sigma=0.1)
    world_to_object_perturbed = perturbed_tf.inverse().get_matrix()

    # ChamferDistance gives the sum of all the distances while we take their mean, so we need to multiply by N
    err = pv.batch_chamfer_dist(world_to_object_perturbed, pts_world, obj, scale=1) * N
    # compare the error to the chamfer distance between the transformed points and the original points
    perturbed_pts = perturbed_tf.transform_points(pts)

    try:
        from chamferdist import ChamferDistance
        d = ChamferDistance()
        gt_dist = d(pts_world.repeat(B, 1, 1), perturbed_pts, reduction=None)
        # gt_dist_r = d(pts_world.repeat(B, 1, 1), perturbed_tf.transform_points(pts), reduction=None, reverse=True)
        # gt_dist_b = 0.5 * d(pts_world.repeat(B, 1, 1), perturbed_tf.transform_points(pts), reduction=None, bidirectional=True)
        # there are some differences because ours compares the unidirectional chamfer distance to a mesh vs point cloud
        # they are always overestimating the distance due to not finding the actual closest point
        assert torch.all(err < gt_dist)
        assert torch.all(gt_dist - err < 0.05 * gt_dist)
    except ImportError:
        print("pip install chamferdist to test against an accelerated implementation of chamfer distance")

    # compare against a manual chamfer distance calculation
    all_dists = torch.cdist(pts_world, perturbed_pts)
    gt_dist_manual = torch.min(all_dists, dim=2).values.square().sum(dim=1)
    assert torch.all(err < gt_dist_manual)
    assert torch.all(gt_dist_manual - err < 0.05 * gt_dist_manual)

    # compare the chamfer distance to the matrix distance of the transform (norm of their difference)
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


def do_test_plausible_diversity(mesh):
    d = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float
    B = 10
    tol = 1e-4
    seed(3)

    # press n to visualize the normals / gradients
    obj = pv.MeshObjectFactory(os.path.join(TEST_DIR, mesh))

    # sample a random transform
    gt_tf = pk.Transform3d(pos=torch.randn(3, device=d), rot=pk.random_rotation(device=d), dtype=dtype, device=d)
    # sample perturbations around it to get a set of plausible gt transforms
    gt_tf = gt_tf.sample_perturbations(B, radian_sigma=0.05, translation_sigma=0.01)

    # plausible diversity of itself should be 0
    pd = pv.PlausibleDiversity(obj)
    pd_ret = pd(gt_tf.inverse().get_matrix(), gt_tf.get_matrix())
    assert pd_ret.plausibility < tol
    assert pd_ret.coverage < tol

    # removing some transforms should keep plausibility at 0, but increase coverage error
    partial_tf = pk.Transform3d(matrix=gt_tf.get_matrix()[:B // 2])
    pd_ret = pd(partial_tf.inverse().get_matrix(), gt_tf.get_matrix(), bidirectional=True)
    assert pd_ret.plausibility < tol
    assert pd_ret.coverage > tol

    # # compare the computed distances against the ground truth
    # pts = pd.model_points_eval
    # # transform the points using both sets of transforms
    # pts_gt = gt_tf.transform_points(pts)
    # pts_partial = partial_tf.transform_points(pts)
    # # compute the chamfer distance between the transformed points

    # going the other way should have the opposite effect
    pd_ret_other = pd(gt_tf.inverse().get_matrix(), partial_tf.get_matrix(), bidirectional=True)
    assert pd_ret_other.plausibility > tol
    assert pd_ret_other.coverage < tol

    # should also be symmetric when created as bidirectional
    # could still have some numerical error due to inverting the matrix
    assert torch.allclose(pd_ret.plausibility, pd_ret_other.coverage)
    assert torch.allclose(pd_ret.coverage, pd_ret_other.plausibility, rtol=0.06)


def test_chamfer_distance():
    do_test_chamfer_distance("probe.obj")
    do_test_chamfer_distance("offset_wrench_nogrip.obj")


def test_plausible_diversity():
    do_test_plausible_diversity("probe.obj")
    do_test_plausible_diversity("offset_wrench_nogrip.obj")


if __name__ == "__main__":
    test_chamfer_distance()
    test_plausible_diversity()
