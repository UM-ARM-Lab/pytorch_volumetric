import time
from typing import NamedTuple

import torch
import pytorch_kinematics as pk
from pytorch_kinematics import transforms as tf
from pytorch_kinematics.transforms.rotation_conversions import matrix_to_pos_rot

from pytorch_volumetric.sdf import ObjectFactory, sample_mesh_points, ObjectFrameSDF


def pairwise_distance(world_to_link_tfs: pk.Transform3d):
    m = world_to_link_tfs.get_matrix()
    t = m[:, :3, 3]
    r = pk.rotation_conversions.matrix_to_rotation_6d(m[:, :3, :3])
    cont_rep = torch.cat((t, r), dim=1)
    return torch.cdist(cont_rep, cont_rep)


def pairwise_distance_chamfer(world_to_link_tfs: pk.Transform3d, obj_factory: ObjectFactory = None,
                              obj_sdf: ObjectFrameSDF = None,
                              model_points_eval: torch.tensor = None, vis=None, scale=1000):
    """
    Compute the pairwise chamfer distance between a set of world to link transforms of an object.
    :param world_to_link_tfs: B x 4 x 4 world to link frame transforms
    :param obj_factory: object factory to evaluate against
    :param obj_sdf: object sdf to evaluate against (accelerates the computation at the cost of accuracy)
    :param model_points_eval: points to evaluate the chamfer distance on; if None, sample from the object
    :param vis: visualizer
    :param scale: units with respect to the position units; e.g. if the position units are in meters, then the scale=1
    :return: B x B chamfer distance matrix in squared distance units, averaged across the model_points_eval
    """
    if model_points_eval is None:
        model_points_eval, _, _ = sample_mesh_points(obj_factory, num_points=500, name=obj_factory.name,
                                                     device=world_to_link_tfs.device)
    B = len(world_to_link_tfs)
    P = B

    # effectively can apply one transform then take the inverse using the other one; if they are the same, then
    # we should end up in the base frame if that T == Tp
    # want pairwise matrix multiplication |T| x |Tp| x 4 x 4 T[0]@Tp[0], T[0]@Tp[1]
    T = world_to_link_tfs.get_matrix()
    # this is fastor than inverting the 4x4 since we can exploit orthogonality
    T_inv = world_to_link_tfs.inverse().get_matrix()
    Iapprox = torch.einsum("bij,pjk->bpik", T_inv, T)
    # the einsum does the multiplication below and is about twice as fast
    # Iapprox = T.inverse().view(-1, 1, 4, 4) @ T.view(1, -1, 4, 4)

    errors_per_batch = batch_chamfer_dist(Iapprox.reshape(B * P, 4, 4), model_points_eval,
                                          obj_factory=obj_factory, obj_sdf=obj_sdf,
                                          viewing_delay=0, vis=vis, scale=scale)
    errors_per_batch = errors_per_batch.view(B, P)
    return errors_per_batch


def batch_chamfer_dist(world_to_object: torch.tensor, model_points_world_frame_eval: torch.tensor,
                       obj_factory: ObjectFactory = None, obj_sdf: ObjectFrameSDF = None, viewing_delay=0, scale=1000.,
                       print_err=False, vis=None):
    """
    Compute batched unidirectional chamfer distance between the observed world frame surface points and the
    surface points of the object transformed by a set of 4x4 rigid transform matrices (from the object frame).
    :param world_to_object: B x 4 x 4 transformation matrices from world to object frame
    :param model_points_world_frame_eval: N x 3 points to evaluate the chamfer distance on
    :param obj_factory: object to evaluate against
    :param obj_sdf: sdf of the object to evaluate against (potentially much faster than obj_sdf, but less accurate)
    :param viewing_delay: if a visualizer is given, sleep between plotting successive elements
    :param scale: units with respect to the position units; e.g. if the position units are in meters, then the scale
    being 1000 will convert the distance to mm
    :param print_err: whether to visualize and print the chamfer error
    :param vis: optional visualizer
    :return: B chamfer error for each transform, averaged across the N points to evaluate against
    """
    B = world_to_object.shape[0]
    eval_num_points = model_points_world_frame_eval.shape[0]
    world_to_link = tf.Transform3d(matrix=world_to_object)
    model_points_object_frame_eval = world_to_link.transform_points(model_points_world_frame_eval)

    if obj_sdf is not None:
        d, _ = obj_sdf(model_points_object_frame_eval)
    elif obj_factory is not None:
        res = obj_factory.object_frame_closest_point(model_points_object_frame_eval)
        d = res.distance
    else:
        raise ValueError("Either obj_sdf or obj_factory must be given")
    # convert to scale such as mm**2
    chamfer_distance = (scale * d) ** 2
    # average across the evaluation points
    errors_per_batch = chamfer_distance.mean(dim=-1)

    if vis is not None and obj_factory is not None:
        link_to_world = world_to_link.inverse()
        res = obj_factory.object_frame_closest_point(model_points_object_frame_eval)
        closest_pt_world_frame = link_to_world.transform_points(res.closest)
        m = link_to_world.get_matrix()
        for b in range(B):
            pos, rot = matrix_to_pos_rot(m[b])
            obj_factory.draw_mesh(vis, "chamfer evaluation", (pos, rot), rgba=(0, 0.1, 0.8, 0.1),
                                  object_id=vis.USE_DEFAULT_ID_FOR_NAME)
            # vis.draw_point("avgerr", [0, 0, 0], (1, 0, 0), label=f"avgerr: {round(errors_per_batch[b].item())}")

            if print_err:
                for i in range(eval_num_points):
                    query = model_points_world_frame_eval[i].cpu()
                    closest = closest_pt_world_frame[b, i].cpu()
                    vis.draw_point(f"query.{i}", query, (0, 1, 0))
                    vis.draw_point(f"closest.{i}", closest, (0, 1, 1))
                    vis.draw_2d_line(f"qc.{i}", query, closest - query, (0, 1, 0), scale=1)

            time.sleep(viewing_delay)

        # move somewhere far away
        obj_factory.draw_mesh(vis, "chamfer evaluation", ([0, 0, 100], [0, 0, 0, 1]), rgba=(0, 0.2, 0.8, 0.2),
                              object_id=vis.USE_DEFAULT_ID_FOR_NAME)
    return errors_per_batch


class PlausibleDiversityReturn(NamedTuple):
    plausibility: torch.tensor
    coverage: torch.tensor
    most_plausible_per_estimated: torch.tensor
    most_covered_per_plausible: torch.tensor


class PlausibleDiversity:
    """
    Compute the plausibility and coverage of an estimated set of transforms against a set of plausible transforms.
    Each transform is a 4x4 homogeneous matrix. The return are in units of squared coordinates (for example if points
    are given with their coordinates in mm, then the return is in mm^2). In some sense it is a set divergence.
    """

    def __init__(self, obj_factory: ObjectFactory, model_points_eval: torch.tensor = None, num_model_points_eval=500,
                 obj_sdf: ObjectFrameSDF = None):
        self.obj_factory = obj_factory
        self.obj_sdf = obj_sdf
        # if model points are not given, sample them from the object
        if model_points_eval is None:
            model_points_eval, _, _ = sample_mesh_points(obj_factory, num_points=num_model_points_eval,
                                                         name=obj_factory.name)
        self.model_points_eval = model_points_eval

    def __call__(self, T_est_inv, T_p, bidirectional=False, scale=1000.):
        """
        Compute the plausibility and coverage of an estimated set of transforms against a set of plausible transforms.

        :param T_est_inv: The inverse of the estimated transforms, in the form of Bx4x4 homogeneous matrices.
        :param T_p: The plausible transforms, in the form of Px4x4 homogeneous matrices.
        :return: plausibility score, coverage score, most_plausible_per_estimated, most_covered_per_plausible
        """
        errors_per_batch = self.compute_tf_pairwise_error_per_batch(T_est_inv, T_p, scale=scale)
        ret = self.do_evaluate_plausible_diversity_on_pairwise_chamfer_dist(errors_per_batch)
        if bidirectional:
            errors_per_batch_rev = self.compute_tf_pairwise_error_per_batch(T_p, T_est_inv, scale=scale)
            ret2 = self.do_evaluate_plausible_diversity_on_pairwise_chamfer_dist(errors_per_batch_rev)
            # the plausibility and coverage are flipped when we reverse the transforms
            ret = PlausibleDiversityReturn(
                plausibility=(ret.plausibility + ret2.coverage) / 2,
                coverage=(ret.coverage + ret2.plausibility) / 2,
                most_plausible_per_estimated=ret.most_plausible_per_estimated,
                most_covered_per_plausible=ret.most_covered_per_plausible,
            )
        return ret

    def compute_tf_pairwise_error_per_batch(self, T_est_inv, T_p, scale=1000.):
        # effectively can apply one transform then take the inverse using the other one; if they are the same, then
        # we should end up in the base frame if that T == Tp
        # want pairwise matrix multiplication |T| x |Tp| x 4 x 4 T[0]@Tp[0], T[0]@Tp[1]
        Iapprox = torch.einsum("bij,pjk->bpik", T_est_inv, T_p)
        # the einsum does the multiplication below and is about twice as fast
        # Iapprox = T_est_inv.view(-1, 1, 4, 4) @ T_p.view(1, -1, 4, 4)

        B, P = Iapprox.shape[:2]
        self.model_points_eval = self.model_points_eval.to(device=Iapprox.device, dtype=Iapprox.dtype)
        errors_per_batch = batch_chamfer_dist(Iapprox.reshape(B * P, 4, 4), self.model_points_eval,
                                              self.obj_factory, obj_sdf=self.obj_sdf, viewing_delay=0, vis=None,
                                              scale=scale)
        errors_per_batch = errors_per_batch.view(B, P)
        return errors_per_batch

    @staticmethod
    def do_evaluate_plausible_diversity_on_pairwise_chamfer_dist(errors_per_batch):
        B, P = errors_per_batch.shape

        best_per_sampled = errors_per_batch.min(dim=1)
        best_per_plausible = errors_per_batch.min(dim=0)

        bp_plausibility = best_per_sampled.values.sum() / B
        bp_coverage = best_per_plausible.values.sum() / P

        return PlausibleDiversityReturn(bp_plausibility, bp_coverage, best_per_sampled, best_per_plausible)
