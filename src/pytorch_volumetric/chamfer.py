import time

import torch
from pytorch_kinematics import transforms as tf
from pytorch_kinematics.transforms.rotation_conversions import matrix_to_pos_rot

from pytorch_volumetric.sdf import ObjectFactory


def batch_chamfer_dist(T: torch.tensor, model_points_world_frame_eval: torch.tensor,
                       obj_factory: ObjectFactory, viewing_delay=0, print_err=False, vis=None):
    """
    Compute batched unidirectional batched chamfer distance.
    :param T: B x 4 x 4 transformation matrices from world to object frame
    :param model_points_world_frame_eval: N x 3 points to evaluate the chamfer distance on
    :param obj_factory: object to evaluate against
    :param viewing_delay: if a visualizer is given, sleep between plotting successive elements
    :param print_err: whether to visualize and print the chamfer error
    :param vis: optional visualizer
    :return: B chamfer error for each transform, averaged across the N points to evaluate against
    """
    # due to inherent symmetry, can't just use the known correspondence to measure error, since it's ok to mirror
    # we're essentially measuring the chamfer distance (acts on 2 point clouds), where one point cloud is the
    # evaluation model points on the ground truth object surface, and the surface points of the object transformed
    # by our estimated pose (which is infinitely dense)
    # this is the unidirectional chamfer distance since we're only measuring dist of eval points to surface
    B = T.shape[0]
    eval_num_points = model_points_world_frame_eval.shape[0]
    world_to_link = tf.Transform3d(matrix=T)
    link_to_world = world_to_link.inverse()
    model_points_object_frame_eval = world_to_link.transform_points(model_points_world_frame_eval)

    res = obj_factory.object_frame_closest_point(model_points_object_frame_eval)
    closest_pt_world_frame = link_to_world.transform_points(res.closest)
    # closest_pt_world_frame = closest_pt_object_frame
    # convert to mm**2
    chamfer_distance = (1000 * res.distance) ** 2
    # average across the evaluation points
    errors_per_batch = chamfer_distance.mean(dim=-1)

    if vis is not None:
        m = link_to_world.get_matrix()
        for b in range(B):
            pos, rot = matrix_to_pos_rot(m[b])
            obj_factory.draw_mesh(vis, "chamfer evaluation", (pos, rot), rgba=(0, 0.1, 0.8, 0.1),
                                  object_id=vis.USE_DEFAULT_ID_FOR_NAME)
            vis.draw_point("avgerr", [0, 0, 0], (1, 0, 0), label=f"avgerr: {round(errors_per_batch[b].item())}")

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
