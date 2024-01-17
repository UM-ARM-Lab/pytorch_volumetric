import copy
import torch
import typing
import numpy as np
import pytorch_kinematics as pk
from pytorch_volumetric import sdf, model_to_sdf
import pytorch_volumetric as pv

from torch.func import jacrev, jacfwd, hessian, vmap
from functools import partial


class RobotScene:

    def __init__(self, robot_sdf: model_to_sdf.RobotSDF, scene_sdf: sdf.ObjectFrameSDF, scene_transform: pk.Transform3d,
                 threshold: float = 0.002, points_per_link: int = 100, softmin_temp: float = 1000,
                 collision_check_links: typing.List[str] = None
                 ):
        """
        :param robot_sdf: the robot sdf
        :param scene_sdf: the scene sdf: Not that this can be a RobotSDF for articulated environments
        :param scene_transform: the transform of from the scene sdf to the robot frame
        :param threshold: the threshold for considering points on the surface
        :param points_per_link: the number of points to sample per link
        :param softmin_temp: the temperature for the softmin to get minimum sdf value
        :param collision_check_links: Robot links on which to generate points for collision checking, defaults
            to all links
        """
        self.robot_sdf = robot_sdf
        self.scene_sdf = scene_sdf
        self.device = self.robot_sdf.device
        self.threshold = threshold
        self.points_per_link = points_per_link
        self.softmin_temp = softmin_temp
        self.desired_link_idx = []
        self.desired_frame_idx = []
        if collision_check_links is not None:
            self.desired_links = collision_check_links
        else:
            self.desired_links = self.robot_sdf.sdf_to_link_name

        self.link_to_actuated_joint_idx = []
        self.num_links = len(self.robot_sdf.sdf_to_link_name)

        self.scene_transform = scene_transform.to(device=self.device)
        self.robot_query_points, self._query_point_mask = self._generate_robot_query_points()

        self.transform_points = vmap(self._transform_points)
        self.transform_to_world = vmap(self._transform_to_world)
        self.transform_world_to_scene = vmap(self._transform_world_to_scene)

        self.grad_smooth_points = 10
        self.grad_points = vmap(jacrev(self._transform_points))
        # self.hess_points = vmap(jacfwd(jacrev(self._transform_points)))

    def _get_desired_tfs(self):
        tfs = self.robot_sdf.sdf.obj_frame_to_link_frame.get_matrix().reshape(self.num_links, -1, 4, 4)
        tfs = tfs[self.desired_link_idx].reshape(-1, 4, 4)
        return pk.Transform3d(matrix=tfs)

    def _generate_robot_query_points(self):
        query_points = []
        for i, link_name in enumerate(self.robot_sdf.sdf_to_link_name):
            if link_name in self.desired_links:
                link_sdf = self.robot_sdf.get_link_sdf(link_name)
                # link_sdf.precompute_sdf()
                # points, _, _ = sdf.sample_mesh_points(link_sdf, self.points_per_link,
                #                                      dbpath=f'{link_name}_points_cache.pkl', device=self.device)
                points, _ = link_sdf.sample_surface_points(self.points_per_link,
                                                           dbpath=f'{link_name}_points_cache.pkl', device=self.device)

                query_points.append(points)
                # TODO: confusing because index from robot_sdf and from chain are not the same, perhaps unify them?
                self.desired_link_idx.append(i)
                self.desired_frame_idx.append(self.robot_sdf.chain.frame_to_idx[link_name])
        self.desired_link_idx = torch.tensor(self.desired_link_idx, device=self.device, dtype=torch.long)
        self.desired_frame_idx = torch.tensor(self.desired_frame_idx, device=self.device, dtype=torch.long)

        query_points = torch.stack(query_points, dim=0)
        # mask out points that are in self-collision with default configuration
        tfs = self._get_desired_tfs().inverse()
        pts = tfs.transform_points(query_points).reshape(-1, 3)
        sdf_vals, _ = self.robot_sdf(pts)
        mask = torch.where(sdf_vals < -self.threshold, torch.zeros_like(sdf_vals), torch.ones_like(sdf_vals))
        return query_points, mask

    def visualize_robot(self, q: torch.Tensor, env_q: torch.Tensor = None):
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        self.robot_sdf.set_joint_configuration(q)
        if env_q is not None:
            self.scene_sdf.set_joint_configuration(env_q)

        tfs = self._get_desired_tfs().inverse()
        pts = tfs.transform_points(self.robot_query_points).reshape(-1, 3)
        pcd.points = o3d.utility.Vector3dVector(pts.cpu().numpy())
        self.scene_sdf.precompute_sdf()
        scene_meshes = self.scene_sdf.get_mesh_list()
        scene_transform = self.scene_transform.get_matrix().cpu().numpy().astype(np.float64).reshape((4, 4))
        scene_meshes = [mesh.transform(scene_transform) for mesh in scene_meshes]
        colors = np.array([
            [0.651, 0.208, 0.290],
            [0.121, 0.470, 0.705],
        ])
        for scene_mesh, c in zip(scene_meshes, colors):
            scene_mesh.paint_uniform_color(c)
        # scene_mesh = self.scene_sdf.obj_factory._mesh.transform(self.scene_transform.get_matrix()[0].cpu().numpy())
        o3d.visualization.draw_geometries(pv.get_transformed_meshes(self.robot_sdf) + [pcd] + scene_meshes,
                                          mesh_show_wireframe=True)

    def _transform_to_world(self, q: torch.Tensor, weight=None):
        """
            Transforms robot query points from link frame to the scene_frame
        :param q: torch.Tensor B x dq joint angles
        :return:
        """
        self.robot_sdf.set_joint_configuration(q)
        tfs = self._get_desired_tfs().inverse()
        batched_query_points = self.robot_query_points
        pts = tfs.transform_points(batched_query_points.reshape(-1, self.points_per_link, 3))
        return pts

    def _transform_world_to_scene(self, pts: torch.Tensor):
        return self.scene_transform.inverse().transform_points(pts)

    def _transform_points(self, q: torch.Tensor, weight=None):
        self.robot_sdf.set_joint_configuration(q)
        tfs = self._get_desired_tfs().inverse()
        batched_query_points = self.robot_query_points
        pts = tfs.transform_points(batched_query_points.reshape(-1, self.points_per_link, 3))
        pts_scene = self.scene_transform.inverse().transform_points(pts)

        if weight is None:
            return pts_scene.reshape(-1, 3)
        else:
            return torch.sum(weight.reshape(-1, 1) * pts_scene.reshape(-1, 3), dim=0)

    def _collision_check(self, q: torch.Tensor, sdf: sdf.ObjectFrameSDF,
                         compute_gradient=False, compute_hessian=False):
        """
        :param q: torch.Tensor B x dq joint angles
        :sdf : function to query SDF
        :param compute_gradient: bool whether to compute gradient of sdf wrt joint angles
        :param compute_hessian: bool whether to compute hessian of sdf wrt joint angles
        """

        # Add leading batch dimension
        if len(q.shape) == 1:
            q = q.reshape(1, -1)
            B = 1
        else:
            B = q.shape[0]

        if compute_hessian and not compute_gradient:
            raise ValueError('Cannot compute hessian without gradient')

        pts_world = self.transform_to_world(q)
        pts = self.transform_world_to_scene(pts_world)

        sdf_result = sdf(pts, return_extra_info=True)
        sdf_vals = sdf_result['sdf_val']
        sdf_grads = sdf_result['sdf_grad']
        sdf_hess = sdf_result['sdf_hess']
        if sdf_hess is not None:
            sdf_hess = sdf_hess.reshape(B, -1, 3, 3)

        # get minimum links
        sdf_vals = sdf_vals.reshape(B, -1)
        sdf_grads = sdf_grads.reshape(B, -1, 3)
        sdf_indices = torch.argsort(sdf_vals, dim=1, descending=False)
        sdf_val = sdf_vals[torch.arange(B), sdf_indices[:, 0]]

        rvals = {
            'sdf': sdf_val,
        }
        if compute_gradient:
            # rather than take a single point we use the softmin to get a weighted average of the gradients for
            # the self.grad_smooth_points number of closest points, this helps smooth the gradient
            B_range = torch.arange(B).unsqueeze(-1)
            closest_indices = sdf_indices[:, :self.grad_smooth_points]
            closest_sdf_grads = sdf_grads[B_range, closest_indices]
            closest_sdf_vals = sdf_vals[B_range, closest_indices]
            h = torch.softmax(-self.softmin_temp * closest_sdf_vals, dim=1)
            new_grad = False
            pts_hessian = None
            if new_grad:
                closest_pts = pts.reshape(B, -1, 3)[B_range, closest_indices].reshape(-1, 3)
                closest_links = self.desired_frame_idx[closest_indices // self.points_per_link].reshape(-1)
                q_repeat = q.unsqueeze(1).repeat(1, self.grad_smooth_points, 1).reshape(B * self.grad_smooth_points, -1)
                if not compute_hessian:
                    pts_jacobian = self.robot_sdf.chain.jacobian(q_repeat,
                                                                 locations=closest_pts,
                                                                 link_indices=closest_links)
                else:
                    pts_jacobian, pts_hessian = self.robot_sdf.chain.jacobian_and_hessian(q_repeat,
                                                                                          locations=closest_pts,
                                                                                          link_indices=closest_links)
                    pts_hessian = pts_hessian[:, :3].reshape(B, self.grad_smooth_points, 3, q.shape[1], q.shape[1])

                pts_jacobian = pts_jacobian[:, :3].reshape(B, self.grad_smooth_points, 3, -1)
                sdf_weighted_grad = h[:, :, None] * closest_sdf_grads
                q_grad = (pts_jacobian.transpose(2, 3) @ sdf_weighted_grad.unsqueeze(-1)).squeeze(-1)
                rvals['grad_sdf'] = torch.sum(q_grad, dim=1)  # B x 14

            else:
                h = torch.softmax(-self.softmin_temp * sdf_vals, dim=1)
                pts_jacobian = self.grad_points(q, h)  # B x 3 x 14
                sdf_weighted_grad = torch.sum(h[:, :, None] * sdf_grads, dim=1)
                q_grad = (pts_jacobian.transpose(1, 2) @ sdf_weighted_grad.unsqueeze(-1)).squeeze(-1)
                rvals['grad_sdf'] = q_grad  # B x 14

            if compute_hessian:
                if pts_hessian is None:
                    pts_hessian = self.hess_points(q, h)
                closest_sdf_hess = sdf_hess[B_range, closest_indices]
                sdf_weighted_hess = h[:, :, None, None] * closest_sdf_hess
                q_hess = torch.sum(pts_hessian * sdf_weighted_grad.reshape(B, -1, 3, 1, 1), dim=2)
                q_hess2 = pts_jacobian.transpose(2, 3) @ sdf_weighted_hess @ pts_jacobian
                q_hess = q_hess + q_hess2
                rvals['hess_sdf'] = torch.sum(q_hess, dim=1)

        return rvals

    def _collision_check_against_robot_sdf(self, q: torch.Tensor, env_q: torch.Tensor,
                                           sdf: model_to_sdf.RobotSDF, compute_gradient=False, compute_hessian=False):
        # This is a function for collision checking when the environment is itself a RobotSDF, i.e. it is
        # an articulated SDF with configuration env_q, vs the robot which has configuration rob_q
        # Add leading batch dimension
        if len(q.shape) == 1:
            q = q.reshape(1, -1)
            B = 1
        else:
            B = q.shape[0]

        if compute_hessian and not compute_gradient:
            raise ValueError('Cannot compute hessian without gradient')

        # set the configuration of the scene
        sdf.set_joint_configuration(env_q)
        pts_world = self.transform_to_world(q)
        pts = self.transform_world_to_scene(pts_world)

        sdf_result = sdf(pts, return_extra_info=True)
        sdf_vals = sdf_result['sdf_val']
        sdf_grads = sdf_result['sdf_grad']
        sdf_hess = sdf_result['sdf_hess']
        # get the index for the closest frame in the environment robotSDF
        sdf_closest_link = sdf_result['closest_sdf']
        sdf_frame_indices = sdf.sdf_index_to_frame_index[sdf_closest_link]
        sdf_frame_indices = sdf_frame_indices.reshape(B, -1)
        sdf_hess = sdf_hess.reshape(B, -1, 3, 3)

        # get minimum links
        sdf_vals = sdf_vals.reshape(B, -1)
        sdf_grads = sdf_grads.reshape(B, -1, 3)
        sdf_indices = torch.argsort(sdf_vals, dim=1, descending=False)
        sdf_val = sdf_vals[torch.arange(B), sdf_indices[:, 0]]

        rvals = {
            'sdf': sdf_val,
        }
        if compute_gradient:
            # rather than take a single point we use the softmin to get a weighted average of the gradients for
            # the self.grad_smooth_points number of closest points, this helps smooth the gradient
            B_range = torch.arange(B).unsqueeze(-1)
            closest_indices = sdf_indices[:, :self.grad_smooth_points]
            closest_sdf_grads = sdf_grads[B_range, closest_indices]
            closest_sdf_vals = sdf_vals[B_range, closest_indices]
            h = torch.softmax(-self.softmin_temp * closest_sdf_vals, dim=1)
            new_grad = True
            pts_hessian = None
            closest_pts = pts.reshape(B, -1, 3)[B_range, closest_indices].reshape(-1, 3)
            closest_links = self.desired_frame_idx[closest_indices // self.points_per_link].reshape(-1)
            q_repeat = q.unsqueeze(1).repeat(1, self.grad_smooth_points, 1).reshape(B * self.grad_smooth_points, -1)

            env_q_repeat = env_q.unsqueeze(1).repeat(1, self.grad_smooth_points, 1).reshape(B * self.grad_smooth_points,
                                                                                            -1)

            closest_env_links = sdf_frame_indices[B_range, closest_indices].reshape(-1)
            if not compute_hessian:
                rob_jacobian = self.robot_sdf.chain.jacobian(q_repeat,
                                                             locations=closest_pts,
                                                             link_indices=closest_links)
                env_jacobian = self.scene_sdf.chain.jacobian(env_q_repeat,
                                                             locations=closest_pts,
                                                             link_indices=closest_env_links)
            else:
                rob_jacobian, rob_hessian = self.robot_sdf.chain.jacobian_and_hessian(q_repeat,
                                                                                      locations=closest_pts,
                                                                                      link_indices=closest_links)
                env_jacobian, env_hessian = self.scene_sdf.chain.jacobian_and_hessian(env_q_repeat,
                                                                                      locations=closest_pts,
                                                                                      link_indices=closest_env_links)

                rob_hessian = rob_hessian[:, :3].reshape(B, self.grad_smooth_points, 3, q.shape[1], q.shape[1])
                env_hessian = env_hessian[:, :3].reshape(B, self.grad_smooth_points, 3, env_q.shape[1], env_q.shape[1])

            rob_jacobian = rob_jacobian[:, :3].reshape(B, self.grad_smooth_points, 3, -1)
            env_jacobian = env_jacobian[:, :3].reshape(B, self.grad_smooth_points, 3, -1)

            sdf_weighted_grad = h[:, :, None] * closest_sdf_grads
            q_grad = (rob_jacobian.transpose(2, 3) @ sdf_weighted_grad.unsqueeze(-1)).squeeze(-1)
            q_env_grad = (env_jacobian.transpose(2, 3) @ -sdf_weighted_grad.unsqueeze(-1)).squeeze(-1)

            rvals['grad_sdf'] = torch.sum(q_grad, dim=1)  # B x 14
            rvals['grad_env_sdf'] = torch.sum(q_env_grad, dim=1)  # B x 14

            if compute_hessian:
                closest_sdf_hess = sdf_hess[B_range, closest_indices]
                sdf_weighted_hess = h[:, :, None, None] * closest_sdf_hess
                q_hess = torch.sum(rob_hessian * sdf_weighted_grad.reshape(B, -1, 3, 1, 1), dim=2)
                q_hess2 = rob_jacobian.transpose(2, 3) @ sdf_weighted_hess @ rob_jacobian
                q_hess = q_hess + q_hess2
                rvals['hess_sdf'] = torch.sum(q_hess, dim=1)
                q_env_hess = torch.sum(env_hessian * -sdf_weighted_grad.reshape(B, -1, 3, 1, 1), dim=2)
                q_env_hess2 = env_jacobian.transpose(2, 3) @ -sdf_weighted_hess @ env_jacobian
                q_env_hess = q_env_hess + q_env_hess2
                rvals['hess_env_sdf'] = torch.sum(q_env_hess, dim=1)
        return rvals

    def scene_collision_check(self, q: torch.Tensor, env_q=None,
                              compute_gradient=False, compute_hessian=False):
        """
           Collision checks robot with scene sdf
           :param q: torch.Tensor B x dq joint angles
           :param compute_gradient: bool whether to compute gradient of sdf wrt joint angles
           :param compute_hessian: bool whether to compute hessian of sdf wrt joint angles
           """
        if isinstance(self.scene_sdf, model_to_sdf.RobotSDF):
            if env_q is None:
                raise ValueError('Must provide environment configuration if scene is articulated SDF')
            return self._collision_check_against_robot_sdf(q, env_q, self.scene_sdf, compute_gradient, compute_hessian)

        return self._collision_check(q, self.scene_sdf, compute_gradient, compute_hessian)

    def self_collision_check(self, q: torch.Tensor, compute_gradient=False, compute_hessian=False):
        """
           Performs self collision check
           :param q: torch.Tensor B x dq joint angles
           :param compute_gradient: bool whether to compute gradient of sdf wrt joint angles
           :param compute_hessian: bool whether to compute hessian of sdf wrt joint angles
           """
        rvals = self._collision_check_against_robot_sdf(q, q, self.robot_sdf, compute_gradient, compute_hessian)
        # Combine gradients
        if compute_gradient:
            rvals['grad_sdf'] += rvals['grad_env_sdf']
            rvals['grad_env_sdf'] = {}
        if compute_hessian:
            rvals['hess_sdf'] += rvals['hess_env_sdf']
            rvals['hess_env_sdf'] = {}
        return rvals