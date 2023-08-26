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
        :param scene_sdf: the scene sdf
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
        if collision_check_links is not None:
            self.desired_links = collision_check_links
        else:
            self.desired_links = self.robot_sdf.sdf_to_link_name

        self.num_links = len(self.robot_sdf.sdf_to_link_name)

        self.scene_transform = scene_transform.to(device=self.device)
        self.robot_query_points, self._query_point_mask = self._generate_robot_query_points()

        self.transform_points = vmap(self._transform_points)
        self.grad_points = vmap(jacrev(self._transform_points))
        self.hess_points = vmap(jacfwd(jacrev(self._transform_points)))

    def _get_desired_tfs(self):
        tfs = self.robot_sdf.sdf.obj_frame_to_link_frame.get_matrix().reshape(self.num_links, -1, 4, 4)
        tfs = tfs[self.desired_link_idx].reshape(-1, 4, 4)
        return pk.Transform3d(matrix=tfs)

    def _generate_robot_query_points(self):
        query_points = []
        for i, link_name in enumerate(self.robot_sdf.sdf_to_link_name):
            if link_name in self.desired_links:
                link_sdf = self.robot_sdf.get_link_obj_factory(link_name)
                link_sdf.precompute_sdf()
                points, _, _ = sdf.sample_mesh_points(link_sdf, self.points_per_link,
                                                      dbpath=f'{link_name}_points_cache.pkl', device=self.device)
                query_points.append(points)
                self.desired_link_idx.append(i)

        query_points = torch.stack(query_points, dim=0)
        # mask out points that are in self-collision with default configuration
        tfs = self._get_desired_tfs().inverse()
        pts = tfs.transform_points(query_points).reshape(-1, 3)
        sdf_vals, _ = self.robot_sdf(pts)
        mask = torch.where(sdf_vals < -self.threshold, torch.zeros_like(sdf_vals), torch.ones_like(sdf_vals))
        return query_points, mask

    def visualize_robot(self, q: torch.Tensor):
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        self.robot_sdf.set_joint_configuration(q)
        tfs = self._get_desired_tfs().inverse()
        pts = tfs.transform_points(self.robot_query_points).reshape(-1, 3)
        pcd.points = o3d.utility.Vector3dVector(pts.cpu().numpy())
        self.scene_sdf.precompute_sdf()
        scene_meshes = self.scene_sdf.get_mesh_list()
        scene_transform = self.scene_transform.get_matrix().cpu().numpy().astype(np.float64).reshape((4, 4))
        scene_meshes = [mesh.transform(scene_transform) for mesh in scene_meshes]

        # scene_mesh = self.scene_sdf.obj_factory._mesh.transform(self.scene_transform.get_matrix()[0].cpu().numpy())
        o3d.visualization.draw_geometries(pv.get_transformed_meshes(self.robot_sdf) + [pcd] + scene_meshes)

    def _transform_points(self, q: torch.Tensor, weight=None):
        self.robot_sdf.set_joint_configuration(q)
        tfs = self._get_desired_tfs().inverse()
        batched_query_points = self.robot_query_points
        pts = tfs.transform_points(batched_query_points.reshape(-1, self.points_per_link, 3))
        pts = self.scene_transform.inverse().transform_points(pts)
        if weight is None:
            return pts.reshape(-1, 3)
        else:
            return torch.sum(weight.reshape(-1, 1) * pts.reshape(-1, 3), dim=0)

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

        if compute_gradient:
            q.requires_grad_(True)
            q.retain_grad()
        pts = self.transform_points(q)

        if compute_hessian:
            sdf_vals, sdf_grads, sdf_hess = sdf.get_hessian(pts)
            sdf_hess = sdf_hess.reshape(B, -1, 3, 3)
        else:
            sdf_vals, sdf_grads = sdf(pts)
            sdf_hess = None

        sdf_vals = sdf_vals.reshape(B, -1)
        sdf_grads = sdf_grads.reshape(B, -1, 3)
        # use softmin to get gradient but use hard min for actual sdf value
        h = torch.softmax(-self.softmin_T * sdf_vals, dim=1)

        sdf_val = torch.min(sdf_vals, dim=1).values

        rvals = {
            'sdf': sdf_val,
        }
        if compute_gradient:
            # pts_jacobian = torch.sum(h[:, :, None, None] * self.grad_points(q, h), dim=1)  # B x 3 x 14
            pts_jacobian = self.grad_points(q, h)  # B x 3 x 14
            sdf_weighted_grad = torch.sum(h[:, :, None] * sdf_grads, dim=1)  # B x 3
            q_grad = (pts_jacobian.transpose(1, 2) @ sdf_weighted_grad.unsqueeze(-1)).squeeze(-1)
            rvals['grad_sdf'] = q_grad  # B x 14

            if compute_hessian:
                # hess_pts = torch.sum(h[:, :, None, None, None] * self.hess_points(q, h), dim=1)
                hess_pts = self.hess_points(q, h)
                sdf_weighted_hess = torch.sum(h[:, :, None, None] * sdf_hess, dim=1)
                q_hess = torch.sum(hess_pts * sdf_weighted_grad.reshape(B, 3, 1, 1), dim=1)
                q_hess2 = pts_jacobian.transpose(1, 2) @ sdf_weighted_hess @ pts_jacobian
                q_hess = q_hess + q_hess2
                rvals['hess_sdf'] = q_hess

            self.robot_query_points.detach_()
            self._query_point_mask.requires_grad_(False)
            q = q.detach()

        return rvals

    def scene_collision_check(self, q: torch.Tensor, compute_gradient=False, compute_hessian=False):
        """
           Collision checks robot with scene sdf
           :param q: torch.Tensor B x dq joint angles
           :param compute_gradient: bool whether to compute gradient of sdf wrt joint angles
           :param compute_hessian: bool whether to compute hessian of sdf wrt joint angles
           """
        return self._collision_check(q, self.scene_sdf, compute_gradient, compute_hessian)

    def self_collision_check(self, q: torch.Tensor, compute_gradient=False, compute_hessian=False):
        """
           Performs self collision check
           :param q: torch.Tensor B x dq joint angles
           :param compute_gradient: bool whether to compute gradient of sdf wrt joint angles
           :param compute_hessian: bool whether to compute hessian of sdf wrt joint angles
           """
        return self._collision_check(q, self.robot_sdf, compute_gradient, compute_hessian)
