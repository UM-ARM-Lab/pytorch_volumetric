import copy
import torch
import typing
import numpy as np
import pytorch_kinematics as pk
from pytorch_volumetric import sdf, model_to_sdf
import pytorch_volumetric as pv
from pytorch_kinematics.transforms.rotation_conversions import euler_angles_to_matrix

from torch.func import jacrev, jacfwd, hessian, vmap
from functools import partial
import open3d as o3d


class RobotScene:

    def __init__(self, robot_sdf: model_to_sdf.RobotSDF, scene_sdf: sdf.ObjectFrameSDF, scene_transform: pk.Transform3d,
                 threshold: float = 0.002, points_per_link: int = 100, softmin_temp: float = 1000,
                 collision_check_links: typing.List[str] = None, partial_patch=False,
                 links_per_finger=1, obj_link_name=None,
                 contact_patch_link_frame_z_max: typing.Optional[float] = None,
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
        
        # partial_patch = True
        self.contact_patch_link_frame_z_max = contact_patch_link_frame_z_max
        self.robot_query_points, self._query_point_mask = self._generate_robot_query_points(
            partial_patch=partial_patch,
            contact_patch_link_frame_z_max=contact_patch_link_frame_z_max,
        )

        self.transform_points = vmap(self._transform_points)
        self.transform_to_world = vmap(self._transform_to_world)
        self.transform_world_to_scene = vmap(self._transform_world_to_scene)

        self.transform_points_to_world = vmap(self._transform_pts_to_world)
        self.grad_smooth_points = 50
        self.grad_points = vmap(jacrev(self._transform_points))

        self.transform_grad_world_to_link = vmap(self._transform_grad_world_to_link)

        self.get_specific_tfs = vmap(self._get_specific_tfs)

        self.links_per_finger = links_per_finger
        self.obj_link_name = obj_link_name
        # self.hess_points = vmap(jacfwd(jacrev(self._transform_points)))

    def find_indices_vectorized(self, source_tensor: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
        """
        Find indices of elements from source_tensor in target_tensor using vectorized operations.
        
        Args:
            source_tensor: Tensor containing elements to find
            target_tensor: Tensor to search in
            
        Returns:
            Tensor of indices where elements from source_tensor appear in target_tensor
            Returns -1 for elements not found
        """
        # Reshape tensors for broadcasting
        source_flat = source_tensor.reshape(-1, 1)
        target_flat = target_tensor.reshape(1, -1)
        
        # Create equality matrix
        matches = (source_flat == target_flat)
        
        # Get first matching index for each source element
        indices = torch.zeros_like(source_flat, dtype=torch.long)
        
        # For each row (source element), find the first True position
        first_matches = matches.to(torch.int8).argmax(dim=1)
        
        # Check if a match was actually found (the argmax will return 0 if no match)
        found_mask = matches.any(dim=1)
        
        # Set indices to -1 where no match was found
        indices = torch.where(
            found_mask, 
            first_matches, 
            torch.tensor(-1, device=source_tensor.device)
        )
        
        return indices.reshape(source_tensor.shape)

    def _get_desired_tfs(self):
        tfs = self.robot_sdf.sdf.obj_frame_to_link_frame.get_matrix().reshape(self.num_links, -1, 4, 4)
        tfs = tfs[self.desired_link_idx].reshape(-1, 4, 4)
        return pk.Transform3d(matrix=tfs)
    
    def _get_specific_tfs(self, frame_indices):
        tfs = self.robot_sdf.sdf.obj_frame_to_link_frame.get_matrix().reshape(self.num_links, -1, 4, 4)
        indices = self.find_indices_vectorized(frame_indices, self.desired_link_idx)
        tfs = tfs[indices].reshape(-1, 4, 4)
        return tfs

    def _filter_contact_patch_points(
        self,
        points: torch.Tensor,
        *,
        link_name: str,
        partial_patch: bool,
        contact_patch_link_frame_z_max: typing.Optional[float],
    ) -> torch.Tensor:
        if partial_patch:
            ee_names = ['allegro_hand_hitosashi_finger_finger_0_aftc_base_link',
            'allegro_hand_naka_finger_finger_1_aftc_base_link',
            'allegro_hand_kusuri_finger_finger_2_aftc_base_link',
            'allegro_hand_oya_finger_3_aftc_base_link']
            if link_name in ee_names: # only do that for finger tips
                loc1, = torch.where(points[:, 1] > 0.005)
                points = points[loc1]
                loc2, = torch.where(points[:, 2] > 0.012)
                points = points[loc2]
        if contact_patch_link_frame_z_max is not None:
            z_max = torch.as_tensor(contact_patch_link_frame_z_max, device=points.device, dtype=points.dtype)
            loc, = torch.where(points[:, 2] <= z_max)
            points = points[loc]
        return points

    def _generate_robot_query_points(self, partial_patch=False, contact_patch_link_frame_z_max=None):
        query_points = []
        for target_link_name in self.desired_links:
            for i, link_name in enumerate(self.robot_sdf.sdf_to_link_name):
                if link_name == target_link_name:
                    link_sdf = self.robot_sdf.get_link_sdf(link_name)
                    # link_sdf.precompute_sdf()
                    # points, _, _ = sdf.sample_mesh_points(link_sdf, self.points_per_link,
                    #                                      dbpath=f'{link_name}_points_cache.pkl', device=self.device)
                    points, _ = link_sdf.sample_surface_points(self.points_per_link,
                                                            dbpath=f'{link_name}_points_cache.pkl', device=self.device)

                    points = self._filter_contact_patch_points(
                        points,
                        link_name=link_name,
                        partial_patch=partial_patch,
                        contact_patch_link_frame_z_max=contact_patch_link_frame_z_max,
                    )
                    if partial_patch or contact_patch_link_frame_z_max is not None:
                        attempts = 0
                        while points.shape[0] < self.points_per_link:
                            new_points, _ = link_sdf.sample_surface_points(
                                self.points_per_link,
                                dbpath=f'{link_name}_points_cache.pkl',
                                device=self.device,
                            )
                            new_points = self._filter_contact_patch_points(
                                new_points,
                                link_name=link_name,
                                partial_patch=partial_patch,
                                contact_patch_link_frame_z_max=contact_patch_link_frame_z_max,
                            )
                            if new_points.shape[0] == 0:
                                raise ValueError(
                                    f"No contact patch points remain for link {link_name} "
                                    f"with contact_patch_link_frame_z_max={contact_patch_link_frame_z_max}"
                                )
                            points = torch.cat([points, new_points], dim=0)
                            attempts += 1
                            if attempts > 16:
                                raise ValueError(
                                    f"Could not collect {self.points_per_link} contact patch points for link "
                                    f"{link_name}; got {points.shape[0]}"
                                )
                        points = points[:self.points_per_link]
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

    def get_visualization_meshes(self, q: torch.Tensor, env_q: torch.Tensor = None, pcd=None):
        self.robot_sdf.set_joint_configuration(q)

        tfs = self._get_desired_tfs().inverse()
        if pcd is None:
            pcd = o3d.geometry.PointCloud()
            pts = tfs.transform_points(self.robot_query_points).reshape(-1, 3)
            pcd.points = o3d.utility.Vector3dVector(pts.cpu().numpy())
        if env_q is not None:
            self.scene_sdf.set_joint_configuration(env_q)
        self.scene_sdf.precompute_sdf()
        scene_meshes = self.scene_sdf.get_mesh_list()
        scene_transform = self.scene_transform.get_matrix().cpu().numpy().astype(np.float64).reshape((4, 4))
        scene_meshes = [mesh.transform(scene_transform) for mesh in scene_meshes]
        colors = np.array([
            # [0.651, 0.208, 0.290],
            # [0.5, 0.5, 0.5],
            # [.8, 0.1, 0.1],
            # [.8, .8, 0.],
            [0.651, 0.208, 0.290],
            [0.121, 0.470, 0.705],
        ])
        for scene_mesh, c in zip(scene_meshes, colors):
            scene_mesh.paint_uniform_color(c)
        # scene_mesh = self.scene_sdf.obj_factory._mesh.transform(self.scene_transform.get_matrix()[0].cpu().numpy())
        # return pv.get_transformed_meshes(self.robot_sdf) + [pcd] + scene_meshes
        return pv.get_transformed_meshes(self.robot_sdf), [pcd] + scene_meshes
        # return pv.get_transformed_meshes(self.robot_sdf), scene_meshes

    def visualize_robot(self, q: torch.Tensor, env_q: torch.Tensor = None):
        meshes = self.get_visualization_meshes(q, env_q)
        o3d.visualization.draw_geometries(meshes, mesh_show_wireframe=True, width=800, height=600)

    def _transform_grad_world_to_link(self, grads: torch.Tensor):
        """
            Transforms robot query points from world frame to link frame
        :param grads: torch.Tensor B x N x 3 grads in world frame
        :return:
        """
        tfs = self._get_desired_tfs()
        grads = tfs.transform_normals(grads.reshape(-1, 3))
        return grads

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

    def _transform_pts_to_world(self, q, pts):
        self.robot_sdf.set_joint_configuration(q)
        tfs = self._get_desired_tfs().inverse()
        return tfs.transform_points(pts.reshape(1, -1, 3))

    def _transform_world_to_scene(self, pts: torch.Tensor):
        # print('device2', self.scene_transform.device, pts.device)
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
        sdf_hess = sdf_result['sdf_hess'] if compute_hessian else None
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

            new_grad = True
            pts_hessian = None
            if new_grad:
                eye = torch.eye(self.grad_smooth_points, device=self.device).unsqueeze(0).repeat(B, 1, 1)
                grad_h = -self.softmin_temp * (torch.diag_embed(h) * eye - h.unsqueeze(-1) * h.unsqueeze(-2))
                # want the closest points in the link frame
                closest_pts = self.robot_query_points.repeat(B, 1, 1, 1).reshape(B, -1, 3)[B_range, closest_indices].reshape(-1, 3)

                # closest points in world frame
                closest_pts_world = pts_world.reshape(B, -1, 3)[B_range, closest_indices]

                # closest points in scene frame
                closest_pts_scene = pts.reshape(B, -1, 3)[B_range, closest_indices].reshape(-1, 3)

                closest_links = self.desired_frame_idx[closest_indices // self.points_per_link].reshape(-1)

                q_repeat = q.unsqueeze(1).repeat(1, self.grad_smooth_points, 1).reshape(B * self.grad_smooth_points, -1)
                if not compute_hessian:
                    pts_jacobian = self.robot_sdf.chain.jacobian(q_repeat,
                                                                 locations=closest_pts,
                                                                 link_indices=closest_links)
                else:
                    pts_jacobian, pts_hessian = self.robot_sdf.chain.jacobian_and_hessian(q_repeat,
                                                                                          locations=0 * closest_pts,
                                                                                          link_indices=closest_links)
                    pts_hessian = pts_hessian[:, :3].reshape(B, self.grad_smooth_points, 3, q.shape[1], q.shape[1])

                pts_jacobian = pts_jacobian[:, :3].reshape(B, self.grad_smooth_points, 3, -1)
                sdf_weighted_grad = h[:, :, None] * closest_sdf_grads
                # transform gradient to world frame
                sdf_grad_world_frame = self.scene_transform.transform_normals(
                    sdf_weighted_grad.reshape(-1, 3)).reshape(B, self.grad_smooth_points, 3)


                # sdf gradients in world frame
                closest_sdf_grads_world = self.scene_transform.transform_normals(
                    closest_sdf_grads.reshape(-1, 3)).reshape(B, self.grad_smooth_points, 3)

                # Compute gradient of closest point with respect to robot points in world frame
                dh_dx = grad_h.unsqueeze(-1) * closest_sdf_grads_world.unsqueeze(-2)
                dclosest_dx = dh_dx.permute(0, 3, 1, 2) @ closest_pts_world.permute(0, 2, 1).unsqueeze(-1)
                dclosest_dx = h[:, :, None] + dclosest_dx.permute(0, 2, 1, 3).squeeze(-1)
                closest_pts_grad = dclosest_dx

                # this closest point grad is B x N x 3
                # it should be B x 3 x N x 3 ->
                closest_pts_grad = torch.diag_embed(closest_pts_grad).permute(0, 2, 1, 3)
                pts_jac_expanded = pts_jacobian.transpose(2, 3).unsqueeze(1).expand(B, 3, self.grad_smooth_points, -1, 3)
                dclosest_dq = (pts_jac_expanded @ closest_pts_grad.unsqueeze(-1)).squeeze(-1)
                dclosest_dq = torch.sum(dclosest_dq, dim=2)

                q_grad = (pts_jacobian.transpose(2, 3) @ sdf_grad_world_frame.unsqueeze(-1)).squeeze(-1)
                rvals['grad_sdf'] = torch.sum(q_grad, dim=1)  # B x 14

                rvals['closest_pt_world'] = torch.sum(h[:, :, None] * closest_pts_world, dim=1)  # B x 3
                rvals['closest_pt_q_grad'] = dclosest_dq  # B x 3 x 14

                ## alternative to doing this, we instead recompute contact jacobian and hessian at the closest point?
                # Recompute jac and hess at the closest point
                if not compute_hessian:
                    pts_jacobian = self.robot_sdf.chain.jacobian(q, locations=rvals['closest_pt_world'],
                                                                 link_indices=rvals['closest_link'])
                else:
                    pts_jacobian, pts_hessian = self.robot_sdf.chain.jacobian_and_hessian(q, locations=rvals['closest_pt_world'],
                                                                                          link_indices=rvals['closest_link'])
                    pts_hessian = pts_hessian[:, :3].reshape(B, 3, 3, q.shape[1], q.shape[1])
                rvals['contact_jacobian'] = pts_jacobian
                # rvals['contact_jacobian'] = torch.sum(h[:, :, None, None] * pts_jacobian, dim=1)  # B x 3 x 16
                ## now want to get the contact hessian dJ/dq
                # rvals['contact_hessian'] = torch.sum(h[:, :, None, None, None] * pts_hessian, dim=1)  # B x 3 x 14 x 14

                # get contact normals
                rvals['contact_normal'] = torch.sum(sdf_grad_world_frame, dim=1)
                rvals['contact_normal'] = rvals['contact_normal'] / torch.norm(rvals['contact_normal'], dim=1, keepdim=True)

                # Only compute hessian-related computations if we need them
                if compute_hessian and sdf_hess is not None:
                    closest_sdf_hess = sdf_hess[B_range, closest_indices]
                    sdf_weighted_hess = h[:, :, None, None] * closest_sdf_hess
                    sdf_hess_world_frame = self.scene_transform.transform_shape_operator(
                        sdf_weighted_hess.reshape(-1, 3, 3)).reshape(B, self.grad_smooth_points, 3, 3)

                    rvals['dnormal_dq'] = torch.sum(sdf_hess_world_frame @ pts_jacobian, dim=1)


            else:
                h = torch.softmax(-self.softmin_temp * sdf_vals, dim=1)
                pts_jacobian = self.grad_points(q, h)  # B x 3 x 14
                # this jacobian is in the scene frame, so don't need to transform gradient
                sdf_weighted_grad = torch.sum(h[:, :, None] * sdf_grads, dim=1)
                q_grad = (pts_jacobian.transpose(1, 2) @ sdf_weighted_grad.unsqueeze(-1)).squeeze(-1)
                rvals['grad_sdf'] = q_grad  # B x 14

            if compute_hessian and sdf_hess is not None:
                closest_sdf_hess = sdf_hess[B_range, closest_indices]
                sdf_weighted_hess = h[:, :, None, None] * closest_sdf_hess
                if pts_hessian is None:
                    pts_hessian = self.hess_points(q, h)
                else:
                    # transform grad and hess into world frame, bc jacobian and hessian are in world frame
                    sdf_weighted_grad = self.scene_transform.transform_normals(
                        sdf_weighted_grad.reshape(-1, 3)).reshape(B, self.grad_smooth_points, 3)
                    sdf_weighted_hess = self.scene_transform.transform_shape_operator(
                        sdf_weighted_hess.reshape(-1, 3, 3)).reshape(B, self.grad_smooth_points, 3, 3)

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
        sdf_hess = sdf_result['sdf_hess'] if compute_hessian else None
        # get the index for the closest frame in the environment robotSDF
        sdf_closest_link = sdf_result['closest_sdf']
        sdf_frame_indices = sdf.sdf_index_to_frame_index[sdf_closest_link]
        sdf_frame_indices = sdf_frame_indices.reshape(B, -1)

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
            pts_hessian = None
            eye = torch.eye(self.grad_smooth_points, device=self.device).unsqueeze(0).repeat(B, 1, 1)
            grad_h = -self.softmin_temp * (torch.diag_embed(h) * eye - h.unsqueeze(-1) * h.unsqueeze(-2))

            # want the closest points in the link frame
            closest_pts_link = self.robot_query_points.repeat(B, 1, 1, 1).reshape(B, -1, 3)[
                B_range, closest_indices].reshape(-1, 3)

            # closest points in world frame
            closest_pts_world = pts_world.reshape(B, -1, 3)[B_range, closest_indices]

            # closest points in scene frame
            closest_pts_scene = pts.reshape(B, -1, 3)[B_range, closest_indices].reshape(-1, 3)

            closest_links = self.desired_frame_idx[closest_indices // self.points_per_link].reshape(-1)
            q_repeat = q.unsqueeze(1).repeat(1, self.grad_smooth_points, 1).reshape(B * self.grad_smooth_points, -1)
            env_q_repeat = env_q.unsqueeze(1).repeat(1, self.grad_smooth_points, 1).reshape(B * self.grad_smooth_points,
                                                                                            -1)
            closest_env_links = sdf_frame_indices[B_range, closest_indices].reshape(-1)

            # Only compute hessian if we need it
            if compute_hessian:
                rob_jacobian, rob_hessian = self.robot_sdf.chain.jacobian_and_hessian(q_repeat,
                                                                                      locations=closest_pts_link,
                                                                                      link_indices=closest_links)
                rob_hessian = rob_hessian[:, :3].reshape(B, self.grad_smooth_points, 3, q.shape[1], q.shape[1])

                # want closest
                env_jacobian, env_hessian = self.scene_sdf.chain.jacobian_and_hessian(env_q_repeat,
                                                                                      locations=closest_pts_scene,
                                                                                      link_indices=closest_env_links,
                                                                                      locations_in_ee_frame=False)
                env_hessian = env_hessian[:, :3].reshape(B, self.grad_smooth_points, 3, env_q.shape[1], env_q.shape[1])
            else:
                rob_jacobian = self.robot_sdf.chain.jacobian(q_repeat,
                                                            locations=closest_pts_link,
                                                            link_indices=closest_links)
                env_jacobian = self.scene_sdf.chain.jacobian(env_q_repeat,
                                                            locations=closest_pts_scene,
                                                            link_indices=closest_env_links,
                                                            locations_in_ee_frame=False)

            # rotation jacobian for environment
            env_rot_jacobian = env_jacobian[:, 3:].reshape(B, self.grad_smooth_points, 3, -1)

            rob_jacobian = rob_jacobian[:, :3].reshape(B, self.grad_smooth_points, 3, -1)
            env_jacobian = env_jacobian[:, :3].reshape(B, self.grad_smooth_points, 3, -1)

            sdf_weighted_grad = h[:, :, None] * closest_sdf_grads

            # transform gradient to world frame
            sdf_grad_world_frame = self.scene_transform.transform_normals(
                sdf_weighted_grad.reshape(-1, 3)).reshape(B, self.grad_smooth_points, 3)

            # sdf gradients in world frame
            closest_sdf_grads_world = self.scene_transform.transform_normals(
                closest_sdf_grads.reshape(-1, 3)).reshape(B, self.grad_smooth_points, 3)

            # Compute gradient of closest point with respect to robot points in world frame
            dh_dx = grad_h.unsqueeze(-1) * closest_sdf_grads_world.unsqueeze(-2)
            dclosest_dx = dh_dx.permute(0, 3, 1, 2) @ closest_pts_world.permute(0, 2, 1).unsqueeze(-1)
            dclosest_dx = h[:, :, None] + dclosest_dx.permute(0, 2, 1, 3).squeeze(-1)
            closest_pts_grad = dclosest_dx

            # this closest point grad is B x N x 3
            # it should be B x 3 x N x 3 ->
            closest_pts_grad = torch.diag_embed(closest_pts_grad).permute(0, 2, 1, 3)
            rob_jac_expanded = rob_jacobian.transpose(2, 3).unsqueeze(1).expand(B, 3, self.grad_smooth_points, -1, 3)
            dclosest_dq = (rob_jac_expanded @ closest_pts_grad.unsqueeze(-1)).squeeze(-1)
            dclosest_dq = torch.sum(dclosest_dq, dim=2)

            env_jac_expanded = env_jacobian.transpose(2, 3).unsqueeze(1).expand(B, 3, self.grad_smooth_points, -1, 3)
            dclosest_denv_q = (env_jac_expanded @ closest_pts_grad.unsqueeze(-1)).squeeze(-1)
            dclosest_denv_q = torch.sum(dclosest_denv_q, dim=2)

            q_grad = (rob_jacobian.transpose(2, 3) @ sdf_grad_world_frame.unsqueeze(-1)).squeeze(-1)
            q_env_grad = (env_jacobian.transpose(2, 3) @ -sdf_weighted_grad.unsqueeze(-1)).squeeze(-1)
            rvals['grad_sdf'] = torch.sum(q_grad, dim=1)  # B x 14
            rvals['grad_env_sdf'] = torch.sum(q_env_grad, dim=1)  # B x 14
            rvals['closest_pt_world'] = torch.sum(h[:, :, None] * closest_pts_world, dim=1)  # B x 3
            rvals['closest_pt_q_grad'] = dclosest_dq  # B x 3 x 14
            rvals['closest_pt_env_q_grad'] = dclosest_denv_q

            ## alternative to doing this, we instead recompute contact jacobian and hessian at the closest point?
            rvals['contact_jacobian'] = torch.sum(h[:, :, None, None] * rob_jacobian, dim=1)  # B x 3 x 16
            ## now want to get the contact hessian dJ/dq
            if compute_hessian:
                rvals['contact_hessian'] = torch.sum(h[:, :, None, None, None] * rob_hessian, dim=1)  # B x 3 x 14 x 14

            # get contact normals
            rvals['contact_normal'] = torch.sum(sdf_grad_world_frame, dim=1)
            rvals['contact_normal'] = rvals['contact_normal'] / torch.norm(rvals['contact_normal'], dim=1, keepdim=True)

            # Only compute hessian-related computations if we need them
            if compute_hessian and sdf_hess is not None:
                closest_sdf_hess = sdf_hess[B_range, closest_indices]
                sdf_weighted_hess = h[:, :, None, None] * closest_sdf_hess
                sdf_hess_world_frame = self.scene_transform.transform_shape_operator(
                    sdf_weighted_hess.reshape(-1, 3, 3)).reshape(B, self.grad_smooth_points, 3, 3)

                rvals['dnormal_dq'] = torch.sum(sdf_hess_world_frame @ rob_jacobian, dim=1)

                # get contact normal in scene frame
                contact_normal_scene = torch.sum(sdf_weighted_grad, dim=1)
                contact_normal_scene = contact_normal_scene / torch.linalg.norm(contact_normal_scene, dim=-1, keepdim=True)

                rvals['env_rot_jacobian'] = torch.sum(h[:, :, None, None] * env_rot_jacobian, dim=1)

                dnormal_denv_q_rigid = torch.cross(contact_normal_scene.reshape(-1, 1, 3),
                                                   rvals['env_rot_jacobian'].transpose(2, 1), dim=-1).transpose(1, 2)

                # this is the gradient of the contact normal in the scene frame, need it in robot frame
                dnormal_denv_q = torch.sum(sdf_weighted_hess @ env_jacobian, dim=1) + dnormal_denv_q_rigid
                rvals['dnormal_denv_q'] = self.scene_transform.transform_normals(
                    dnormal_denv_q.transpose(1, 2).reshape(-1, 3)
                ).reshape(B, -1, 3).transpose(1, 2)

                q_hess = torch.sum(rob_hessian * sdf_grad_world_frame.reshape(B, -1, 3, 1, 1), dim=2)
                q_hess2 = rob_jacobian.transpose(2, 3) @ sdf_hess_world_frame @ rob_jacobian
                q_hess = q_hess + q_hess2
                rvals['hess_sdf'] = torch.sum(q_hess, dim=1)
                q_env_hess = torch.sum(env_hessian * -sdf_weighted_grad.reshape(B, -1, 3, 1, 1), dim=2)
                q_env_hess2 = env_jacobian.transpose(2, 3) @ -sdf_weighted_hess @ env_jacobian
                q_env_hess = q_env_hess + q_env_hess2
                rvals['hess_env_sdf'] = torch.sum(q_env_hess, dim=1)

        return rvals

    def _collision_check_against_robot_sdf_per_link(self, q: torch.Tensor, env_q: torch.Tensor,
                                                    sdf: model_to_sdf.RobotSDF, compute_gradient=False,
                                                    compute_hessian=False,
                                                    compute_closest_obj_point=False,
                                                    rob_link_idx=None,
                                                    contact_constraint_only=False,
                                                    tactile_controller=None,
                                                    skip_csvto=False):
        # This is a function for collision checking when the environment is itself a RobotSDF, i.e. it is
        # an articulated SDF with configuration env_q, vs the robot which has configuration rob_q
        # Add leading batch dimension
        if len(q.shape) == 1:
            q = q.reshape(1, -1)
            B = 1
        else:
            B = q.shape[0]
        num_fingers = len(self.desired_links) // self.links_per_finger

        if compute_hessian and not compute_gradient:
            raise ValueError('Cannot compute hessian without gradient')

        # Only compute hessian if both conditions are met
        should_compute_hessian = (compute_hessian or (not contact_constraint_only)) and (not tactile_controller or not skip_csvto)
        should_compute_hessian_jac = (compute_hessian or (not contact_constraint_only))

        # set the configuration of the scene
        sdf.set_joint_configuration(env_q)
        pts_world = self.transform_to_world(q)
        pts = self.transform_world_to_scene(pts_world)
        robot_frame_indices = self.desired_frame_idx[None, :, None].repeat(B, 1, self.points_per_link).reshape(B, num_fingers, -1)
        sdf_result = sdf(pts, return_extra_info=True, compute_hessian=should_compute_hessian)
        sdf_vals = sdf_result['sdf_val']
        sdf_grads = sdf_result['sdf_grad']
        sdf_hess = sdf_result['sdf_hess'] if should_compute_hessian else None

        # get the index for the closest frame in the environment robotSDF
        sdf_closest_link = sdf_result['closest_sdf']
        sdf_frame_indices = sdf.sdf_index_to_frame_index[sdf_closest_link]
        sdf_frame_indices = sdf_frame_indices.reshape(B, num_fingers, -1)
        
        if sdf_hess is not None:
            sdf_hess = sdf_hess.reshape(B, num_fingers, -1, 3, 3)

        # get minimum links
        sdf_vals = sdf_vals.reshape(B, num_fingers, -1)
        sdf_grads = sdf_grads.reshape(B, num_fingers, -1, 3)
        sdf_indices = torch.argsort(sdf_vals, dim=-1, descending=False)
        #sdf_val = sdf_vals[torch.arange(B*), torch.arange(num_fingers), sdf_indices[:, :, 0]]

        sdf_val = sdf_vals.reshape(B*num_fingers, -1)[torch.arange(B*num_fingers), sdf_indices.reshape(B*num_fingers, -1)[:, 0]]
        rvals = {
            'sdf': sdf_val,
        }
        if compute_gradient:
            # rather than take a single point we use the softmin to get a weighted average of the gradients for
            # the self.grad_smooth_points number of closest points, this helps smooth the gradient
            BN = B*num_fingers
            B_range = torch.arange(BN).unsqueeze(-1)
            closest_indices = sdf_indices[:, :, :self.grad_smooth_points].reshape(BN, self.grad_smooth_points)
            closest_sdf_grads = sdf_grads.reshape(BN, -1, 3)[B_range, closest_indices]
            closest_sdf_vals = sdf_vals.reshape(BN, -1)[B_range, closest_indices]
            h = torch.softmax(-self.softmin_temp * closest_sdf_vals, dim=1)
            new_grad = True
            pts_hessian = None
            eye = torch.eye(self.grad_smooth_points, device=self.device).unsqueeze(0).repeat(BN, 1, 1)
            grad_h = -self.softmin_temp * (torch.diag_embed(h) * eye - h.unsqueeze(-1) * h.unsqueeze(-2))

            # want the closest points in the link frame
            closest_pts_link = self.robot_query_points.repeat(BN, 1, 1, 1).reshape(BN, -1, 3)[
                B_range, closest_indices].reshape(-1, 3)

            # closest points in world frame
            closest_pts_world = pts_world.reshape(BN, -1, 3)[B_range, closest_indices]
        
            # closest points in scene frame
            closest_pts_scene = pts.reshape(BN, -1, 3)[B_range, closest_indices].reshape(-1, 3)

            # closest_links = self.desired_frame_idx[closest_indices // (self.points_per_link * 2)].reshape(-1)
            # closest_links = self.desired_frame_idx[None, :, None].repeat(B, 1, self.grad_smooth_points).reshape(-1)
            closest_links_not_flat = torch.gather(robot_frame_indices, 2, closest_indices.reshape(B, -1, self.grad_smooth_points))
            closest_links = closest_links_not_flat.reshape(-1)
            q_repeat = q[:, None, None, :].repeat(1, num_fingers, self.grad_smooth_points, 1)
            q_repeat = q_repeat.reshape(BN * self.grad_smooth_points, -1)

            env_q_repeat = env_q[:, None, None, :].repeat(1, num_fingers,
                                                          self.grad_smooth_points,
                                                          1).reshape(BN * self.grad_smooth_points,-1)

            closest_env_links = sdf_frame_indices.reshape(BN, -1)[B_range, closest_indices].reshape(-1)

            rvals['closest_pt_world'] = torch.sum(h[:, :, None] * closest_pts_world, dim=1)  # B x 3

            sdf_weighted_grad = h[:, :, None] * closest_sdf_grads

            # transform gradient to world frame
            sdf_grad_world_frame = self.scene_transform.transform_normals(
                sdf_weighted_grad.reshape(-1, 3)).reshape(BN, self.grad_smooth_points, 3)
            if not skip_csvto:
                # want closest
                # Only compute robot hessian if we need it
                if should_compute_hessian_jac:
                    rob_jacobian, rob_hessian = self.robot_sdf.chain.jacobian_and_hessian(q_repeat,
                                                                                        locations=closest_pts_link,
                                                                                        link_indices=closest_links)
                    rob_hessian = rob_hessian[:, :3].reshape(BN, self.grad_smooth_points, 3, q.shape[1], q.shape[1])
                else:
                    rob_jacobian = self.robot_sdf.chain.jacobian(q_repeat,
                                                                locations=closest_pts_link,
                                                                link_indices=closest_links)

                # Only compute environment hessian if we need it
                if should_compute_hessian:
                    env_jacobian, env_hessian = self.scene_sdf.chain.jacobian_and_hessian(env_q_repeat,
                                                                                        locations=closest_pts_scene,
                                                                                        link_indices=closest_env_links,
                                                                                        locations_in_ee_frame=False)
                    env_hessian = env_hessian[:, :3].reshape(BN, self.grad_smooth_points, 3, env_q.shape[1], env_q.shape[1])
                else:
                    env_jacobian = self.scene_sdf.chain.jacobian(env_q_repeat,
                                                                locations=closest_pts_scene,
                                                                link_indices=closest_env_links,
                                                                locations_in_ee_frame=False)

                # rotation jacobian for environment
                env_rot_jacobian = env_jacobian[:, 3:].reshape(BN, self.grad_smooth_points, 3, -1)


                rob_jacobian = rob_jacobian[:, :3].reshape(BN, self.grad_smooth_points, 3, -1)
                env_jacobian = env_jacobian[:, :3].reshape(BN, self.grad_smooth_points, 3, -1)


                # sdf gradients in world frame
                closest_sdf_grads_world = self.scene_transform.transform_normals(
                    closest_sdf_grads.reshape(-1, 3)).reshape(BN, self.grad_smooth_points, 3)

                # Compute gradient of closest point with respect to robot points in world frame
                dh_dx = grad_h.unsqueeze(-1) * closest_sdf_grads_world.unsqueeze(-2)
                dclosest_dx = dh_dx.permute(0, 3, 1, 2) @ closest_pts_world.permute(0, 2, 1).unsqueeze(-1)
                dclosest_dx = h[:, :, None] + dclosest_dx.permute(0, 2, 1, 3).squeeze(-1)
                closest_pts_grad = dclosest_dx

                # this closest point grad is B x N x 3
                # it should be B x 3 x N x 3 ->
                closest_pts_grad = torch.diag_embed(closest_pts_grad).permute(0, 2, 1, 3)
                rob_jac_expanded = rob_jacobian.transpose(2, 3).unsqueeze(1).expand(BN, 3, self.grad_smooth_points, -1, 3)
                dclosest_dq = (rob_jac_expanded @ closest_pts_grad.unsqueeze(-1)).squeeze(-1)
                dclosest_dq = torch.sum(dclosest_dq, dim=2)

                env_jac_expanded = env_jacobian.transpose(2, 3).unsqueeze(1).expand(BN, 3, self.grad_smooth_points, -1, 3)
                dclosest_denv_q = (env_jac_expanded @ closest_pts_grad.unsqueeze(-1)).squeeze(-1)
                dclosest_denv_q = torch.sum(dclosest_denv_q, dim=2)

                q_grad = (rob_jacobian.transpose(2, 3) @ sdf_grad_world_frame.unsqueeze(-1)).squeeze(-1)
                q_env_grad = (env_jacobian.transpose(2, 3) @ -sdf_weighted_grad.unsqueeze(-1)).squeeze(-1)

                rvals['grad_sdf'] = torch.sum(q_grad, dim=1)  # B x 14
                rvals['grad_env_sdf'] = torch.sum(q_env_grad, dim=1)  # B x 14
            
                rvals['contact_jacobian'] = torch.sum(h[:, :, None, None] * rob_jacobian, dim=1)  # B x 3 x 16
                # Only compute contact hessian if we need it
                if should_compute_hessian_jac:
                    rvals['contact_hessian'] = torch.sum(h[:, :, None, None, None] * rob_hessian, dim=1)  # B x 3 x 14 x 14
            
                if not contact_constraint_only:
        
                        # if 'cross' in self.obj_link_name:
                    rvals['closest_rob_pt_scene'] = torch.sum(h[:, :, None] * closest_pts_scene.reshape(closest_pts_world.shape), dim=1)  # B x 3
                    
                    rvals['closest_pt_q_grad'] = dclosest_dq  # B x 3 x 14
                    rvals['closest_pt_env_q_grad'] = dclosest_denv_q
                    # Transform dclosest_dq to the scene frame
                    rvals['closest_pt_q_grad_scene'] = self.scene_transform.inverse().transform_normals(
                        dclosest_dq.transpose(1, 2).reshape(-1, 3)
                    ).reshape(BN, -1, 3).transpose(1, 2)
                    rvals['closest_pt_env_q_grad_scene'] = self.scene_transform.inverse().transform_normals(
                        dclosest_denv_q.transpose(1, 2).reshape(-1, 3)
                    ).reshape(BN, -1, 3).transpose(1, 2)

                    env_q_for_transform = env_q.clone()
                    # Ignore screwdriver yaw
                    if self.obj_link_name == 'screwdriver_body':
                        env_q_for_transform[:, -2:] = 0.0
                    # Transform dclosest_dq to the object frame
                    object_trans_dict = sdf.chain.forward_kinematics(
                        env_q_for_transform)
                    object_trans_mat = object_trans_dict[self.obj_link_name].inverse().get_matrix()
                    
                    object_trans_mat = object_trans_mat.unsqueeze(1).expand(-1, num_fingers, -1, -1).reshape(BN, 4, 4)
                    object_trans = pk.Transform3d(matrix=object_trans_mat)

                    # Convert from scene frame to object frame
                    rvals['closest_pt_q_grad_object'] = object_trans.transform_normals(
                        rvals['closest_pt_q_grad_scene'].transpose(1, 2)
                    ).reshape(BN, -1, 3).transpose(1, 2)
                    rvals['closest_pt_env_q_grad_object'] = object_trans.transform_normals(
                        rvals['closest_pt_env_q_grad_scene'].transpose(1, 2)
                    ).reshape(BN, -1, 3).transpose(1, 2)

                    if self.obj_link_name == 'screwdriver_body':
                        rvals['closest_pt_env_q_grad_object'][..., -2:] = 0.0

                    rvals['closest_rob_pt_object'] = object_trans.transform_points(rvals['closest_rob_pt_scene'].unsqueeze(1)).squeeze(1)

                    if compute_closest_obj_point:
                        return_frame = 1 if 'cross' in self.obj_link_name else None
                        input_ = rvals['closest_rob_pt_object']
                        if 'cross' in self.obj_link_name:
                            input_ = rvals['closest_rob_pt_scene']
                        res = sdf.object_frame_closest_point(input_, return_frame=return_frame)
                        rvals['closest_obj_pt_object'] = res.closest    
                    
                    if compute_closest_obj_point or rob_link_idx is not None:
                        
                        if compute_closest_obj_point:
                            rvals['closest_pt_closest_link'] = closest_links_not_flat[..., 0].flatten()
                            closest_pt_link_tfs = self.get_specific_tfs(rvals['closest_pt_closest_link'].unsqueeze(0)).squeeze(0)
                        else:
                            closest_pt_link_tfs = self.get_specific_tfs(rob_link_idx.unsqueeze(0)).reshape(BN, 4, 4)
                        closest_pt_link_tfs = pk.Transform3d(matrix=closest_pt_link_tfs)
                        rvals['closest_rob_pt_link'] = closest_pt_link_tfs.transform_points(
                            rvals['closest_pt_world'].reshape(BN, -1, 3)
                        ).reshape(BN, 3)  # B x 3

                        rvals['closest_pt_q_grad_link'] = closest_pt_link_tfs.transform_normals(
                            dclosest_dq.transpose(1, 2)
                        ).reshape(BN, -1, 3).transpose(1, 2)
                        rvals['closest_pt_env_q_grad_link'] = closest_pt_link_tfs.transform_normals(
                            dclosest_denv_q.transpose(1, 2)
                        ).reshape(BN, -1, 3).transpose(1, 2)


            contact_normal = torch.sum(sdf_grad_world_frame, dim=1)
            norm = torch.norm(contact_normal, dim=1, keepdim=True)
            contact_normal = contact_normal / norm
            rvals['contact_normal'] = contact_normal
            
            if tactile_controller:
                
                # Only compute robot contact jacobian at 
                closest_pt_closest_link = closest_links_not_flat[..., 0]
                closest_pt_link_tfs = self.get_specific_tfs(closest_pt_closest_link).flatten(0, 1)
                closest_pt_link_tfs = pk.Transform3d(matrix=closest_pt_link_tfs)
                rvals['closest_rob_pt_link'] = closest_pt_link_tfs.transform_points(
                    rvals['closest_pt_world'].reshape(BN, -1, 3)
                ).reshape(BN, 3)  # B x 3
                if skip_csvto:
                    rob_jacobian = self.robot_sdf.chain.jacobian(q.unsqueeze(1).repeat(1, num_fingers, 1).flatten(0, 1),
                                                                    locations=rvals['closest_rob_pt_link'],
                                                                    link_indices=closest_pt_closest_link.flatten())
                    rob_jacobian = rob_jacobian[:, :3].reshape(BN, 3, -1)
                    
                rvals['J_q'] = rob_jacobian
                
                rvals['contact_n'] = -contact_normal
                # Compute 2 vectors that are perpendicular to the contact normal
                # and lie in the plane of the contact
                cross_axis = torch.tensor([1, 0, 0], device=rvals['contact_n'].device, dtype=rvals['contact_n'].dtype)
                cross_axis = cross_axis.unsqueeze(0).expand(BN, -1)
                contact_normal_perp1 = torch.cross(rvals['contact_n'], cross_axis)
                contact_normal_perp2 = torch.cross(rvals['contact_n'], contact_normal_perp1)
                contact_normal_perp1 = contact_normal_perp1 / torch.norm(contact_normal_perp1, dim=1, keepdim=True)
                contact_normal_perp2 = contact_normal_perp2 / torch.norm(contact_normal_perp2, dim=1, keepdim=True)
                rvals['contact_o'] = contact_normal_perp1
                rvals['contact_t'] = contact_normal_perp2
                
                # Form rotation matrix from contact normal and contact tangent
                R = torch.stack([rvals['contact_n'], rvals['contact_o'], rvals['contact_t']], dim=-1).transpose(1, 2)
                rvals['contact_r_bar'] = torch.zeros((BN, 6, 6), device=rvals['contact_n'].device)
                rvals['contact_r_bar'][:, :3, :3] = R
                rvals['contact_r_bar'][:, 3:6, 3:6] = R
                
                # Get screwdriver body position in world frame
                screwdriver_fk = sdf.chain.forward_kinematics(env_q)
                screwdriver_body_pos = screwdriver_fk[self.obj_link_name].get_matrix()[:, :3, 3]
                screwdriver_body_pos = self.scene_transform.transform_points(screwdriver_body_pos.unsqueeze(0)).squeeze(0)
                screwdriver_body_pos = screwdriver_body_pos.reshape(B, 3).repeat(num_fingers, 1)
                rvals['screwdriver_body_pos'] = screwdriver_body_pos
                                
                r = rvals['closest_pt_world'] - rvals['screwdriver_body_pos']
                
                rx = r[:, 0]
                ry = r[:, 1]
                rz = r[:, 2]
                
                S_r = torch.zeros((BN, 3, 3), device=r.device)
                S_r[:, 0, 1] = -rz
                S_r[:, 0, 2] = ry
                S_r[:, 1, 0] = rz
                S_r[:, 1, 2] = -rx
                S_r[:, 2, 0] = -ry
                S_r[:, 2, 1] = rx
                
                # rvals['S_r'] = S_r
                
                P = torch.eye(6, device=r.device).unsqueeze(0).expand(BN, -1, -1).contiguous()
                P[:, 3:6, :3] = S_r
                
                # rvals['P'] = P
                
                G_T = torch.bmm(rvals['contact_r_bar'].transpose(1, 2), P.transpose(1,2))
                
                G_T = G_T.reshape(B, num_fingers, 6, 6)[:, :, :3, :]
                
                rvals['G_o'] = G_T.flatten(1, 2).transpose(1, 2)
                
                if q.shape[0] == 2:
                    changed_q = True
                    q_jac = q.repeat(2, 1)
                else:
                    changed_q = False
                    q_jac = q
                    
                # rvals['H_q'] = rob_hessian
            
            # Only compute hessian-related computations if we need them
            if should_compute_hessian and sdf_hess is not None:
                # get contact normals

                closest_sdf_hess = sdf_hess.reshape(BN, -1, 3, 3)[B_range, closest_indices]
                sdf_weighted_hess = h[:, :, None, None] * closest_sdf_hess
                sdf_hess_world_frame = self.scene_transform.transform_shape_operator(
                    sdf_weighted_hess.reshape(-1, 3, 3)).reshape(BN, self.grad_smooth_points, 3, 3)

                # grad of normal normalization
                n_xx = (torch.pow(contact_normal[:, 1], 2) + torch.pow(contact_normal[:, 2], 2)) / torch.pow(norm, 3).squeeze(-1)
                n_xy = -contact_normal[:, 0] * contact_normal[:, 1] / torch.pow(norm, 3).squeeze(-1)
                n_xz = -contact_normal[:, 0] * contact_normal[:, 2] / torch.pow(norm, 3).squeeze(-1)
                n_yx = -contact_normal[:, 1] * contact_normal[:, 0] / torch.pow(norm, 3).squeeze(-1)
                n_yy = (torch.pow(contact_normal[:, 0], 2) + torch.pow(contact_normal[:, 2], 2)) / torch.pow(norm, 3).squeeze(-1)
                n_yz = -contact_normal[:, 1] * contact_normal[:, 2] / torch.pow(norm, 3).squeeze(-1)
                n_zx = -contact_normal[:, 2] * contact_normal[:, 0] / torch.pow(norm, 3).squeeze(-1)
                n_zy = -contact_normal[:, 2] * contact_normal[:, 1] / torch.pow(norm, 3).squeeze(-1)
                n_zz = (torch.pow(contact_normal[:, 0], 2) + torch.pow(contact_normal[:, 1], 2)) / torch.pow(norm, 3).squeeze(-1)
                n_x = torch.stack([n_xx, n_xy, n_xz], dim=-1)
                n_y = torch.stack([n_yx, n_yy, n_yz], dim=-1)
                n_z = torch.stack([n_zx, n_zy, n_zz], dim=-1)
                n_grad = torch.stack([n_x, n_y, n_z], dim=-1)
                rvals['dnormal_dq'] = n_grad @ torch.sum(sdf_hess_world_frame @ rob_jacobian, dim=1)

                # get contact normal in scene frame
                contact_normal_scene = torch.sum(sdf_weighted_grad, dim=1)
                contact_normal_scene = contact_normal_scene / torch.linalg.norm(contact_normal_scene, dim=-1, keepdim=True)

                rvals['env_rot_jacobian'] = torch.sum(h[:, :, None, None] * env_rot_jacobian, dim=1)

                dnormal_denv_q_rigid = torch.cross(contact_normal_scene.reshape(-1, 1, 3),
                                                   rvals['env_rot_jacobian'].transpose(2, 1), dim=-1).transpose(1, 2)

                # this is the gradient of the contact normal in the scene frame, need it in robot frame
                dnormal_denv_q = torch.sum(sdf_weighted_hess @ env_jacobian, dim=1) + dnormal_denv_q_rigid

                rvals['dnormal_denv_q'] = self.scene_transform.transform_normals(
                    dnormal_denv_q.transpose(1, 2).reshape(-1, 3)
                ).reshape(BN, -1, 3).transpose(1, 2)

                # Compute hessian terms
                q_hess = torch.sum(rob_hessian * sdf_grad_world_frame.reshape(BN, -1, 3, 1, 1), dim=2)
                q_hess2 = rob_jacobian.transpose(2, 3) @ sdf_hess_world_frame @ rob_jacobian
                q_hess = q_hess + q_hess2
                rvals['hess_sdf'] = torch.sum(q_hess, dim=1)
                q_env_hess = torch.sum(env_hessian * -sdf_weighted_grad.reshape(BN, -1, 3, 1, 1), dim=2)
                q_env_hess2 = env_jacobian.transpose(2, 3) @ -sdf_weighted_hess @ env_jacobian
                q_env_hess = q_env_hess + q_env_hess2

                rvals['hess_env_sdf'] = torch.sum(q_env_hess, dim=1)

        for key, item in rvals.items():
            if key != 'G_o':
                rvals[key] = item.reshape(B, num_fingers, *item.shape[1:])

        return rvals

    def _get_sdf(self, q: torch.Tensor, env_q: torch.Tensor,
                sdf: model_to_sdf.RobotSDF, compute_gradient=False,
                compute_hessian=False,
                compute_closest_obj_point=False,
                rob_link_idx=None):
        # This is a function for collision checking when the environment is itself a RobotSDF, i.e. it is
        # an articulated SDF with configuration env_q, vs the robot which has configuration rob_q
        # Add leading batch dimension
        if len(q.shape) == 1:
            q = q.reshape(1, -1)
            B = 1
        else:
            B = q.shape[0]
        num_fingers = len(self.desired_links) // self.links_per_finger

        if compute_hessian and not compute_gradient:
            raise ValueError('Cannot compute hessian without gradient')

        # set the configuration of the scene
        sdf.set_joint_configuration(env_q)
        pts_world = self.transform_to_world(q)
        pts = self.transform_world_to_scene(pts_world)
        robot_frame_indices = self.desired_frame_idx[None, :, None].repeat(B, 1, self.points_per_link).reshape(B, num_fingers, -1)
        sdf_result = sdf(pts, return_extra_info=True)
        sdf_vals = sdf_result['sdf_val']
        sdf_grads = sdf_result['sdf_grad']
        sdf_hess = sdf_result['sdf_hess'] if compute_hessian else None

        # get the index for the closest frame in the environment robotSDF
        sdf_closest_link = sdf_result['closest_sdf']
        sdf_frame_indices = sdf.sdf_index_to_frame_index[sdf_closest_link]
        sdf_frame_indices = sdf_frame_indices.reshape(B, num_fingers, -1)
        if sdf_hess is not None:
            sdf_hess = sdf_hess.reshape(B, num_fingers, -1, 3, 3)

        # get minimum links
        sdf_vals = sdf_vals.reshape(B, num_fingers, -1)
        sdf_grads = sdf_grads.reshape(B, num_fingers, -1, 3)
        sdf_indices = torch.argsort(sdf_vals, dim=-1, descending=False)
        #sdf_val = sdf_vals[torch.arange(B*), torch.arange(num_fingers), sdf_indices[:, :, 0]]

        sdf_val = sdf_vals.reshape(B*num_fingers, -1)[torch.arange(B*num_fingers), sdf_indices.reshape(B*num_fingers, -1)[:, 0]]
        rvals = {
            'sdf': sdf_val,
        }
        return rvals

    def scene_collision_check(self, q: torch.Tensor, env_q=None,
                              compute_gradient=False, compute_hessian=False,
                              compute_closest_obj_point=False,
                              rob_link_idx=None,
                              contact_constraint_only=False,
                              tactile_controller=None,
                              skip_csvto=False):
        """
           Collision checks robot with scene sdf
           :param q: torch.Tensor B x dq joint angles
           :param compute_gradient: bool whether to compute gradient of sdf wrt joint angles
           :param compute_hessian: bool whether to compute hessian of sdf wrt joint angles
           """
        if isinstance(self.scene_sdf, model_to_sdf.RobotSDF):
            if env_q is None:
                raise ValueError('Must provide environment configuration if scene is articulated SDF')
            return self._collision_check_against_robot_sdf_per_link(q, env_q, self.scene_sdf, compute_gradient,
                                                                    compute_hessian,compute_closest_obj_point,
                                                                    rob_link_idx,
                                                                    contact_constraint_only,
                                                                    tactile_controller,
                                                                    skip_csvto)

        return self._collision_check(q, self.scene_sdf, compute_gradient, compute_hessian, contact_constraint_only)
    
    def scene_get_sdf(self, q: torch.Tensor, env_q=None,
                              compute_gradient=False, compute_hessian=False,
                              compute_closest_obj_point=False,
                              rob_link_idx=None):
        """
           Collision checks robot with scene sdf
           :param q: torch.Tensor B x dq joint angles
           :param compute_gradient: bool whether to compute gradient of sdf wrt joint angles
           :param compute_hessian: bool whether to compute hessian of sdf wrt joint angles
           """
        if isinstance(self.scene_sdf, model_to_sdf.RobotSDF):
            if env_q is None:
                raise ValueError('Must provide environment configuration if scene is articulated SDF')
            return self._get_sdf(q, env_q, self.scene_sdf, compute_gradient,
                                                                    compute_hessian,compute_closest_obj_point,
                                                                    rob_link_idx)
        else:
            # For non-RobotSDF scene objects, create a simple wrapper
            if len(q.shape) == 1:
                q = q.reshape(1, -1)
                B = 1
            else:
                B = q.shape[0]
            
            pts_world = self.transform_to_world(q)
            pts = self.transform_world_to_scene(pts_world)
            
            sdf_result = self.scene_sdf(pts, return_extra_info=True)
            sdf_vals = sdf_result['sdf_val']
            sdf_vals = sdf_vals.reshape(B, -1)
            sdf_val = torch.min(sdf_vals, dim=1)[0]
            
            return {'sdf': sdf_val}

    def self_collision_check(self, q: torch.Tensor, compute_gradient=False, compute_hessian=False):
        """
           Performs self collision check
           :param q: torch.Tensor B x dq joint angles
           :param compute_gradient: bool whether to compute gradient of sdf wrt joint angles
           :param compute_hessian: bool whether to compute hessian of sdf wrt joint angles
           """
        raise NotImplementedError("Not correctly implemented yet")
        # TODO: this is not correct yet, currently function uses scene transforms
        rvals = self._collision_check_against_robot_sdf(q, q, self.robot_sdf, compute_gradient, compute_hessian)
        # Combine gradients
        if compute_gradient:
            rvals['grad_sdf'] += rvals['grad_env_sdf']
            rvals['grad_env_sdf'] = {}
        if compute_hessian:
            rvals['hess_sdf'] += rvals['hess_env_sdf']
            rvals['hess_env_sdf'] = {}
        return rvals
