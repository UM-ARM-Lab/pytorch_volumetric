import torch
import typing
import pytorch_kinematics as pk
from pytorch_volumetric import sdf, model_to_sdf
import pytorch_volumetric as pv


class RobotScene:

    def __init__(self, robot_sdf: model_to_sdf.RobotSDF,
                 scene_sdf: sdf.ObjectFrameSDF,
                 scene_transform: pk.Transform3d):

        self.robot_sdf = robot_sdf
        self.scene_sdf = scene_sdf
        self.device = self.robot_sdf.device
        self.threshold = 0.002
        self.points_per_link = 100
        self.softmin_T = 1000
        self.scene_transform = scene_transform.to(device=self.device)
        self.robot_query_points, self._query_point_mask = self._generate_robot_query_points()

    def _generate_robot_query_points(self):
        query_points = []
        for link_name in self.robot_sdf.sdf_to_link_name:
            link_sdf = self.robot_sdf.get_link_obj_factory(link_name)
            link_sdf.precompute_sdf()
            points, _, _ = sdf.sample_mesh_points(link_sdf, self.points_per_link,
                                                  dbpath=f'{link_name}_points_cache.pkl', device=self.device)
            query_points.append(points)
        query_points = torch.stack(query_points, dim=0)

        # mask out points that are in self-collision with default configuration
        tfs = self.robot_sdf.sdf.obj_frame_to_link_frame.inverse()
        pts = tfs.transform_points(query_points).reshape(-1, 3)
        sdf_vals, _ = self.robot_sdf(pts)
        mask = torch.where(sdf_vals < -self.threshold, torch.zeros_like(sdf_vals), torch.ones_like(sdf_vals))
        return query_points, mask

    def visualize_robot(self, q: torch.Tensor):
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        self.robot_sdf.set_joint_configuration(q)
        tfs = self.robot_sdf.sdf.obj_frame_to_link_frame.inverse()
        pts = tfs.transform_points(self.robot_query_points).reshape(-1, 3)
        pcd.points = o3d.utility.Vector3dVector(pts.cpu().numpy())
        self.scene_sdf.obj_factory.precompute_sdf()
        scene_mesh = self.scene_sdf.obj_factory._mesh.transform(self.scene_transform.get_matrix()[0].cpu().numpy())
        o3d.visualization.draw_geometries(pv.get_transformed_meshes(self.robot_sdf) + [pcd, scene_mesh])

    def scene_collision_check(self, q: torch.Tensor, compute_gradient=False, compute_hessian=False):
        """
        :param q: torch.Tensor B x dq joint angles
        """
        # Add leading batch dimension
        if len(q.shape) == 1:
            q = q.reshape(1, -1)
            B = 1
        else:
            B = q.shape[0]

        if compute_gradient:
            q.requires_grad_(True)
            q.retain_grad()

        self.robot_sdf.set_joint_configuration(q)
        tfs = self.robot_sdf.sdf.obj_frame_to_link_frame.inverse()

        # Convention in RobotSDF is that transforms are shape (num_links, num_configs, 4, 4) before flattening
        # the first two dimensions. Thus points need to be (num_links, num_configs, num_points, 3)
        batched_query_points = self.robot_query_points.unsqueeze(1).repeat(1, B, 1, 1)
        pts = tfs.transform_points(batched_query_points.reshape(-1, self.points_per_link, 3))
        pts = self.scene_transform.inverse().transform_points(pts)
        # Now we have transformed points, want to permute to be (num_configs, num_links*num_points, 3)
        pts = pts.reshape(-1, B, self.points_per_link, 3)
        pts = pts.permute(1, 0, 2, 3).reshape(B, -1, 3)
        sdf_vals, sdf_grads = self.scene_sdf(pts)

        # use softmin to get gradient but use hard min for actual sdf value
        h = torch.softmax(-self.softmin_T * sdf_vals, dim=1)
        sdf_val = torch.min(sdf_vals, dim=1).values

        if compute_gradient:
            # Get gradient of softmin
            h_grad = h * (h - 1)
            pts_grad_1 = (h_grad * sdf_vals).unsqueeze(-1) * sdf_grads
            pts_grad_2 = h.unsqueeze(-1) * sdf_grads
            pts_grad = (pts_grad_1 + pts_grad_2)
            pts.backward(pts_grad)
            q_grad = q.grad
            self.robot_query_points.detach_()
            self._query_point_mask.requires_grad_(False)
            q.detach_()
            return sdf_val, q_grad.detach()

        return sdf_val

    def self_collision_check(self, q: torch.Tensor, compute_gradient=False, compute_hessian=False):
        """
        :param q: torch.Tensor B x dq joint angles
        """
        # Add leading batch dimension
        if len(q.shape) == 1:
            q = q.reshape(1, -1)
            B = 1
        else:
            B = q.shape[0]
        if compute_gradient:
            q.requires_grad_(True)
            q.retain_grad()

        self.robot_sdf.set_joint_configuration(q)
        tfs = self.robot_sdf.sdf.obj_frame_to_link_frame.inverse()
        # Convention in RobotSDF is that transforms are shape (num_links, num_configs, 4, 4) before flattening
        # the first two dimensions. Thus points need to be (num_links, num_configs, num_points, 3)
        batched_query_points = self.robot_query_points.unsqueeze(1).repeat(1, B, 1, 1)
        pts = tfs.transform_points(batched_query_points.reshape(-1, self.points_per_link, 3))
        pts = self.scene_transform.inverse().transform_points(pts)
        # Now we have transformed points, want to permute to be (num_configs, num_links*num_points, 3)
        pts = pts.reshape(-1, B, self.points_per_link, 3)
        pts = pts.permute(1, 0, 2, 3).reshape(B, -1, 3)
        sdf_vals, sdf_grads = self.robot_sdf(pts)

        # only consider gradient on points that are in collision
        # scale sdf grads by magnitude of sdf value
        sdf_vals = sdf_vals * self._query_point_mask
        sdf_grads = sdf_grads * self._query_point_mask.unsqueeze(-1)

        h = torch.softmax(-self.softmin_T * sdf_vals, dim=1)
        sdf_val = torch.sum(h * sdf_vals, dim=1)

        if compute_gradient:
            # Get gradient of softmin
            h_grad = h * (h - 1)
            pts_grad_1 = (h_grad * sdf_vals).unsqueeze(-1) * sdf_grads
            pts_grad_2 = h.unsqueeze(-1) * sdf_grads
            pts_grad = (pts_grad_1 + pts_grad_2)
            pts.backward(pts_grad)
            q_grad = q.grad

            self.robot_query_points.detach_()
            self._query_point_mask.requires_grad_(False)
            q.detach_()
            return sdf_val, q_grad.detach()

        return sdf_val
