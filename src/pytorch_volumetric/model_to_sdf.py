import typing

import numpy as np
import torch
import pytorch_kinematics as pk
from pytorch_volumetric import sdf
import logging

logger = logging.getLogger(__file__)


class RobotSDF(sdf.ObjectFrameSDF):
    """Create an SDF for a robot model described by a pytorch_kinematics Chain.
    The SDF is conditioned on a joint configuration which must be set."""

    def __init__(self, chain: pk.Chain, default_joint_config=None, path_prefix='',
                 link_sdf_cls: typing.Callable[[sdf.ObjectFactory], sdf.ObjectFrameSDF] = sdf.MeshSDF):
        """

        :param chain: Robot description; each link should be a mesh type - non-mesh geometries are ignored
        :param default_joint_config: values for each joint of the robot by default; None results in all zeros
        :param path_prefix: path to search for referenced meshes inside the robot description (e.g. URDF) which may use
        relative paths. This given path is prefixed onto those relative paths in order to find the meshes.
        :param link_sdf_cls: Factory of each link's SDFs; **kwargs are forwarded to this factory
        :param kwargs: Keyword arguments fed to link_sdf_cls
        """
        self.chain = chain
        self.dtype = self.chain.dtype
        self.device = self.chain.device
        self.q = None
        self.object_to_link_frames: typing.Optional[pk.Transform3d] = None
        self.joint_names = self.chain.get_joint_parameter_names()
        self.frame_names = self.chain.get_frame_names(exclude_fixed=False)
        self.sdf: typing.Optional[sdf.ComposedSDF] = None
        self.sdf_to_link_name = []
        self.configuration_batch = None

        sdfs = []
        offsets = []
        # get the link meshes from the frames and create meshes
        for frame_name in self.frame_names:
            frame = self.chain.find_frame(frame_name)
            # TODO create SDF for non-mesh primitives
            # TODO consider the visual offset transform
            for link_vis in frame.link.visuals:
                if link_vis.geom_type == "mesh":
                    logger.info(f"{frame.link.name} offset {link_vis.offset}")
                    link_obj = sdf.MeshObjectFactory(link_vis.geom_param[0],
                                                     scale=link_vis.geom_param[1],
                                                     path_prefix=path_prefix)
                    link_sdf = link_sdf_cls(link_obj)
                    self.sdf_to_link_name.append(frame.link.name)
                    sdfs.append(link_sdf)
                    offsets.append(link_vis.offset)
                else:
                    logger.warning(f"Cannot handle non-mesh link visual type {link_vis}")

        self.offset_transforms = offsets[0].stack(*offsets[1:]).to(device=self.device, dtype=self.dtype)
        self.sdf = sdf.ComposedSDF(sdfs, self.object_to_link_frames)
        self.set_joint_configuration(default_joint_config)

    def surface_bounding_box(self, **kwargs):
        return self.sdf.surface_bounding_box(**kwargs)

    def link_bounding_boxes(self):
        """
        Get the bounding box of each link in the robot's frame under the current configuration.
        Note that the bounding box is not necessarily axis-aligned, so the returned bounding box is not just
        the min and max of the points.
        :return: [A x] [B x] 8 x 3 points of the bounding box for each link in the robot's frame
        """
        tfs = self.sdf.obj_frame_to_link_frame.inverse()
        bbs = []
        for i in range(len(self.sdf.sdfs)):
            sdf = self.sdf.sdfs[i]
            bb = aabb_to_ordered_end_points(sdf.surface_bounding_box(padding=0))
            bb = tfs.transform_points(torch.tensor(bb, device=tfs.device, dtype=tfs.dtype))[
                self.sdf.ith_transform_slice(i)]
            bbs.append(bb)
        return torch.stack(bbs).squeeze()

    def set_joint_configuration(self, joint_config=None):
        """
        Set the joint configuration of the robot
        :param joint_config: [A x] M optionally arbitrarily batched joint configurations. There are M joints; A can be
        any number of batched dimensions.
        :return:
        """
        M = len(self.joint_names)
        if joint_config is None:
            joint_config = torch.zeros(M, device=self.device, dtype=self.dtype)
        # Transform3D only works with 1 batch dimension, so we need to manually flatten any additional ones
        # save the batch dimensions for when retrieving points
        if len(joint_config.shape) > 1:
            self.configuration_batch = joint_config.shape[:-1]
            joint_config = joint_config.reshape(-1, M)
        else:
            self.configuration_batch = None
        tf = self.chain.forward_kinematics(joint_config, end_only=False)
        tsfs = []
        for link_name in self.sdf_to_link_name:
            tsfs.append(tf[link_name].get_matrix())
        # make offset transforms have compatible batch dimensions
        offset_tsf = self.offset_transforms.inverse()
        if self.configuration_batch is not None:
            # must be of shape (num_links, *self.configuration_batch, 4, 4) before flattening
            expand_dims = (None,) * len(self.configuration_batch)
            offset_tsf_mat = offset_tsf.get_matrix()[(slice(None),) + expand_dims]
            offset_tsf_mat = offset_tsf_mat.repeat(1, *self.configuration_batch, 1, 1)
            offset_tsf = pk.Transform3d(matrix=offset_tsf_mat.reshape(-1, 4, 4))

        tsfs = torch.cat(tsfs)
        self.object_to_link_frames = offset_tsf.compose(pk.Transform3d(matrix=tsfs).inverse())
        if self.sdf is not None:
            self.sdf.set_transforms(self.object_to_link_frames, batch_dim=self.configuration_batch)

    def __call__(self, points_in_object_frame):
        """
        Query for SDF value and SDF gradients for points in the robot's frame
        :param points_in_object_frame: [B x] N x 3 optionally arbitrarily batched points in the robot frame; B can be
        any number of batch dimensions.
        :return: [A x] [B x] N SDF value, and [A x] [B x] N x 3 SDF gradient. A are the configurations' arbitrary
        number of batch dimensions.
        """
        return self.sdf(points_in_object_frame)


def cache_link_sdf_factory(resolution=0.01, padding=0.1, **kwargs):
    def create_sdf(obj_factory: sdf.ObjectFactory):
        gt_sdf = sdf.MeshSDF(obj_factory)
        return sdf.CachedSDF(obj_factory.name, resolution, obj_factory.bounding_box(padding=padding), gt_sdf, **kwargs)

    return create_sdf


def aabb_to_ordered_end_points(aabb, arrange_in_sequential_order=False):
    aabbMin = aabb[:, 0]
    aabbMax = aabb[:, 1]
    if arrange_in_sequential_order:
        arr = [
            [aabbMin[0], aabbMin[1], aabbMin[2]],
            [aabbMax[0], aabbMin[1], aabbMin[2]],
            [aabbMax[0], aabbMax[1], aabbMin[2]],
            [aabbMin[0], aabbMax[1], aabbMin[2]],
            [aabbMin[0], aabbMin[1], aabbMin[2]],
            [aabbMin[0], aabbMin[1], aabbMax[2]],
            [aabbMax[0], aabbMin[1], aabbMax[2]],
            [aabbMax[0], aabbMin[1], aabbMin[2]],
            [aabbMax[0], aabbMin[1], aabbMax[2]],
            [aabbMax[0], aabbMax[1], aabbMax[2]],
            [aabbMax[0], aabbMax[1], aabbMin[2]],
            [aabbMax[0], aabbMax[1], aabbMax[2]],
            [aabbMin[0], aabbMax[1], aabbMax[2]],
            [aabbMin[0], aabbMax[1], aabbMin[2]],
            [aabbMin[0], aabbMax[1], aabbMax[2]],
            [aabbMin[0], aabbMin[1], aabbMax[2]],
        ]
    else:
        arr = [
            [aabbMin[0], aabbMin[1], aabbMin[2]],
            [aabbMax[0], aabbMin[1], aabbMin[2]],
            [aabbMin[0], aabbMax[1], aabbMin[2]],
            [aabbMin[0], aabbMin[1], aabbMax[2]],
            [aabbMin[0], aabbMax[1], aabbMax[2]],
            [aabbMax[0], aabbMin[1], aabbMax[2]],
            [aabbMax[0], aabbMax[1], aabbMin[2]],
            [aabbMax[0], aabbMax[1], aabbMax[2]]
        ]
    if torch.is_tensor(aabb):
        return torch.tensor(arr, device=aabb.device, dtype=aabb.dtype)
    return np.array(arr)
