import typing
import torch
import pytorch_kinematics as pk
from pytorch_volumetric import sdf
import logging

logger = logging.getLogger(__file__)


class RobotSDF(sdf.ObjectFrameSDF):
    """Create an SDF for a robot model described by a pytorch_kinematics Chain.
    The SDF is conditioned on a joint configuration which must be set."""

    def __init__(self, chain: pk.Chain, default_joint_config=None, path_prefix='',
                 link_sdf_cls: typing.Callable[[sdf.ObjectFactory, ...], sdf.ObjectFrameSDF] = sdf.MeshSDF,
                 **kwargs):
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

        sdfs = []
        # get the link meshes from the frames and create meshes
        for frame_name in self.frame_names:
            frame = self.chain.find_frame(frame_name)
            # TODO create SDF for non-mesh primitives
            # TODO consider the visual offset transform
            for link_vis in frame.link.visuals:
                if link_vis.geom_type == "mesh":
                    logger.info(f"{frame.link.name} offset {link_vis.offset}")
                    link_obj = sdf.StubObjectFactory(link_vis.geom_param, path_prefix=path_prefix)
                    link_sdf = link_sdf_cls(link_obj, **kwargs)
                    self.sdf_to_link_name.append(frame.link.name)
                    sdfs.append(link_sdf)
                else:
                    logger.warning(f"Cannot handle non-mesh link visual type {link_vis}")

        self.sdf = sdf.ComposedSDF(sdfs, self.object_to_link_frames)
        self.set_joint_configuration(default_joint_config)

    def set_joint_configuration(self, joint_config=None):
        """Set the joint configuration of the robot"""
        if joint_config is None:
            joint_config = torch.zeros(len(self.joint_names), device=self.device, dtype=self.dtype)
        tf = self.chain.forward_kinematics(joint_config, end_only=False)
        tsfs = []
        for link_name in self.sdf_to_link_name:
            tsfs.append(tf[link_name].get_matrix())
        self.object_to_link_frames = pk.Transform3d(matrix=torch.cat(tsfs).inverse())
        if self.sdf is not None:
            self.sdf.obj_frame_to_each_frame = self.object_to_link_frames

    def __call__(self, points_in_object_frame):
        return self.sdf(points_in_object_frame)
