import abc
import enum
import math
import os
import typing
from typing import NamedTuple, Union

import numpy as np
import open3d as o3d

import torch
from arm_pytorch_utilities import tensor_utils, rand
from multidim_indexing import torch_view
from functools import partial

from pytorch_volumetric.voxel import VoxelGrid, get_divisible_range_by_resolution, get_coordinates_and_points_in_grid
import pytorch_kinematics as pk
import logging

logger = logging.getLogger(__name__)


class SDFQuery(NamedTuple):
    closest: torch.Tensor
    distance: torch.Tensor
    gradient: torch.Tensor
    normal: Union[torch.Tensor, None]


class ObjectFactory(abc.ABC):
    def __init__(self, name, scale=1.0, vis_frame_pos=(0, 0, 0), vis_frame_rot=(0, 0, 0, 1),
                 plausible_suboptimality=0.001, **kwargs):
        self.name = name
        self.scale = scale if scale is not None else 1.0
        # frame from model's base frame to the simulation's use of the model
        self.vis_frame_pos = vis_frame_pos
        self.vis_frame_rot = vis_frame_rot
        self.other_load_kwargs = kwargs
        self.plausible_suboptimality = plausible_suboptimality

        # use external mesh library to compute closest point for non-convex meshes
        self._mesh = None
        self._mesht = None
        self._raycasting_scene = None
        self._face_normals = None

    def __reduce__(self):
        return partial(self.__class__, scale=self.scale, vis_frame_pos=self.vis_frame_pos,
                       vis_frame_rot=self.vis_frame_rot,
                       plausible_suboptimality=self.plausible_suboptimality, **self.other_load_kwargs), \
            (self.name,)

    @abc.abstractmethod
    def make_collision_obj(self, z, rgba=None):
        """Create collision object of fixed and position along x-y; returns the object ID and bounding box"""

    @abc.abstractmethod
    def get_mesh_resource_filename(self):
        """Return the path to the mesh resource file (.obj, .stl, ...)"""

    def get_mesh_high_poly_resource_filename(self):
        """Return the path to the high poly mesh resource file"""
        return self.get_mesh_resource_filename()

    def draw_mesh(self, dd, name, pose, rgba, object_id=None):
        frame_pos = np.array(self.vis_frame_pos) * self.scale
        return dd.draw_mesh(name, self.get_mesh_resource_filename(), pose, scale=self.scale, rgba=rgba,
                            object_id=object_id, vis_frame_pos=frame_pos, vis_frame_rot=self.vis_frame_rot)

    def bounding_box(self, padding=0., padding_ratio=0.):
        if self._mesh is None:
            self.precompute_sdf()

        aabb = self._mesh.get_axis_aligned_bounding_box()
        world_min = aabb.get_min_bound()
        world_max = aabb.get_max_bound()
        # already scaled, but we add a little padding
        ranges = np.array(list(zip(world_min, world_max)))
        extents = ranges[:, 1] - ranges[:, 0]
        ranges[:, 0] -= padding + padding_ratio * extents
        ranges[:, 1] += padding + padding_ratio * extents
        return ranges

    def center(self):
        """Get center of mass assuming uniform density. Return is in object frame"""
        if self._mesh is None:
            self.precompute_sdf()
        return self._mesh.get_center()

    def precompute_sdf(self):
        # scale mesh the approrpiate amount
        full_path = self.get_mesh_high_poly_resource_filename()
        full_path = os.path.expanduser(full_path)
        if not os.path.exists(full_path):
            raise RuntimeError(f"Expected mesh file does not exist: {full_path}")
        self._mesh = o3d.io.read_triangle_mesh(full_path)
        # scale mesh
        scale_transform = np.eye(4)
        np.fill_diagonal(scale_transform[:3, :3], self.scale)
        self._mesh.transform(scale_transform)
        # convert from mesh object frame to simulator object frame
        x, y, z, w = self.vis_frame_rot
        self._mesh = self._mesh.rotate(o3d.geometry.get_rotation_matrix_from_quaternion((w, x, y, z)),
                                       center=[0, 0, 0])
        self._mesh = self._mesh.translate(np.array(self.vis_frame_pos) * self.scale)

        self._mesht = o3d.t.geometry.TriangleMesh.from_legacy(self._mesh)
        self._raycasting_scene = o3d.t.geometry.RaycastingScene()
        _ = self._raycasting_scene.add_triangles(self._mesht)
        self._mesh.compute_triangle_normals()
        self._face_normals = np.asarray(self._mesh.triangle_normals)

    @tensor_utils.handle_batch_input(n=2)
    def _do_object_frame_closest_point(self, points_in_object_frame, compute_normal=False):
        if self._mesh is None:
            self.precompute_sdf()

        if torch.is_tensor(points_in_object_frame):
            dtype = points_in_object_frame.dtype
            device = points_in_object_frame.device
            points_in_object_frame = points_in_object_frame.detach().cpu().numpy()
        else:
            dtype = torch.float
            device = "cpu"
        points_in_object_frame = points_in_object_frame.astype(np.float32)

        closest = self._raycasting_scene.compute_closest_points(points_in_object_frame)
        closest_points = closest['points']
        face_ids = closest['primitive_ids']
        pts = closest_points.numpy()
        # negative SDF gradient outside the object and positive SDF gradient inside the object
        gradient = pts - points_in_object_frame

        distance = np.linalg.norm(gradient, axis=-1)
        # normalize gradients
        has_direction = distance > 0
        gradient[has_direction] = gradient[has_direction] / distance[has_direction, None]

        # ensure ray destination is outside the object
        ray_destination = np.repeat(self.bounding_box(padding=1.0)[None, :, 1], points_in_object_frame.shape[0], axis=0)
        # add noise to ray destination, this helps reduce artifacts in the sdf
        ray_destination = ray_destination + 1e-4 * np.random.randn(*points_in_object_frame.shape)
        ray_destination = ray_destination.astype(np.float32)
        # check if point is inside the object
        rays = np.concatenate([points_in_object_frame, ray_destination], axis=-1)
        intersection_counts = self._raycasting_scene.count_intersections(rays).numpy()
        is_inside = intersection_counts % 2 == 1
        distance[is_inside] = distance[is_inside] * -1
        # fix gradient direction to point away from surface outside
        gradient[~is_inside] = gradient[~is_inside] * -1

        # for any points very close to the surface, it is better to use the surface normal as the gradient
        # this is because the closest point on the surface may be noisy when close by
        # e.g. if you are actually on the surface, the closest surface point is itself so you get no gradient info
        on_surface = np.abs(distance) < 1e-3
        surface_normals = self._face_normals[face_ids.numpy()[on_surface]]
        gradient[on_surface] = surface_normals

        pts, distance, gradient = tensor_utils.ensure_tensor(device, dtype, pts, distance, gradient)

        normals = None
        if compute_normal:
            normals = self._face_normals[face_ids.numpy()]
            normals = torch.tensor(normals, device=device, dtype=dtype)
        return pts, distance, gradient, normals

    def object_frame_closest_point(self, points_in_object_frame, compute_normal=False) -> SDFQuery:
        """
        Assumes the input is in the simulator object frame and will return outputs
        also in the simulator object frame. Note that the simulator object frame and the mesh object frame may be
        different

        :param points_in_object_frame: N x 3 points in the object frame
        (can have arbitrary batch dimensions in front of N)
        :param compute_normal: bool: whether to compute surface normal at the closest point or not
        :return: dict(closest: N x 3, distance: N, gradient: N x 3, normal: N x 3)
        the closest points on the surface, their corresponding signed distance to the query point, the negative SDF
        gradient at the query point if the query point is outside, otherwise it's the positive SDF gradient
        (points from the query point to the closest point), and the surface normal at the closest point
        """

        return SDFQuery(*self._do_object_frame_closest_point(points_in_object_frame, compute_normal=compute_normal))


class MeshObjectFactory(ObjectFactory):
    def __init__(self, mesh_name, path_prefix='', **kwargs):
        self.path_prefix = path_prefix
        # whether to strip the package:// prefix from the mesh name, for example if we are loading a mesh manually
        # with a path prefix
        self.strip_package_prefix = path_prefix != ''
        # specify ranges=None to infer the range from the object's bounding box
        super(MeshObjectFactory, self).__init__(mesh_name, **kwargs)

    def __reduce__(self):
        return partial(self.__class__, path_prefix=self.path_prefix, scale=self.scale, vis_frame_pos=self.vis_frame_pos,
                       vis_frame_rot=self.vis_frame_rot,
                       plausible_suboptimality=self.plausible_suboptimality, **self.other_load_kwargs), \
            (self.name,)

    def make_collision_obj(self, z, rgba=None):
        return None, None

    def get_mesh_resource_filename(self):
        mesh_path = self.name
        if self.strip_package_prefix:
            mesh_path = mesh_path.replace("package://", "")
        return os.path.join(self.path_prefix, mesh_path)


class ObjectFrameSDF(abc.ABC):
    @abc.abstractmethod
    def __call__(self, points_in_object_frame):
        """
        Evaluate the signed distance function at given points in the object frame
        :param points_in_object_frame: B x N x d d-dimensional points (2 or 3) of B batches; located in object frame
        :return: tuple of B x N signed distance from closest object surface in m and B x N x d SDF gradient pointing
            towards higher SDF values (away from surface when outside the object and towards the surface when inside)
        """

    @abc.abstractmethod
    def surface_bounding_box(self, padding=0., padding_ratio=0.):
        """
        Get the bounding box for the 0-level set in the form of a sequence of (min,max) coordinates
        :param padding: amount to inflate the min and max from the actual bounding box
        :param padding_ratio: ratio of the extent of that dimension to use for padding; added on top of absolute padding
        :return: (min,max) for each dimension
        """

    def outside_surface(self, points_in_object_frame, surface_level=0):
        """
        Check if query points are outside the surface level set; separate from querying the values since some
        implementations may have a more efficient way of computing this
        :param points_in_object_frame:
        :param surface_level: The level set value for separating points
        :return: B x N bool
        """
        sdf_values, _ = self.__call__(points_in_object_frame)
        outside = sdf_values > surface_level
        return outside

    def get_voxel_view(self, voxels: VoxelGrid = None, dtype=torch.float, device='cpu') -> torch_view.TorchMultidimView:
        """
        Get a voxel view of a part of the SDF
        :param voxels: the voxel over which to evaluate the SDF; if left as none, take the default range which is
        implementation dependent
        :param dtype: torch type of the default voxel grid (can be safely omitted if voxels is supplied)
        :param device: torch device of the default voxel grid (can be safely omitted if voxels is supplied)
        :return:
        """
        if voxels is None:
            voxels = VoxelGrid(0.01, self.surface_bounding_box(padding=0.1).cpu().numpy(), dtype=dtype, device=device)

        pts = voxels.get_voxel_center_points()
        sdf_val, sdf_grad = self.__call__(pts.unsqueeze(0))
        cached_underlying_sdf = sdf_val.reshape([len(coord) for coord in voxels.coords])

        return torch_view.TorchMultidimView(cached_underlying_sdf, voxels.range_per_dim, invalid_value=self.__call__)

    def get_filtered_points(self, unary_filter, voxels: VoxelGrid = None, dtype=torch.float,
                            device='cpu') -> torch.tensor:
        """
        Get a N x d sequence of points extracted from a voxel grid such that their SDF values satisfy a given
        unary filter (on their SDF value)
        :param unary_filter: filter on the SDF value of each point, evaluating to true results in accepting that point
        :param voxels: voxel grid over which to evaluate each point (there can be infinitely many points satisfying
        the unary filter and we need to restrict our search over a grid of points [center of the voxels])
        :param dtype: torch type of the default voxel grid (can be safely omitted if voxels is supplied)
        :param device: torch device of the default voxel grid (can be safely omitted if voxels is supplied)
        :return:
        """
        model_voxels = self.get_voxel_view(voxels, dtype=dtype, device=device)
        interior = unary_filter(model_voxels.raw_data)
        indices = interior.nonzero()
        # these points are in object frame
        return model_voxels.ensure_value_key(indices)


class MeshSDF(ObjectFrameSDF):
    """SDF generated from direct ray-tracing calls to the mesh. This is relatively expensive."""

    def __init__(self, obj_factory: ObjectFactory, vis=None):
        self.obj_factory = obj_factory
        self.vis = vis

    def surface_bounding_box(self, **kwargs):
        return torch.tensor(self.obj_factory.bounding_box(**kwargs))

    def __call__(self, points_in_object_frame):
        N, d = points_in_object_frame.shape[-2:]

        # compute SDF value for new sampled points
        res = self.obj_factory.object_frame_closest_point(points_in_object_frame)

        # points are transformed to link frame, thus it needs to compare against the object in link frame
        # objId is not in link frame and shouldn't be moved
        if self.vis is not None:
            for i in range(N):
                self.vis.draw_point("test_point", points_in_object_frame[..., i, :], color=(1, 0, 0), length=0.005)
                self.vis.draw_2d_line(f"test_grad", points_in_object_frame[..., i, :],
                                      res.gradient[..., i, :].detach().cpu(), color=(0, 0, 0),
                                      size=2., scale=0.03)
                self.vis.draw_point("test_point_surf", res.closest[..., i, :].detach().cpu(), color=(0, 1, 0),
                                    length=0.005,
                                    label=f'{res.distance[..., i].item():.5f}')
        return res.distance, res.gradient


class ComposedSDF(ObjectFrameSDF):
    def __init__(self, sdfs: typing.Sequence[ObjectFrameSDF], obj_frame_to_each_frame: pk.Transform3d):
        """

        :param sdfs: S Object frame SDFs
        :param obj_frame_to_each_frame: [B*]S x 4 x 4 transforms from the shared object frame to the frame of each SDF
        These transforms are potentially arbitrarily batched B. Since Transform3D can only have one batch dimension,
        they are flattened
        """
        self.sdfs = sdfs
        self.obj_frame_to_link_frame: typing.Optional[pk.Transform3d] = None
        self.link_frame_to_obj_frame: typing.Optional[typing.Sequence[pk.Transform3d]] = None
        self.tsf_batch = None
        self.set_transforms(obj_frame_to_each_frame)

    def surface_bounding_box(self, **kwargs):
        bounds = []
        tsf = self.obj_frame_to_link_frame.inverse()
        for i, sdf in enumerate(self.sdfs):
            pts = sdf.surface_bounding_box(**kwargs)
            pts = tsf[self.ith_transform_slice(i)].transform_points(
                pts.to(dtype=tsf.dtype, device=tsf.device).transpose(0, 1))
            # edge case where the batch is a single element
            if self.tsf_batch is not None and len(pts.shape) == 2:
                pts = pts.unsqueeze(0)
            bounds.append(pts)
        bounds = torch.stack(bounds)

        # min over everything except the batch dimensions and the last dimension
        if self.tsf_batch is not None:
            # ignore the batch dimension
            dims = (0,) + tuple(range(2, len(bounds.shape) - 1))
        else:
            dims = tuple(range(len(bounds.shape) - 1))
        mins = bounds.amin(dim=dims)
        maxs = bounds.amax(dim=dims)
        return torch.stack((mins, maxs), dim=-1)

    def set_transforms(self, tsf: pk.Transform3d, batch_dim=None):
        self.obj_frame_to_link_frame = tsf
        self.link_frame_to_obj_frame = []
        self.tsf_batch = batch_dim
        # assume a single batch dimension when not given B x N x 4 x 4
        if tsf is not None:
            S = len(self.sdfs)
            S_tsf = len(self.obj_frame_to_link_frame)
            if self.tsf_batch is None and (S_tsf != S):
                self.tsf_batch = (S_tsf / S,)
            m = tsf.get_matrix().inverse()
            for i in range(S):
                self.link_frame_to_obj_frame.append(
                    pk.Transform3d(matrix=m[self.ith_transform_slice(i)]))

    def ith_transform_slice(self, i):
        if self.tsf_batch is None:
            return slice(i, i + 1)
        else:
            total_to_slice = math.prod(list(self.tsf_batch))
            return slice(i * total_to_slice, (i + 1) * total_to_slice)

    def __call__(self, points_in_object_frame):
        pts_shape = points_in_object_frame.shape
        # flatten it for the transform
        points_in_object_frame = points_in_object_frame.view(-1, 3)
        flat_shape = points_in_object_frame.shape
        S = len(self.sdfs)
        # pts[i] are now points in the ith SDF's frame
        pts = self.obj_frame_to_link_frame.transform_points(points_in_object_frame)
        # S x B x N x 3
        if self.tsf_batch is not None:
            pts = pts.reshape(S, *self.tsf_batch, *flat_shape)
        sdfv = []
        sdfg = []
        for i, sdf in enumerate(self.sdfs):
            # B x N for v and B x N x 3 for g
            v, g = sdf(pts[i])
            # need to transform the gradient back to the object frame
            g = self.link_frame_to_obj_frame[i].transform_normals(g)
            sdfv.append(v)
            sdfg.append(g)

        # attempt at doing things in higher dimensions
        sdfv = torch.cat(sdfv)
        sdfg = torch.cat(sdfg)

        # easier solution for flattening
        v = sdfv.reshape(S, -1)
        g = sdfg.reshape(S, -1, 3)
        # ensure S is the first dimension and take min across S (the different links)
        closest = torch.argmin(v, 0)

        all = torch.arange(0, v.shape[1])
        # B*N for vv and B*N x 3 for gg
        vv = v[closest, all]
        gg = g[closest, all]

        if self.tsf_batch is not None:
            # retrieve the original query points batch dimensions - note that they are after configuration batch
            vv = vv.reshape(*self.tsf_batch, *pts_shape[:-1])
            gg = gg.reshape(*self.tsf_batch, *pts_shape[:-1], 3)

        return vv, gg


class OutOfBoundsStrategy(enum.Enum):
    LOOKUP_GT_SDF = 0
    BOUNDING_BOX = 1  # will also always under-approximate the SDF value, but more accurate than sphere approximation


class CachedSDF(ObjectFrameSDF):
    """SDF via looking up precomputed voxel grids requiring a ground truth SDF to default to on uncached queries."""

    def __init__(self, object_name, resolution, range_per_dim, gt_sdf: ObjectFrameSDF,
                 out_of_bounds_strategy=OutOfBoundsStrategy.BOUNDING_BOX,
                 device="cpu", clean_cache=False,
                 debug_check_sdf=False, cache_path="sdf_cache.pkl"):
        """

        :param object_name: str readable name of the object; combined with the resolution and range for cache
        :param resolution: side length of each voxel cell
        :param range_per_dim: (min, max) sequence for each dimension (e.g. 3 for 3D)
        :param gt_sdf: ground truth SDF used to generate the cache and default to on queries outside of the cache
        :param out_of_bounds_strategy: what to do when a query is outside the cached range.
        LOOKUP_GT_SDF: use the ground truth SDF for the value and gradient (relatively expensive)
        BOUNDING_BOX: use the distance to the bounding box (under-approximates the SDF value)
        :param device: pytorch compatible device
        :param clean_cache: whether to ignore the existing cache and force recomputation
        :param debug_check_sdf: check that the generated SDF matches the ground truth SDF
        :param cache_path: path where to store the SDF cache for efficient loading
        """
        self.device = device
        # cache for signed distance field to object
        self.voxels = None
        # voxel grid can't handle vector values yet
        self.voxels_grad = None
        self.out_of_bounds_strategy = out_of_bounds_strategy

        cached_underlying_sdf = None
        cached_underlying_sdf_grad = None

        self.gt_sdf = gt_sdf
        self.resolution = resolution

        range_per_dim = get_divisible_range_by_resolution(resolution, range_per_dim)
        self.ranges = range_per_dim

        self.name = f"{object_name} {resolution} {tuple(range_per_dim)}"
        self.debug_check_sdf = debug_check_sdf

        if os.path.exists(cache_path):
            data = torch.load(cache_path) or {}
            try:
                cached_underlying_sdf, cached_underlying_sdf_grad = data[self.name]
                logger.info("cached sdf for %s loaded from %s", self.name, cache_path)
            except (ValueError, KeyError):
                logger.info("cached sdf invalid %s from %s, recreating", self.name, cache_path)
        else:
            data = {}

        # if we didn't load anything, then we need to create the cache and save to it
        if cached_underlying_sdf is None or clean_cache:
            if gt_sdf is None:
                raise RuntimeError("Cached SDF did not find the cache and requires an initialize queryable SDF")

            coords, pts = get_coordinates_and_points_in_grid(self.resolution, self.ranges)
            sdf_val, sdf_grad = gt_sdf(pts)
            cached_underlying_sdf = sdf_val.reshape([len(coord) for coord in coords])
            cached_underlying_sdf_grad = sdf_grad.squeeze(0)
            # cached_underlying_sdf_grad = sdf_grad.reshape(cached_underlying_sdf.shape + (3,))
            # confirm the values work
            if self.debug_check_sdf:
                debug_view = torch_view.TorchMultidimView(cached_underlying_sdf, self.ranges,
                                                          invalid_value=self._fallback_sdf_value_func)
                query = debug_view[pts]
                assert torch.allclose(sdf_val, query)

            data[self.name] = cached_underlying_sdf, cached_underlying_sdf_grad

            torch.save(data, cache_path)
            logger.info("caching sdf for %s to %s", self.name, cache_path)

        cached_underlying_sdf = cached_underlying_sdf.to(device=device)
        cached_underlying_sdf_grad = cached_underlying_sdf_grad.to(device=device)
        self.voxels = torch_view.TorchMultidimView(cached_underlying_sdf, range_per_dim,
                                                   invalid_value=self._fallback_sdf_value_func)
        self.voxels_grad = cached_underlying_sdf_grad.squeeze()

        self.bb = self.surface_bounding_box().to(device=device)

    def surface_bounding_box(self, **kwargs):
        return self.gt_sdf.surface_bounding_box(**kwargs)

    def _fallback_sdf_value_func(self, *args, **kwargs):
        sdf_val, _ = self.gt_sdf(*args, **kwargs)
        sdf_val = sdf_val.to(device=self.device)
        return sdf_val

    def __call__(self, points_in_object_frame):
        # check when points are out of cached range and use ground truth sdf for both value and grad
        keys = self.voxels.ensure_index_key(points_in_object_frame)
        keys_ravelled = self.voxels.ravel_multi_index(keys, self.voxels.shape)

        inbound_keys = self.voxels.get_valid_values(points_in_object_frame)
        out_of_bound_keys = ~inbound_keys

        # logger.info(f"out of bound keys: {out_of_bound_keys.sum()}/{out_of_bound_keys.numel()}")

        dtype = points_in_object_frame.dtype
        val = torch.zeros(keys_ravelled.shape, device=self.device, dtype=dtype)
        grad = torch.zeros(keys.shape, device=self.device, dtype=dtype)

        val[inbound_keys] = self.voxels.raw_data[keys_ravelled[inbound_keys]]
        grad[inbound_keys] = self.voxels_grad[keys_ravelled[inbound_keys]]

        points_oob = points_in_object_frame[out_of_bound_keys]
        if self.out_of_bounds_strategy == OutOfBoundsStrategy.LOOKUP_GT_SDF:
            val[out_of_bound_keys], grad[out_of_bound_keys] = self.gt_sdf(points_oob)
        elif self.out_of_bounds_strategy == OutOfBoundsStrategy.BOUNDING_BOX:
            if self.bb.dtype != dtype:
                self.bb = self.bb.to(dtype=dtype)
            # distance to bounding box
            dmin = self.bb[:, 0] - points_oob
            dmin_active = dmin > 0
            dmin[~dmin_active] = 0
            dmax = points_oob - self.bb[:, 1]
            dmax_active = dmax > 0
            dmax[~dmax_active] = 0
            dtotal = dmin + dmax
            # convert to gradient; for dmin, the dtotal component should be negative; for dmax, positive
            dtotal[dmin_active] = -dtotal[dmin_active]
            dist = dtotal.norm(dim=-1)
            # normalize gradient
            grad[out_of_bound_keys] = dtotal / dist.unsqueeze(-1)
            val[out_of_bound_keys] = dist

            # comparison with ground truth
            if self.debug_check_sdf:
                val_gt, grad_gt = self.gt_sdf(points_oob)
                diff = val_gt - val[out_of_bound_keys]
                # always under-approximate the SDF value
                assert torch.all(diff > 0)
                # cosine similarity to compare the gradient vectors
                diff_grad = torch.cosine_similarity(grad_gt, grad[out_of_bound_keys], dim=-1)
                assert torch.all(diff_grad > 0.7)
                assert diff_grad.mean() > 0.95

        if self.debug_check_sdf:
            val_gt = self._fallback_sdf_value_func(points_in_object_frame)
            # the ones that are valid should be close enough to the ground truth
            diff = torch.abs(val - val_gt)
            close_enough = diff < self.resolution
            within_bounds = self.voxels.get_valid_values(points_in_object_frame)
            assert torch.all(close_enough[within_bounds])
        return val, grad

    def outside_surface(self, points_in_object_frame, surface_level=0):
        keys = self.voxels.ensure_index_key(points_in_object_frame)
        keys_ravelled = self.voxels.ravel_multi_index(keys, self.voxels.shape)

        inbound_keys = self.voxels.get_valid_values(points_in_object_frame)

        # assume out of bound keys are outside
        outside = torch.ones(keys_ravelled.shape, device=self.device, dtype=torch.bool)
        outside[inbound_keys] = self.voxels.raw_data[keys_ravelled[inbound_keys]] > surface_level
        return outside

    def get_voxel_view(self, voxels: VoxelGrid = None, dtype=torch.float, device='cpu') -> torch_view.TorchMultidimView:
        if voxels is None:
            return self.voxels

        pts = voxels.get_voxel_center_points()
        sdf_val, sdf_grad = self.gt_sdf(pts.unsqueeze(0))
        sdf_val = sdf_val.to(device=self.device)
        cached_underlying_sdf = sdf_val.reshape([len(coord) for coord in voxels.coords])

        return torch_view.TorchMultidimView(cached_underlying_sdf, voxels.range_per_dim,
                                            invalid_value=self._fallback_sdf_value_func)


def sample_mesh_points(obj_factory: ObjectFactory = None, num_points=100, seed=0, name="",
                       clean_cache=False, dtype=torch.float, min_init_sample_points=200,
                       dbpath='model_points_cache.pkl', device="cpu", cache=None):
    given_cache = cache is not None
    if cache is not None or os.path.exists(dbpath):
        if cache is None:
            cache = torch.load(dbpath)

        if name not in cache:
            cache[name] = {}
        if seed not in cache[name]:
            cache[name][seed] = {}
        if not clean_cache and num_points in cache[name][seed]:
            res = cache[name][seed][num_points]
            res = list(v.to(device=device, dtype=dtype) if v is not None else None for v in res)
            return *res[:-1], cache
    else:
        cache = {name: {seed: {}}}

    if obj_factory is None:
        raise RuntimeError(f"Expect model points to be cached for {name} {seed} {num_points} in {dbpath}")

    if obj_factory._mesh is None:
        obj_factory.precompute_sdf()

    mesh = obj_factory._mesh

    with rand.SavedRNG():
        rand.seed(seed)

        # because the point sampling is not dispersed, we do the dispersion ourselves
        # we accomplish this by sampling more points than we need then randomly selecting a subset
        sample_num_points = max(min_init_sample_points, 2 * num_points)

        # assume mesh is in object frame
        # pcd = mesh.sample_points_poisson_disk(number_of_points=num_points, init_factor=init_factor, seed=seed)
        pcd = mesh.sample_points_uniformly(number_of_points=sample_num_points)
        points = np.asarray(pcd.points)

        # subsample
        points = np.random.permutation(points)[:num_points]

        res = obj_factory.object_frame_closest_point(points, compute_normal=True)

    points = torch.tensor(points)
    normals = res.normal

    cache[name][seed][num_points] = points, normals.cpu(), None
    # otherwise assume will be saved by the user
    if not given_cache:
        torch.save(cache, dbpath)

    return points.to(device=device, dtype=dtype), normals.to(device=device, dtype=dtype), cache
