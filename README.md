## Pytorch Volumetric

- signed distance field (SDF) pytorch implementation with parallelized query for value and gradients
- voxel grids with automatic expanding range
- unidirectional chamfer distance (points to mesh)
- robot model to SDF with parallelized query over robot configurations and points

## Installation

```shell
pip install pytorch-volumetric
```

For development, clone repository somewhere, then `pip3 install -e .` to install in editable mode.
For testing, run `pytest` in the root directory.

## Usage

See `tests` for code samples; some are also shown here

### SDF from mesh

```python
import pytorch_volumetric as pv

# supposing we have an object mesh (most formats supported) - from https://github.com/eleramp/pybullet-object-models
obj = pv.MeshObjectFactory("YcbPowerDrill/textured_simple_reoriented.obj")
sdf = pv.MeshSDF(obj)
```

### Cached SDF

```python
import pytorch_volumetric as pv

obj = pv.MeshObjectFactory("YcbPowerDrill/textured_simple_reoriented.obj")
sdf = pv.MeshSDF(obj)
# caching the SDF via a voxel grid to accelerate queries
cached_sdf = pv.CachedSDF('drill', resolution=0.01, range_per_dim=obj.bounding_box(padding=0.1), gt_sdf=sdf)
```

By default, query points outside the cache will be compared against the object bounding box.
To instead use the ground truth SDF, pass `out_of_bounds_strategy=pv.OutOfBoundsStrategy.LOOKUP_GT_SDF` to 
the constructor.

Note that the bounding box comparison will always under-approximate the SDF value, but empirically it is sufficient
for most applications when querying out of bound points. It is **dramatically faster** than using the ground truth SDF.

### Composed SDF
Multiple SDFs can be composed together to form an SDF that is convenient to query. This may be because your scene
is composed of multiple objects and you have them as separate meshes. Note: the objects should not be overlapping or
share faces, otherwise there will be artifacts in the SDF query in determining interior-ness. 

```python
import pytorch_volumetric as pv
import pytorch_kinematics as pk

obj = pv.MeshObjectFactory("YcbPowerDrill/textured_simple_reoriented.obj")

# 2 drills in the world
sdf1 = pv.MeshSDF(obj)
sdf2 = pv.MeshSDF(obj)
# need to specify the transform of each SDF frame
tsf1 = pk.Translate(0.1, 0, 0)
tsf2 = pk.Translate(-0.2, 0, 0.2)
sdf = pv.ComposedSDF([sdf1, sdf2], tsf1.stack(tsf2))
```

### SDF value and gradient queries

Suppose we have an `ObjectFrameSDF` (such as created from above)

```python
import numpy as np
import pytorch_volumetric as pv

# get points in a grid in the object frame
query_range = np.array([
    [-1, 0.5],
    [-0.5, 0.5],
    [-0.2, 0.8],
])

coords, pts = pv.get_coordinates_and_points_in_grid(0.01, query_range)
# N x 3 points 
# we can also query with batched points B x N x 3, B can be any number of batch dimensions
sdf_val, sdf_grad = sdf(pts)
# sdf_val is N, or B x N, the SDF value in meters
# sdf_grad is N x 3 or B x N x 3, the normalized SDF gradient (points along steepest increase in SDF)
```

### Plotting SDF Slice

```python
import pytorch_volumetric as pv
import numpy as np

# supposing we have an object mesh (most formats supported) - from https://github.com/eleramp/pybullet-object-models
obj = pv.MeshObjectFactory("YcbPowerDrill/textured_simple_reoriented.obj")
sdf = pv.MeshSDF(obj)
# need a dimension with no range to slice; here it's y
query_range = np.array([
    [-0.15, 0.2],
    [0, 0],
    [-0.1, 0.2],
])
pv.draw_sdf_slice(sdf, query_range)
```

![drill SDF](https://i.imgur.com/TFaGmx6.png)

### Robot Model to SDF

For many applications such as collision checking, it is useful to have the
SDF of a multi-link robot in certain configurations.
First, we create the robot model (loaded from URDF, SDF, MJCF, ...) with
[pytorch kinematics](https://github.com/UM-ARM-Lab/pytorch_kinematics).
For example, we will be using the KUKA 7 DOF arm model from pybullet data

```python
import os
import torch
import pybullet_data
import pytorch_kinematics as pk
import pytorch_volumetric as pv

urdf = "kuka_iiwa/model.urdf"
search_path = pybullet_data.getDataPath()
full_urdf = os.path.join(search_path, urdf)
chain = pk.build_serial_chain_from_urdf(open(full_urdf).read(), "lbr_iiwa_link_7")
d = "cuda" if torch.cuda.is_available() else "cpu"

chain = chain.to(device=d)
# paths to the link meshes are specified with their relative path inside the URDF
# we need to give them the path prefix as we need their absolute path to load
s = pv.RobotSDF(chain, path_prefix=os.path.join(search_path, "kuka_iiwa"))
```

By default, each link will have a `MeshSDF`. To instead use `CachedSDF` for faster queries

```python
s = pv.RobotSDF(chain, path_prefix=os.path.join(search_path, "kuka_iiwa"),
                link_sdf_cls=pv.cache_link_sdf_factory(resolution=0.02, padding=1.0, device=d))
```

Which when the `y=0.02` SDF slice is visualized:
![sdf slice](https://i.imgur.com/Putw72A.png)

With surface points corresponding to:
![wireframe](https://i.imgur.com/L3atG9h.png)
![solid](https://i.imgur.com/XiAks7a.png)

Queries on this SDF is dependent on the joint configurations (by default all zero).
**Queries are batched across configurations and query points**. For example, we have a batch of
joint configurations to query

```python
th = torch.tensor([0.0, -math.pi / 4.0, 0.0, math.pi / 2.0, 0.0, math.pi / 4.0, 0.0], device=d)
N = 200
th_perturbation = torch.randn(N - 1, 7, device=d) * 0.1
# N x 7 joint values
th = torch.cat((th.view(1, -1), th_perturbation + th))
```

And also a batch of points to query (same points for each configuration):

```python
y = 0.02
query_range = np.array([
    [-1, 0.5],
    [y, y],
    [-0.2, 0.8],
])
# M x 3 points
coords, pts = pv.get_coordinates_and_points_in_grid(0.01, query_range, device=s.device)
```

We set the batch of joint configurations and query:

```python
s.set_joint_configuration(th)
# N x M SDF value
# N x M x 3 SDF gradient
sdf_val, sdf_grad = s(pts)
```

Queries are reasonably quick. For the 7 DOF Kuka arm (8 links), using `CachedSDF` on a RTX 2080 Ti,
and using CUDA, we get

```shell
N=20, M=15251, elapsed: 37.688577ms time per config and point: 0.000124ms
N=200, M=15251, elapsed: elapsed: 128.645445ms time per config and point: 0.000042ms
```