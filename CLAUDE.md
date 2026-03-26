# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

pytorch_volumetric is a PyTorch library for geometric/volumetric data structures used in robotics, particularly signed distance fields (SDFs), voxel grids, and chamfer distances. Designed to work alongside `pytorch_kinematics` for robot-aware collision checking and spatial queries.

## Build & Test Commands

```bash
pip install -e .                          # Install in development mode
pip install -e ".[test]"                  # Install with test dependencies (pytest, pybullet)
pytest                                    # Run all tests
pytest tests/test_sdf.py                  # Run a single test file
pytest tests/test_sdf.py::test_name       # Run a single test function
```

No linter or formatter is configured.

## Architecture

### Core Abstractions (src/pytorch_volumetric/)

**SDF hierarchy** (`sdf.py`): All SDFs extend `ObjectFrameSDF`, which is a PyTorch `Function` with custom backward pass (gradient of SDF value w.r.t. query points = SDF gradient). Queries return `SDFQuery` namedtuple: `(closest, distance, gradient, normal)`.

- `MeshSDF` — accurate ray-tracing SDF via Open3D (expensive per-query)
- `CachedSDF` — voxel-grid lookup with configurable `OutOfBoundsStrategy` (fall back to ground truth or use bounding box)
- `ComposedSDF` — unions multiple SDFs with per-object transforms for multi-object scenes
- `SphereSDF` — analytic sphere primitive
- `ObjectFactory` / `MeshObjectFactory` — load meshes, provide `object_frame_closest_point()` and `bounding_box()`

**Robot SDF** (`model_to_sdf.py`): `RobotSDF` wraps a `pytorch_kinematics.Chain` to compute SDF for a full robot given joint configurations. Uses `cache_link_sdf_factory()` to build per-link CachedSDFs. Supports batched queries over both configurations and query points.

**Voxel structures** (`voxel.py`): `VoxelGrid` is a regular 3D grid; `ExpandingVoxelGrid` auto-expands bounds; `VoxelSet` is sparse storage. All implement the `Voxels` interface.

**Distance metrics** (`chamfer.py`): `batch_chamfer_dist()` for point-to-mesh distance, `PlausibleDiversity` for evaluating transform set coverage.

### Key Design Patterns

- **Autograd compatibility**: SDF queries support gradient computation through PyTorch autograd. The custom backward in `ObjectFrameSDF` passes SDF gradients as the gradient of distance w.r.t. query points.
- **Arbitrary batch dimensions**: Query points and robot configurations support arbitrary leading batch dimensions via reshape/flatten patterns and `arm_pytorch_utilities.handle_batch_input`.
- **Transform composition**: Uses `pytorch_kinematics.Transform3d` throughout for coordinate frame management.
- **Voxel caching**: `CachedSDF` uses `multidim_indexing.TorchMultidimView` for efficient N-dimensional interpolated lookup.

### Key Dependencies

- `pytorch_kinematics` — `Chain` (URDF loader), `Transform3d` (rigid transforms)
- `open3d` — mesh I/O and ray-casting for closest-point queries
- `arm-pytorch-utilities` — tensor batch handling utilities (`handle_batch_input` decorator)
- `multidim-indexing` — `TorchMultidimView` for voxel grid interpolation
