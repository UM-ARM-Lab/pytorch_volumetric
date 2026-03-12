"""
Performance benchmarks using pytest-benchmark.

Run with: pytest tests/test_benchmarks.py -v
Results are saved to .benchmarks/ and can be compared across runs with:
    pytest tests/test_benchmarks.py --benchmark-compare
"""
import os
import math

import torch
import pytorch_kinematics as pk
import pytorch_volumetric as pv
from pytorch_volumetric.voxel import VoxelGrid

import pytest

TEST_DIR = os.path.dirname(__file__)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def mesh_obj():
    return pv.MeshObjectFactory(os.path.join(TEST_DIR, "YcbPowerDrill/textured_simple_reoriented.obj"))


@pytest.fixture(scope="module")
def mesh_sdf(mesh_obj):
    return pv.MeshSDF(mesh_obj)


@pytest.fixture(scope="module")
def cached_sdf_bb(mesh_obj, mesh_sdf, device):
    return pv.CachedSDF(
        'bench_drill_bb', resolution=0.01,
        range_per_dim=mesh_obj.bounding_box(padding=0.1),
        gt_sdf=mesh_sdf,
        out_of_bounds_strategy=pv.OutOfBoundsStrategy.BOUNDING_BOX,
        device=device, clean_cache=True, cache_path="bench_bb_cache.pkl",
    )


@pytest.fixture(scope="module")
def cached_sdf_gt(mesh_obj, mesh_sdf, device):
    return pv.CachedSDF(
        'bench_drill_gt', resolution=0.01,
        range_per_dim=mesh_obj.bounding_box(padding=0.1),
        gt_sdf=mesh_sdf,
        out_of_bounds_strategy=pv.OutOfBoundsStrategy.LOOKUP_GT_SDF,
        device=device, clean_cache=True, cache_path="bench_gt_cache.pkl",
    )


@pytest.fixture(scope="module")
def sample_pts(mesh_obj, device):
    """Pre-generate sample points at various sizes."""
    coords, all_pts = pv.get_coordinates_and_points_in_grid(0.002, mesh_obj.bounding_box(0.01), device=device)
    perm = torch.randperm(len(all_pts), device=device)
    return {n: all_pts[perm[:n]] for n in [100, 1000, 5000, 20000, 100000]}


@pytest.fixture(scope="module")
def robot_sdf(device):
    pybullet_data = pytest.importorskip("pybullet_data")
    search_path = pybullet_data.getDataPath()
    urdf = os.path.join(search_path, "kuka_iiwa/model.urdf")
    chain = pk.build_serial_chain_from_urdf(open(urdf).read(), "lbr_iiwa_link_7")
    chain = chain.to(device=device)
    s = pv.RobotSDF(chain, path_prefix=os.path.join(search_path, "kuka_iiwa"),
                     link_sdf_cls=pv.cache_link_sdf_factory(resolution=0.02, padding=1.0, device=device))
    th = torch.tensor([0.0, -math.pi / 4.0, 0.0, math.pi / 2.0, 0.0, math.pi / 4.0, 0.0], device=device)
    s.set_joint_configuration(th)
    return s


# ── MeshSDF vs CachedSDF ─────────────────────────────────────────────────────

@pytest.mark.parametrize("n_points", [100, 1000, 5000, 20000, 100000])
def test_bench_mesh_sdf(benchmark, mesh_sdf, sample_pts, n_points):
    pts = sample_pts[n_points]
    benchmark(mesh_sdf, pts)


@pytest.mark.parametrize("n_points", [100, 1000, 5000, 20000, 100000])
def test_bench_cached_sdf_bounding_box(benchmark, cached_sdf_bb, sample_pts, n_points):
    pts = sample_pts[n_points]
    benchmark(cached_sdf_bb, pts)


@pytest.mark.parametrize("n_points", [100, 1000, 5000, 20000, 100000])
def test_bench_cached_sdf_gt_fallback(benchmark, cached_sdf_gt, sample_pts, n_points):
    pts = sample_pts[n_points]
    benchmark(cached_sdf_gt, pts)


# ── MeshSDF with autograd ────────────────────────────────────────────────────

@pytest.mark.parametrize("n_points", [100, 1000, 5000, 20000, 100000])
def test_bench_mesh_sdf_with_grad(benchmark, mesh_sdf, sample_pts, n_points):
    pts = sample_pts[n_points].clone().requires_grad_(True)

    def query_with_backward():
        if pts.grad is not None:
            pts.grad.zero_()
        val, grad = mesh_sdf(pts)
        val.sum().backward()

    benchmark(query_with_backward)


# ── SphereSDF (baseline, no mesh I/O) ────────────────────────────────────────

@pytest.mark.parametrize("n_points", [1000, 10000, 100000])
def test_bench_sphere_sdf(benchmark, device, n_points):
    sdf = pv.SphereSDF(1.0)
    pts = torch.randn(n_points, 3, device=device)
    benchmark(sdf, pts)


# ── VoxelGrid operations ─────────────────────────────────────────────────────

def test_bench_voxel_grid_set(benchmark, device):
    resolution = 0.01
    range_per_dim = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    pts = torch.rand(10000, 3, device=device)
    values = torch.rand(10000, device=device)

    def do_set():
        vg = VoxelGrid(resolution, range_per_dim, device=device)
        vg[pts] = values

    benchmark(do_set)


def test_bench_voxel_grid_get(benchmark, device):
    resolution = 0.01
    range_per_dim = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    vg = VoxelGrid(resolution, range_per_dim, device=device)
    pts = torch.rand(10000, 3, device=device)
    vg[pts] = torch.rand(10000, device=device)

    query_pts = torch.rand(10000, 3, device=device)
    benchmark(vg.__getitem__, query_pts)


# ── RobotSDF ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def robot_chain(device):
    pybullet_data = pytest.importorskip("pybullet_data")
    search_path = pybullet_data.getDataPath()
    urdf = os.path.join(search_path, "kuka_iiwa/model.urdf")
    chain = pk.build_serial_chain_from_urdf(open(urdf).read(), "lbr_iiwa_link_7")
    return chain.to(device=device)


def test_bench_robot_sdf_single_config(benchmark, robot_sdf, device):
    query_range = [[-0.5, 0.5], [0.0, 0.0], [-0.2, 0.8]]
    _, pts = pv.get_coordinates_and_points_in_grid(0.05, query_range, device=device)
    benchmark(robot_sdf, pts)


def test_bench_robot_sdf_batch_configs(benchmark, robot_sdf, device):
    N = 10
    th = torch.randn(N, 7, device=device) * 0.1
    robot_sdf.set_joint_configuration(th)
    query_range = [[-0.5, 0.5], [0.0, 0.0], [-0.2, 0.8]]
    _, pts = pv.get_coordinates_and_points_in_grid(0.05, query_range, device=device)
    benchmark(robot_sdf, pts)


@pytest.mark.parametrize("n_points", [1000, 100000])
def test_bench_robot_sdf_query(benchmark, robot_sdf, device, n_points):
    """Benchmark full RobotSDF query (FK already done) at various point counts."""
    th = torch.tensor([0.0, -math.pi / 4.0, 0.0, math.pi / 2.0, 0.0, math.pi / 4.0, 0.0], device=device)
    robot_sdf.set_joint_configuration(th)
    pts = torch.randn(n_points, 3, device=device) * 0.5
    benchmark(robot_sdf, pts)


# ── Forward Kinematics ───────────────────────────────────────────────────────

def test_bench_fk_single_config(benchmark, robot_chain, device):
    """Benchmark forward kinematics for a single configuration."""
    th = torch.zeros(7, device=device)
    benchmark(robot_chain.forward_kinematics, th, end_only=False)


@pytest.mark.parametrize("n_configs", [10, 100, 1000, 100000])
def test_bench_fk_batch_configs(benchmark, robot_chain, device, n_configs):
    """Benchmark batched forward kinematics at various batch sizes."""
    th = torch.randn(n_configs, 7, device=device) * 0.1
    benchmark(robot_chain.forward_kinematics, th, end_only=False)


@pytest.mark.parametrize("n_configs", [10, 100, 1000, 100000])
def test_bench_fk_end_only(benchmark, robot_chain, device, n_configs):
    """Benchmark FK computing only end-effector (not all links)."""
    th = torch.randn(n_configs, 7, device=device) * 0.1
    benchmark(robot_chain.forward_kinematics, th, end_only=True)


# ── Forward Kinematics with torch.compile ─────────────────────────────────────

@pytest.fixture(scope="module")
def compiled_fk(robot_chain):
    return torch.compile(robot_chain.forward_kinematics)


@pytest.fixture(scope="module")
def compiled_fk_end_only(robot_chain):
    # Wrap to bind end_only=True since compile doesn't handle keyword arg changes well
    def fk_end_only(th):
        return robot_chain.forward_kinematics(th, end_only=True)
    return torch.compile(fk_end_only)


@pytest.mark.parametrize("n_configs", [10, 100, 1000, 100000])
def test_bench_fk_compiled(benchmark, compiled_fk, device, n_configs):
    """Benchmark torch.compiled forward kinematics (all links)."""
    th = torch.randn(n_configs, 7, device=device) * 0.1
    # Warmup to trigger compilation
    compiled_fk(th, end_only=False)
    benchmark(compiled_fk, th, end_only=False)


@pytest.mark.parametrize("n_configs", [10, 100, 1000, 100000])
def test_bench_fk_compiled_end_only(benchmark, compiled_fk_end_only, device, n_configs):
    """Benchmark torch.compiled FK end-effector only."""
    th = torch.randn(n_configs, 7, device=device) * 0.1
    # Warmup
    compiled_fk_end_only(th)
    benchmark(compiled_fk_end_only, th)
