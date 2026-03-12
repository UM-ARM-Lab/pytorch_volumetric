"""
Benchmarks specifically for ComposedSDF.forward to measure optimization impact.

Run with: pytest tests/test_bench_composed_sdf.py -v
"""
import os
import math
import time

import torch
import pytorch_kinematics as pk
import pytorch_volumetric as pv

import pytest

TEST_DIR = os.path.dirname(__file__)


@pytest.fixture(scope="module")
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


# ── SphereSDF-based ComposedSDF (no mesh I/O, isolates ComposedSDF overhead) ──

@pytest.fixture(scope="module")
def composed_spheres_8(device):
    """8 spheres at different positions (mimics robot with 8 links)."""
    sdfs = [pv.SphereSDF(0.5) for _ in range(8)]
    offsets = [float(i) * 2.0 for i in range(8)]
    tsfs = [pk.Translate(o, 0, 0, device=device) for o in offsets]
    tsf = tsfs[0].stack(*tsfs[1:])
    return pv.ComposedSDF(sdfs, tsf)


@pytest.mark.parametrize("n_points", [1000, 10000, 100000])
def test_bench_composed_spheres_8(benchmark, composed_spheres_8, device, n_points):
    """ComposedSDF with 8 SphereSDF primitives — isolates loop + transform overhead."""
    pts = torch.randn(n_points, 3, device=device) * 5.0
    benchmark(composed_spheres_8, pts)


# ── RobotSDF-based ComposedSDF (real CachedSDF lookups) ──────────────────────

@pytest.fixture(scope="module")
def robot_composed(device):
    """Build RobotSDF and return its internal ComposedSDF (already has FK applied)."""
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


@pytest.mark.parametrize("n_points", [1000, 10000, 100000])
def test_bench_robot_composed_sdf(benchmark, robot_composed, device, n_points):
    """Full RobotSDF query (8 CachedSDF links) with gradient computation."""
    pts = torch.randn(n_points, 3, device=device) * 0.5
    benchmark(robot_composed, pts)


@pytest.mark.parametrize("n_points", [1000, 10000, 100000])
def test_bench_robot_composed_sdf_no_grad(benchmark, robot_composed, device, n_points):
    """Full RobotSDF query (8 CachedSDF links) without gradient computation."""
    pts = torch.randn(n_points, 3, device=device) * 0.5
    benchmark(robot_composed, pts, compute_grad=False)


@pytest.mark.parametrize("n_points", [1000, 10000, 100000])
def test_bench_robot_composed_sdf_with_grad(benchmark, robot_composed, device, n_points):
    """RobotSDF query with autograd backward pass."""
    pts = torch.randn(n_points, 3, device=device) * 0.5
    pts.requires_grad_(True)

    def query_with_backward():
        if pts.grad is not None:
            pts.grad.zero_()
        v, g = robot_composed(pts)
        v.sum().backward()

    benchmark(query_with_backward)


# ── Breakdown benchmarks (individual steps of ComposedSDF.forward) ───────────

@pytest.fixture(scope="module")
def composed_internals(robot_composed, device):
    """Pre-compute data needed to benchmark individual steps."""
    composed = robot_composed.sdf
    N = 100000
    pts = torch.randn(N, 3, device=device) * 0.5
    return composed, pts


def test_bench_step_transform_points(benchmark, composed_internals, device):
    """Benchmark: transform query points into all link frames."""
    composed, pts = composed_internals
    pts_flat = pts.view(-1, 3)

    def do_transform():
        composed.obj_frame_to_link_frame.transform_points(pts_flat)
        if device == "cuda":
            torch.cuda.synchronize()

    benchmark(do_transform)


def test_bench_step_per_link_sdf(benchmark, composed_internals, device):
    """Benchmark: sequential CachedSDF queries across all links."""
    composed, pts = composed_internals
    pts_flat = pts.view(-1, 3)
    S = len(composed.sdfs)
    transformed = composed.obj_frame_to_link_frame.transform_points(pts_flat)
    transformed = transformed.reshape(S, *pts_flat.shape)

    def do_sdf_queries():
        for i, sdf_i in enumerate(composed.sdfs):
            sdf_i(transformed[i])
        if device == "cuda":
            torch.cuda.synchronize()

    benchmark(do_sdf_queries)


def test_bench_step_transform_grads(benchmark, composed_internals, device):
    """Benchmark: transform gradients back to object frame for all links."""
    composed, pts = composed_internals
    N = pts.shape[0]
    S = len(composed.sdfs)
    dummy_grads = [torch.randn(N, 3, device=device) for _ in range(S)]

    def do_grad_transform():
        for i in range(S):
            composed.link_frame_to_obj_frame[i].transform_normals(dummy_grads[i])
        if device == "cuda":
            torch.cuda.synchronize()

    benchmark(do_grad_transform)


def test_bench_step_argmin_gather(benchmark, composed_internals, device):
    """Benchmark: argmin across links + gather."""
    composed, pts = composed_internals
    N = pts.shape[0]
    S = len(composed.sdfs)
    v = torch.randn(S, N, device=device)
    g = torch.randn(S, N, 3, device=device)

    def do_argmin():
        closest = torch.argmin(v, 0)
        idx = torch.arange(N, device=device)
        _ = v[closest, idx]
        _ = g[closest, idx]
        if device == "cuda":
            torch.cuda.synchronize()

    benchmark(do_argmin)
