"""
Benchmark script for batched-config no-grad SDF queries.
Measures runtime and peak memory for the multi-config planning use case.
Tests both scattered and localized query points to evaluate AABB rejection.

Run: python tests/bench_multiconfig.py [--device cuda]
"""
import argparse
import math
import os
import time

import torch
import pytorch_kinematics as pk
import pytorch_volumetric as pv


def build_robot_sdf(device):
    import pybullet_data
    search_path = pybullet_data.getDataPath()
    urdf = os.path.join(search_path, "kuka_iiwa/model.urdf")
    chain = pk.build_serial_chain_from_urdf(open(urdf).read(), "lbr_iiwa_link_7")
    chain = chain.to(device=device)
    return pv.RobotSDF(chain, path_prefix=os.path.join(search_path, "kuka_iiwa"),
                        link_sdf_cls=pv.cache_link_sdf_factory(resolution=0.02, padding=1.0, device=device))


def set_batch_config(robot_sdf, n_configs, device):
    th_base = torch.tensor([0.0, -math.pi / 4.0, 0.0, math.pi / 2.0, 0.0, math.pi / 4.0, 0.0], device=device)
    th = th_base.unsqueeze(0) + torch.randn(n_configs, 7, device=device) * 0.1
    robot_sdf.set_joint_configuration(th)


def measure(fn, warmup=3, repeats=10, device="cpu"):
    """Run fn, return (mean_ms, std_ms, peak_memory_mb)."""
    for _ in range(warmup):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated()

    times = []
    for _ in range(repeats):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    mean = sum(times) / len(times)
    std = (sum((t - mean) ** 2 for t in times) / len(times)) ** 0.5

    if device == "cuda":
        peak_mem = (torch.cuda.max_memory_allocated() - mem_before) / 1e6
    else:
        peak_mem = float('nan')

    return mean, std, peak_mem


def make_localized_points(robot_sdf, n_points, device):
    """Generate points localized near the end effector (link 7)."""
    # Get end-effector position from FK with a single config
    th = torch.tensor([[0.0, -math.pi / 4.0, 0.0, math.pi / 2.0, 0.0, math.pi / 4.0, 0.0]], device=device)
    fk = robot_sdf.chain.forward_kinematics(th)
    ee_pos = fk.get_matrix()[0, :3, 3]
    # Small cube (0.1m) around end effector
    pts = ee_pos + (torch.rand(n_points, 3, device=device) - 0.5) * 0.1
    return pts


def run_benchmark(robot, configs, pts_factory, label, device):
    print(f"\n--- {label} ---")
    print(f"{'B':>6} x {'N':>6} | {'Time (ms)':>12} | {'Std (ms)':>10} | {'Per-cfg (ms)':>13} | {'Peak MB':>10}")
    print("-" * 75)

    for B, N in configs:
        pts = pts_factory(N)
        torch.manual_seed(42)
        set_batch_config(robot, B, device)

        try:
            mean, std, peak = measure(lambda: robot(pts, compute_grad=False), device=device)
            print(f"{B:>6} x {N:>6} | {mean:>12.1f} | {std:>10.1f} | {mean/B:>13.3f} | {peak:>10.1f}")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"{B:>6} x {N:>6} | {'OOM':>12} |")
                if device == "cuda":
                    torch.cuda.empty_cache()
            else:
                raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = args.device

    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    robot = build_robot_sdf(device)

    configs = [
        (10, 10000),
        (100, 10000),
        (500, 10000),
        (1000, 10000),
        (2000, 10000),
        (5000, 10000),
        (1000, 1000),
        (5000, 1000),
    ]

    # Scattered points: 0.5m std around origin (covers whole workspace)
    run_benchmark(robot, configs,
                  lambda n: torch.randn(n, 3, device=device) * 0.5,
                  "Scattered points (0.5m std)", device)

    # Localized points: 0.1m cube near end effector
    run_benchmark(robot, configs,
                  lambda n: make_localized_points(robot, n, device),
                  "Localized points (0.1m cube near EE)", device)


if __name__ == "__main__":
    main()
