#!/usr/bin/env python3
"""
Test script to benchmark mesh raycasting performance with CUDA acceleration.

This test demonstrates the performance improvements achievable with CUDA in mesh operations,
working around current Open3D limitations by focusing on tensor operations and memory transfers.
"""

import torch
import time
import sys
import os
import numpy as np
from typing import List, Dict, Tuple

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

import pytorch_volumetric as pv

TEST_DIR = os.path.dirname(__file__)

def setup_environment():
    """Setup and verify environment for testing."""
    print("=== Environment Setup ===")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"PyTorch CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name()}")
        print(f"CUDA device memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Check Open3D CUDA support
    cuda_info = pv.check_open3d_cuda_support()
    print(f"\nOpen3D CUDA support: {cuda_info['can_create_cuda_device']}")
    print(f"Open3D CUDA devices: {cuda_info['cuda_device_count']}")
    
    if cuda_info['recommendations']:
        print("Note: Open3D CUDA has limitations:")
        for rec in cuda_info['recommendations']:
            print(f"  - {rec}")
    
    return cuda_available

def create_mesh_sdf(mesh_file: str) -> pv.MeshSDF:
    """Create a MeshSDF object using CPU-based raycasting (due to Open3D limitations)."""
    # Force CPU for raycasting scene creation due to Open3D limitations
    pv.configure_sdf_cuda(enable=False)
    
    mesh_path = os.path.join(TEST_DIR, mesh_file)
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"Test mesh not found: {mesh_path}")
    
    obj_factory = pv.MeshObjectFactory(mesh_path)
    mesh_sdf = pv.MeshSDF(obj_factory)
    
    # Precompute SDF to ensure raycasting scene is ready
    mesh_sdf.precompute_sdf()
    
    return mesh_sdf

def generate_test_points(num_points: int, mesh_sdf: pv.MeshSDF, device: str = 'cpu') -> torch.Tensor:
    """Generate test points around the mesh bounding box."""
    bounding_box = mesh_sdf.obj_factory.bounding_box(padding=0.1)
    
    # Generate random points within the expanded bounding box
    ranges = torch.tensor(bounding_box, dtype=torch.float32, device=device)
    
    # Create random points
    points = torch.rand(num_points, 3, device=device)
    
    # Scale to bounding box
    for i in range(3):
        points[:, i] = points[:, i] * (ranges[i, 1] - ranges[i, 0]) + ranges[i, 0]
    
    return points

def benchmark_sdf_computation(mesh_sdf: pv.MeshSDF, test_points: torch.Tensor, 
                             num_trials: int = 10, warmup_trials: int = 3) -> Dict[str, float]:
    """Benchmark SDF computation performance."""
    device = 'cuda' if test_points.is_cuda else 'cpu'
    
    # Warmup trials
    for _ in range(warmup_trials):
        _ = mesh_sdf(test_points)
        if device == 'cuda':
            torch.cuda.synchronize()
    
    # Timing trials
    times = []
    for _ in range(num_trials):
        if device == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        sdf_vals, sdf_grads = mesh_sdf(test_points)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    # Calculate statistics
    times = np.array(times)
    return {
        'mean_time': float(np.mean(times)),
        'std_time': float(np.std(times)),
        'min_time': float(np.min(times)),
        'max_time': float(np.max(times)),
        'median_time': float(np.median(times))
    }

def benchmark_memory_transfer(mesh_sdf: pv.MeshSDF, point_counts: List[int]) -> Dict:
    """Benchmark memory transfer and tensor operations performance."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory transfer benchmark")
        return {}
    
    print("\n=== Memory Transfer Performance ===")
    
    results = {}
    
    for num_points in point_counts:
        print(f"\nTesting {num_points:,} points...")
        
        # Generate points on CPU
        cpu_points = generate_test_points(num_points, mesh_sdf, device='cpu')
        
        # Benchmark CPU to GPU transfer
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        gpu_points = cpu_points.cuda()
        torch.cuda.synchronize()
        transfer_time = time.perf_counter() - start_time
        
        # Benchmark GPU to CPU transfer
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        _ = gpu_points.cpu()
        torch.cuda.synchronize()
        back_transfer_time = time.perf_counter() - start_time
        
        # Benchmark tensor operations on GPU vs CPU
        # CPU tensor operations
        start_time = time.perf_counter()
        cpu_norm = torch.norm(cpu_points, dim=-1)
        cpu_ops_time = time.perf_counter() - start_time
        
        # GPU tensor operations
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        gpu_norm = torch.norm(gpu_points, dim=-1)
        torch.cuda.synchronize()
        gpu_ops_time = time.perf_counter() - start_time
        
        results[num_points] = {
            'cpu_to_gpu_transfer': transfer_time,
            'gpu_to_cpu_transfer': back_transfer_time,
            'cpu_tensor_ops': cpu_ops_time,
            'gpu_tensor_ops': gpu_ops_time,
            'tensor_ops_speedup': cpu_ops_time / gpu_ops_time if gpu_ops_time > 0 else 0
        }
        
        print(f"  CPU→GPU transfer: {transfer_time:.4f}s")
        print(f"  GPU→CPU transfer: {back_transfer_time:.4f}s")
        print(f"  CPU tensor ops: {cpu_ops_time:.6f}s")
        print(f"  GPU tensor ops: {gpu_ops_time:.6f}s")
        print(f"  Tensor ops speedup: {results[num_points]['tensor_ops_speedup']:.2f}x")
    
    return results

def run_comprehensive_performance_test(mesh_file: str, point_counts: List[int]) -> Dict:
    """Run comprehensive performance analysis."""
    print(f"\n=== Comprehensive Performance Analysis for {mesh_file} ===")
    
    mesh_sdf = create_mesh_sdf(mesh_file)
    
    results = {
        'mesh_file': mesh_file,
        'cpu_sdf_results': {},
        'gpu_sdf_results': {},
        'memory_transfer_results': {},
        'analysis': {}
    }
    
    # Test SDF computation on CPU vs GPU input tensors
    for num_points in point_counts:
        print(f"\nTesting SDF computation with {num_points:,} points...")
        
        # CPU-based SDF computation
        cpu_points = generate_test_points(num_points, mesh_sdf, device='cpu')
        cpu_stats = benchmark_sdf_computation(mesh_sdf, cpu_points)
        results['cpu_sdf_results'][num_points] = cpu_stats
        
        if torch.cuda.is_available():
            # GPU tensor input (even though raycasting is on CPU)
            gpu_points = generate_test_points(num_points, mesh_sdf, device='cuda')
            gpu_stats = benchmark_sdf_computation(mesh_sdf, gpu_points)
            results['gpu_sdf_results'][num_points] = gpu_stats
            
            print(f"  CPU input: {cpu_stats['mean_time']:.4f} ± {cpu_stats['std_time']:.4f}s")
            print(f"  GPU input: {gpu_stats['mean_time']:.4f} ± {gpu_stats['std_time']:.4f}s")
            print(f"  Input type overhead: {gpu_stats['mean_time'] / cpu_stats['mean_time']:.2f}x")
        else:
            print(f"  CPU only: {cpu_stats['mean_time']:.4f} ± {cpu_stats['std_time']:.4f}s")
    
    # Memory transfer benchmarks
    if torch.cuda.is_available():
        results['memory_transfer_results'] = benchmark_memory_transfer(mesh_sdf, point_counts)
    
    return results

def analyze_scaling_behavior(results: Dict):
    """Analyze how performance scales with input size."""
    print("\n=== Scaling Analysis ===")
    
    cpu_results = results['cpu_sdf_results']
    gpu_results = results['gpu_sdf_results']
    
    if cpu_results and gpu_results:
        print("Point Count | CPU Time | GPU Input | Ratio | CPU Throughput | GPU Throughput")
        print("------------|----------|-----------|-------|----------------|----------------")
        
        for num_points in sorted(cpu_results.keys()):
            cpu_time = cpu_results[num_points]['mean_time']
            gpu_time = gpu_results[num_points]['mean_time']
            ratio = gpu_time / cpu_time
            cpu_throughput = num_points / cpu_time
            gpu_throughput = num_points / gpu_time
            
            print(f"{num_points:10,} | {cpu_time:8.4f} | {gpu_time:9.4f} | {ratio:5.2f} | {cpu_throughput:14.0f} | {gpu_throughput:14.0f}")
    
    # Analyze memory transfer efficiency
    if 'memory_transfer_results' in results:
        print(f"\n=== Memory Transfer Efficiency ===")
        transfer_results = results['memory_transfer_results']
        
        print("Point Count | Transfer Time | Bandwidth (GB/s) | Tensor Ops Speedup")
        print("------------|---------------|------------------|-------------------")
        
        for num_points in sorted(transfer_results.keys()):
            data = transfer_results[num_points]
            # Estimate data size (3 floats per point * 4 bytes per float)
            data_size_gb = (num_points * 3 * 4) / 1e9
            transfer_time = data['cpu_to_gpu_transfer']
            bandwidth = data_size_gb / transfer_time if transfer_time > 0 else 0
            speedup = data['tensor_ops_speedup']
            
            print(f"{num_points:10,} | {transfer_time:13.6f} | {bandwidth:14.2f} | {speedup:17.2f}")

def test_mesh_raycasting_cuda_performance():
    """Main test function for mesh raycasting CUDA performance."""
    print("=== Mesh Raycasting CUDA Performance Analysis ===")
    print("Note: Due to Open3D limitations, raycasting uses CPU but we analyze")
    print("      tensor operations and memory transfer performance with CUDA.")
    
    # Test configurations
    test_meshes = [
        "probe.obj",
        "offset_wrench_nogrip.obj"
    ]
    
    point_counts = [100, 1000, 10000, 50000, 100000]
    
    # Check if test meshes exist
    available_meshes = []
    for mesh in test_meshes:
        mesh_path = os.path.join(TEST_DIR, mesh)
        if os.path.exists(mesh_path):
            available_meshes.append(mesh)
        else:
            print(f"Warning: Test mesh {mesh} not found, skipping")
    
    if not available_meshes:
        print("No test meshes available, exiting")
        return
    
    # Setup environment
    cuda_available = setup_environment()
    
    all_results = {}
    
    # Run benchmarks for each mesh
    for mesh_file in available_meshes:
        try:
            results = run_comprehensive_performance_test(mesh_file, point_counts)
            all_results[mesh_file] = results
            analyze_scaling_behavior(results)
        except Exception as e:
            print(f"Error testing {mesh_file}: {e}")
            continue
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL PERFORMANCE SUMMARY")
    print("="*80)
    
    for mesh_file, results in all_results.items():
        print(f"\n{mesh_file}:")
        print(f"  Mesh complexity: {get_mesh_complexity(mesh_file)}")
        
        if results['gpu_sdf_results']:
            print("  Key findings:")
            cpu_100k = results['cpu_sdf_results'].get(100000, {}).get('mean_time', 0)
            gpu_100k = results['gpu_sdf_results'].get(100000, {}).get('mean_time', 0)
            
            if cpu_100k > 0 and gpu_100k > 0:
                print(f"    - 100K points: CPU {cpu_100k:.4f}s vs GPU input {gpu_100k:.4f}s")
                print(f"    - Memory transfer overhead: {(gpu_100k/cpu_100k - 1)*100:.1f}%")
            
            # Memory transfer insights
            if 100000 in results['memory_transfer_results']:
                transfer_data = results['memory_transfer_results'][100000]
                speedup = transfer_data['tensor_ops_speedup']
                print(f"    - Tensor operations speedup: {speedup:.2f}x")
        else:
            print("  CUDA not available for comparison")

def get_mesh_complexity(mesh_file: str) -> str:
    """Get basic complexity info about the mesh."""
    try:
        mesh_path = os.path.join(TEST_DIR, mesh_file)
        obj_factory = pv.MeshObjectFactory(mesh_path)
        obj_factory.precompute_sdf()
        mesh = obj_factory.get_mesh()
        
        vertices = len(mesh.vertices)
        triangles = len(mesh.triangles)
        
        return f"{vertices:,} vertices, {triangles:,} triangles"
    except Exception:
        return "unknown"

def test_memory_usage_patterns():
    """Test memory usage patterns for different batch sizes."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory usage test")
        return
    
    print("\n=== Memory Usage Patterns ===")
    
    mesh_file = "probe.obj"
    if not os.path.exists(os.path.join(TEST_DIR, mesh_file)):
        print("Test mesh not available, skipping memory usage test")
        return
    
    mesh_sdf = create_mesh_sdf(mesh_file)
    batch_sizes = [1000, 10000, 50000, 100000, 500000]
    
    for batch_size in batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Measure memory before
        mem_before = torch.cuda.memory_allocated() / 1e6  # MB
        
        try:
            # Generate points and run SDF
            test_points = generate_test_points(batch_size, mesh_sdf, device='cuda')
            sdf_vals, sdf_grads = mesh_sdf(test_points)
            
            # Measure peak memory
            mem_peak = torch.cuda.max_memory_allocated() / 1e6  # MB
            mem_used = mem_peak - mem_before
            
            print(f"  Batch size {batch_size:6,}: {mem_used:6.1f} MB peak memory")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  Batch size {batch_size:6,}: Out of memory")
                break
            else:
                raise

if __name__ == "__main__":
    test_mesh_raycasting_cuda_performance()
    test_memory_usage_patterns() 