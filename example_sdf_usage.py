#!/usr/bin/env python3
"""
Example script demonstrating how to use the SDF with optional hessian computation.
Run with --no-hessian to disable hessian computation for better performance.
"""

import torch
import numpy as np
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pytorch_volumetric as pv

def main():
    print("=== PyTorch Volumetric SDF Example ===")
    
    # Check CUDA availability for Open3D
    print("\n1. Checking Open3D CUDA Support:")
    cuda_info = pv.check_open3d_cuda_support()
    print(f"Has CUDA module: {cuda_info['has_cuda_module']}")
    print(f"CUDA devices: {cuda_info['cuda_device_count']}")
    print(f"Can create CUDA device: {cuda_info['can_create_cuda_device']}")
    
    if cuda_info['recommendations']:
        print("Recommendations:")
        for rec in cuda_info['recommendations']:
            print(f"  - {rec}")
    
    # Configure CUDA if available
    if cuda_info['can_create_cuda_device']:
        print("\n2. Enabling CUDA acceleration...")
        pv.configure_sdf_cuda(enable=True)
    else:
        print("\n2. Using CPU-only mode...")
        pv.configure_sdf_cuda(enable=False)
    
    # Create a simple box SDF
    print("\n3. Creating Box SDF...")
    box_sdf = pv.BoxSDF([0.2, 0.3, 0.1], device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test with some query points
    print("\n4. Testing SDF queries...")
    query_points = torch.tensor([
        [0.0, 0.0, 0.0],  # inside
        [0.5, 0.0, 0.0],  # outside
        [0.1, 0.1, 0.05], # near surface
    ], dtype=torch.float32)
    
    if torch.cuda.is_available():
        query_points = query_points.cuda()
    
    sdf_values, sdf_gradients = box_sdf(query_points.unsqueeze(0))
    
    print("Query points:")
    print(query_points)
    print("SDF values:")
    print(sdf_values)
    print("SDF gradients:")
    print(sdf_gradients)
    
    # Performance comparison example (if CUDA available)
    if cuda_info['can_create_cuda_device']:
        print("\n5. Performance comparison (CPU vs CUDA)...")
        
        # Generate more points for timing
        n_points = 10000
        large_query = torch.randn(1, n_points, 3) * 0.5
        
        # Time CPU version
        pv.configure_sdf_cuda(enable=False)
        box_sdf_cpu = pv.BoxSDF([0.2, 0.3, 0.1], device='cpu')
        
        import time
        start = time.time()
        _, _ = box_sdf_cpu(large_query)
        cpu_time = time.time() - start
        
        # Time CUDA version (if available)
        if torch.cuda.is_available():
            pv.configure_sdf_cuda(enable=True)
            box_sdf_cuda = pv.BoxSDF([0.2, 0.3, 0.1], device='cuda')
            large_query_cuda = large_query.cuda()
            
            # Warm up
            _, _ = box_sdf_cuda(large_query_cuda)
            
            start = time.time()
            _, _ = box_sdf_cuda(large_query_cuda)
            cuda_time = time.time() - start
            
            print(f"CPU time: {cpu_time:.4f}s")
            print(f"CUDA time: {cuda_time:.4f}s")
            print(f"Speedup: {cpu_time/cuda_time:.2f}x")
        else:
            print("PyTorch CUDA not available for comparison")

if __name__ == "__main__":
    main() 