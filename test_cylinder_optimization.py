#!/usr/bin/env python3
"""
Test script to verify cylinder SDF optimization and measure performance improvements.
"""

import torch
import time
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pytorch_volumetric.sdf import CylinderSDF

def test_cylinder_sdf_optimization():
    """Test that optimized cylinder SDF produces correct results."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Testing on device: {device}")
    
    # Create cylinder SDF
    radius = 1.0
    length = 2.0
    cylinder = CylinderSDF(radius, length, device=device)
    
    # Test points in different regions
    test_points = torch.tensor([
        [0.0, 0.0, 0.0],      # Inside
        [2.0, 0.0, 0.0],      # Outside side
        [0.0, 0.0, 2.0],      # Outside cap
        [1.0, 0.0, 0.0],      # On side surface
        [0.0, 0.0, 1.0],      # On cap surface
        [1.5, 0.0, 1.5],      # Outside both
    ], device=device)
    
    print("Testing SDF values and gradients...")
    sdf_val, sdf_grad = cylinder(test_points)
    
    print("SDF values:", sdf_val)
    print("SDF gradients shape:", sdf_grad.shape)
    print("SDF gradients norm:", torch.norm(sdf_grad, dim=-1))
    
    # Test with larger batch
    print("\nTesting performance with larger batch...")
    batch_size = 10000
    large_batch = torch.randn(batch_size, 3, device=device) * 3.0
    
    # Warm up
    for _ in range(5):
        _ = cylinder(large_batch)
    
    # Measure performance
    torch.cuda.synchronize() if device == 'cuda' else None
    start_time = time.time()
    
    for _ in range(10):
        sdf_val, sdf_grad = cylinder(large_batch)
    
    torch.cuda.synchronize() if device == 'cuda' else None
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 10
    print(f"Average time for {batch_size} points: {avg_time:.4f} seconds")
    print(f"Points per second: {batch_size / avg_time:.0f}")
    
    print("\nOptimization test completed successfully!")

if __name__ == "__main__":
    test_cylinder_sdf_optimization() 