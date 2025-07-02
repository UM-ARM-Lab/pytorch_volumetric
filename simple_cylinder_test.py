#!/usr/bin/env python3
"""
Simple test for the optimized and original cylinder projection methods, with speed comparison.
"""

import torch
import time

class SimpleCylinderSDF:
    def __init__(self, radius, length, device='cpu'):
        self.r = radius
        self.l = length / 2
        self.device = device

    def original_project_to_cylinder(self, points_in_object_frame):
        p_xy = points_in_object_frame[..., :2]
        p_z = points_in_object_frame[..., 2]
        octant = torch.sign(points_in_object_frame)
        p_z_abs = torch.abs(p_z)
        new_z = torch.clamp(p_z_abs, min=None, max=self.l)
        d_to_xy = torch.linalg.norm(p_xy, dim=-1)
        d_to_disc = p_z_abs - self.l
        new_xy = p_xy / d_to_xy.unsqueeze(-1) * self.r
        d_to_xy_ = d_to_xy.unsqueeze(-1)
        d_to_disc_ = d_to_disc.unsqueeze(-1)
        new_xy = torch.where(d_to_xy_ < self.r,
                             torch.where(d_to_disc_ > 0, p_xy,
                                         torch.where(self.r - d_to_xy_ > torch.abs(d_to_disc_), p_xy, new_xy)), new_xy)
        in_flag = torch.where(d_to_xy_ > self.r, 1, -1)
        in_flag = torch.where(d_to_disc_ > 0, 1, in_flag)
        d_to_xy = d_to_xy_.squeeze(-1)
        d_to_disc = d_to_disc_.squeeze(-1)
        new_z = torch.where(d_to_xy < self.r,
                            torch.where(d_to_disc > 0, self.l,
                                        torch.where(self.r - d_to_xy > torch.abs(d_to_disc), self.l, new_z)), new_z)
        new_z = new_z * octant[..., 2]
        new_points = torch.cat((new_xy, new_z.unsqueeze(-1)), dim=-1)
        grad = (new_points - points_in_object_frame)
        sdf_val = in_flag * torch.linalg.norm(grad, dim=-1, keepdim=True)
        z_vector = torch.tensor([0., 0., 1.], device=self.device).expand_as(points_in_object_frame)
        xy_vector = torch.cat((p_xy / d_to_xy.unsqueeze(-1), torch.zeros(*p_xy.shape[:-1], 1, device=self.device)), dim=-1)
        z_vector = z_vector * octant[..., 2, None]
        grad = torch.where(sdf_val.abs() < 1e-6,
                           torch.where((d_to_disc.unsqueeze(-1)).abs() < 1e-6, z_vector, xy_vector), -grad / sdf_val)
        return sdf_val.squeeze(-1), grad

    def optimized_project_to_cylinder(self, points_in_object_frame):
        p_xy = points_in_object_frame[..., :2]
        p_z = points_in_object_frame[..., 2]
        p_z_abs = torch.abs(p_z)
        d_to_xy = torch.linalg.norm(p_xy, dim=-1)
        on_axis = d_to_xy < 1e-8
        d_to_xy = torch.where(on_axis, torch.ones_like(d_to_xy), d_to_xy)
        side_region = (p_z_abs <= self.l) & (d_to_xy > self.r) & ~on_axis
        cap_region = p_z_abs > self.l
        inside_region = (p_z_abs <= self.l) & (d_to_xy <= self.r)
        new_points = torch.zeros_like(points_in_object_frame)
        in_flag = torch.ones_like(d_to_xy)
        if side_region.any():
            xy_dir = p_xy[side_region] / d_to_xy[side_region].unsqueeze(-1)
            new_points[side_region, :2] = xy_dir * self.r
            new_points[side_region, 2] = p_z[side_region]
            in_flag[side_region] = -1
        if cap_region.any():
            xy_dir = p_xy[cap_region] / d_to_xy[cap_region].unsqueeze(-1)
            new_points[cap_region, :2] = xy_dir * self.r
            new_points[cap_region, 2] = torch.sign(p_z[cap_region]) * self.l
            in_flag[cap_region] = -1
        if inside_region.any():
            new_points[inside_region] = points_in_object_frame[inside_region]
            in_flag[inside_region] = 1
        if on_axis.any():
            new_points[on_axis, :2] = 0.0
            new_points[on_axis, 2] = torch.clamp(p_z[on_axis], -self.l, self.l)
            in_flag[on_axis] = torch.where(
                p_z_abs[on_axis] <= self.l,
                torch.tensor(1.0, device=in_flag.device, dtype=in_flag.dtype),
                torch.tensor(-1.0, device=in_flag.device, dtype=in_flag.dtype)
            )
        grad = new_points - points_in_object_frame
        sdf_val = in_flag * torch.linalg.norm(grad, dim=-1)
        near_surface = torch.abs(sdf_val) < 1e-6
        if near_surface.any():
            z_vector = torch.zeros_like(points_in_object_frame)
            z_vector[..., 2] = torch.sign(p_z)
            xy_vector = torch.zeros_like(points_in_object_frame)
            xy_dir_near = p_xy[near_surface] / d_to_xy[near_surface].unsqueeze(-1)
            xy_vector[near_surface, :2] = xy_dir_near
            on_cap = near_surface & (p_z_abs > self.l - 1e-6)
            grad = torch.where(on_cap.unsqueeze(-1), z_vector, xy_vector)
        else:
            grad_norm = torch.linalg.norm(grad, dim=-1, keepdim=True)
            grad = torch.where(grad_norm > 1e-8, grad / grad_norm, grad)
        return sdf_val, grad

    def hybrid_project_to_cylinder(self, points_in_object_frame):
        # Extract coordinates and compute key values once
        p_xy = points_in_object_frame[..., :2]
        p_z = points_in_object_frame[..., 2]
        p_z_abs = torch.abs(p_z)
        d_to_xy = torch.linalg.norm(p_xy, dim=-1)
        d_to_disc = p_z_abs - self.l

        # Handle points on z-axis without branching
        on_axis_mask = d_to_xy < 1e-8
        d_to_xy = torch.where(on_axis_mask, torch.ones_like(d_to_xy), d_to_xy)
        
        # Compute normalized xy direction for all points at once
        xy_dir = p_xy / d_to_xy.unsqueeze(-1)
        
        # Project points to cylinder surface in one go
        new_xy = xy_dir * self.r
        new_z = torch.clamp(p_z, -self.l, self.l)
        
        # Combine into new points
        new_points = torch.empty_like(points_in_object_frame)
        new_points[..., :2] = torch.where(
            (d_to_xy <= self.r).unsqueeze(-1) & (p_z_abs <= self.l).unsqueeze(-1),
            p_xy,  # Inside points keep original xy
            new_xy  # Outside points get projected
        )
        new_points[..., 2] = new_z
        
        # Fix points exactly on z-axis
        new_points[on_axis_mask] = torch.zeros_like(points_in_object_frame[on_axis_mask])
        new_points[on_axis_mask, 2] = new_z[on_axis_mask]
        
        # Compute signed distance and gradient
        grad = new_points - points_in_object_frame
        grad_norm = torch.linalg.norm(grad, dim=-1, keepdim=True)
        
        # Determine inside/outside without branches
        in_flag = torch.where(
            (d_to_xy <= self.r) & (p_z_abs <= self.l),
            torch.ones_like(d_to_xy),
            -torch.ones_like(d_to_xy)
        )
        sdf_val = in_flag * grad_norm.squeeze(-1)
        
        # Handle surface normals for near-surface points
        near_surface = torch.abs(sdf_val) < 1e-6
        if near_surface.any():
            # Pre-compute surface normals for all points
            surface_normals = torch.zeros_like(points_in_object_frame)
            # For points near caps
            on_cap = near_surface & (p_z_abs > self.l - 1e-6)
            surface_normals[on_cap, 2] = torch.sign(p_z[on_cap])
            # For points near sides
            on_side = near_surface & ~on_cap & ~on_axis_mask
            surface_normals[on_side, :2] = xy_dir[on_side]
            
            grad = torch.where(
                near_surface.unsqueeze(-1),
                surface_normals,
                grad / (grad_norm + 1e-8)  # Add epsilon to avoid division by zero
            )
        else:
            grad = grad / (grad_norm + 1e-8)
            
        return sdf_val, grad

def test_optimization():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Testing on device: {device}")
    
    cylinder = SimpleCylinderSDF(1.0, 2.0, device=device)
    
    # Test correctness first
    test_points = torch.tensor([
        [0.0, 0.0, 0.0],      # Inside
        [2.0, 0.0, 0.0],      # Outside side
        [0.0, 0.0, 2.0],      # Outside cap
        [1.0, 0.0, 0.0],      # On side surface
        [0.0, 0.0, 1.0],      # On cap surface
        [1.5, 0.0, 1.5],      # Outside both
        [0.0, 0.0, 0.5],      # On z-axis inside
        [0.0, 0.0, 1.5],      # On z-axis outside
    ], device=device)
    
    print("\nTesting correctness...")
    orig_val, orig_grad = cylinder.original_project_to_cylinder(test_points)
    opt_val, opt_grad = cylinder.optimized_project_to_cylinder(test_points)
    hyb_val, hyb_grad = cylinder.hybrid_project_to_cylinder(test_points)
    
    print("Original SDF values:", orig_val)
    print("Optimized SDF values:", opt_val)
    print("Hybrid SDF values:", hyb_val)
    print("\nMax difference between implementations:")
    print("Original vs Optimized:", torch.max(torch.abs(orig_val - opt_val)).item())
    print("Original vs Hybrid:", torch.max(torch.abs(orig_val - hyb_val)).item())
    print("Optimized vs Hybrid:", torch.max(torch.abs(opt_val - hyb_val)).item())
    
    # Benchmark with larger batch
    print("\nBenchmarking performance...")
    batch_size = 1000000  # Increased batch size for better timing
    large_batch = torch.randn(batch_size, 3, device=device) * 3.0
    
    # Warm up
    for _ in range(5):
        _ = cylinder.original_project_to_cylinder(large_batch)
        _ = cylinder.optimized_project_to_cylinder(large_batch)
        _ = cylinder.hybrid_project_to_cylinder(large_batch)
    
    num_trials = 10
    
    # Time original
    torch.cuda.synchronize() if device == 'cuda' else None
    start_time = time.time()
    for _ in range(num_trials):
        sdf_val, sdf_grad = cylinder.original_project_to_cylinder(large_batch)
    torch.cuda.synchronize() if device == 'cuda' else None
    orig_avg = (time.time() - start_time) / num_trials
    
    # Time optimized
    torch.cuda.synchronize() if device == 'cuda' else None
    start_time = time.time()
    for _ in range(num_trials):
        sdf_val, sdf_grad = cylinder.optimized_project_to_cylinder(large_batch)
    torch.cuda.synchronize() if device == 'cuda' else None
    opt_avg = (time.time() - start_time) / num_trials
    
    # Time hybrid
    torch.cuda.synchronize() if device == 'cuda' else None
    start_time = time.time()
    for _ in range(num_trials):
        sdf_val, sdf_grad = cylinder.hybrid_project_to_cylinder(large_batch)
    torch.cuda.synchronize() if device == 'cuda' else None
    hyb_avg = (time.time() - start_time) / num_trials
    
    print(f"\nResults for {batch_size:,} points:")
    print(f"Original: {orig_avg*1000:.3f} ms ({batch_size/orig_avg:,.0f} pts/sec)")
    print(f"Optimized: {opt_avg*1000:.3f} ms ({batch_size/opt_avg:,.0f} pts/sec)")
    print(f"Hybrid: {hyb_avg*1000:.3f} ms ({batch_size/hyb_avg:,.0f} pts/sec)")
    print(f"\nSpeedup vs original:")
    print(f"Optimized: {orig_avg/opt_avg:.2f}x")
    print(f"Hybrid: {orig_avg/hyb_avg:.2f}x")
    
    print("\nOptimization test completed successfully!")

if __name__ == "__main__":
    test_optimization() 