"""
Camera Utilities — 3D Projection Math
=======================================
Core formulas for converting between 2D pixels and 3D world coordinates.

THE KEY EQUATION (pixel → 3D world):
    X = (u - cx) × Z / fx
    Y = (v - cy) × Z / fy
    Z = depth (meters, predicted by our model)

Where:
    (u, v)      = pixel coordinates in the image
    (X, Y, Z)   = real 3D position in meters
    (fx, fy)     = focal lengths in pixels (from calibration)
    (cx, cy)     = principal point (from calibration)
"""

import numpy as np


def depth_to_3d_points(depth_map, rgb_image, fx, fy, cx, cy, max_depth=None):
    """
    Convert a depth map to a colored 3D point cloud.
    
    Each pixel (u, v) with depth Z becomes a 3D point (X, Y, Z).
    """
    h, w = depth_map.shape
    
    u = np.arange(w)
    v = np.arange(h)
    u, v = np.meshgrid(u, v)
    
    valid = depth_map > 0.01
    if max_depth:
        valid &= depth_map < max_depth
    
    Z = depth_map[valid]
    X = (u[valid] - cx) * Z / fx
    Y = (v[valid] - cy) * Z / fy
    
    points = np.stack([X, Y, Z], axis=-1)
    
    if rgb_image is not None:
        import cv2
        if rgb_image.shape[:2] != (h, w):
            rgb_image = cv2.resize(rgb_image, (w, h))
        colors = rgb_image[valid]
    else:
        colors = np.zeros((len(points), 3), dtype=np.uint8)
    
    return points, colors


def save_pointcloud_ply(points, colors, filepath, max_points=500000):
    """
    Save 3D point cloud as PLY file.
    Open with: MeshLab, CloudCompare, open3d, Blender
    """
    if len(points) > max_points:
        idx = np.random.choice(len(points), max_points, replace=False)
        points, colors = points[idx], colors[idx]
    
    with open(filepath, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(len(points)):
            f.write(f"{points[i,0]:.4f} {points[i,1]:.4f} {points[i,2]:.4f} "
                   f"{int(colors[i,0])} {int(colors[i,1])} {int(colors[i,2])}\n")
    
    print(f"  Point cloud: {filepath} ({len(points)} points)")


def pixel_to_world(u, v, depth, fx, fy, cx, cy):
    """Convert single pixel to 3D world coordinate."""
    X = (u - cx) * depth / fx
    Y = (v - cy) * depth / fy
    return np.array([X, Y, depth])


def world_to_pixel(X, Y, Z, fx, fy, cx, cy):
    """Project 3D point back to pixel coordinates."""
    u = fx * X / Z + cx
    v = fy * Y / Z + cy
    return np.array([u, v])


def estimate_focal_from_known_object(real_size_m, pixel_size, depth_m):
    """
    Estimate focal length from a known-size object.
    f = (pixel_size × depth) / real_size
    """
    return (pixel_size * depth_m) / real_size_m


def intrinsic_matrix(fx, fy, cx, cy):
    """Build 3×3 camera intrinsic matrix K."""
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def rescale_intrinsics(fx, fy, cx, cy, orig_w, orig_h, new_w, new_h):
    """Rescale intrinsics when image is resized. CRITICAL for correctness."""
    sx, sy = new_w / orig_w, new_h / orig_h
    return fx * sx, fy * sy, cx * sx, cy * sy
