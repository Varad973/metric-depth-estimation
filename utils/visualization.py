"""
Visualization Utilities
========================
Colorize depth maps, create comparison images, and generate overlays.
Used by Step 4 (inference) and Step 5 (validation).
"""

import numpy as np
import cv2


def colorize_depth(depth_map, min_depth=None, max_depth=None,
                   colormap=cv2.COLORMAP_MAGMA, invalid_color=(0, 0, 0)):
    """
    Convert depth map (meters) → colorized RGB image for human viewing.
    
    Args:
        depth_map:  (H, W) float array, values in meters
        min_depth:  clip floor (auto = 2nd percentile)
        max_depth:  clip ceil  (auto = 98th percentile)
        colormap:   cv2 colormap (MAGMA, TURBO, JET, INFERNO)
    
    Returns:
        (H, W, 3) uint8 RGB image
    """
    valid = depth_map > 0
    if valid.sum() == 0:
        return np.zeros((*depth_map.shape, 3), dtype=np.uint8)
    
    if min_depth is None:
        min_depth = float(np.percentile(depth_map[valid], 2))
    if max_depth is None:
        max_depth = float(np.percentile(depth_map[valid], 98))
    
    # Normalize to 0-255
    norm = (depth_map - min_depth) / (max_depth - min_depth + 1e-8)
    norm = np.clip(norm, 0, 1)
    u8 = (norm * 255).astype(np.uint8)
    
    # Apply colormap (OpenCV returns BGR)
    colored = cv2.applyColorMap(u8, colormap)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    
    # Mark invalid pixels
    colored[~valid] = invalid_color
    
    # Add depth scale text
    h, w = depth_map.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(colored, f"{min_depth:.1f}m", (5, h - 10),
                font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(colored, f"{max_depth:.1f}m", (w - 60, h - 10),
                font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    
    return colored


def create_side_by_side(rgb, depth_colored, labels=True):
    """
    Create side-by-side: [RGB Input | Predicted Depth]
    
    Args:
        rgb:            (H, W, 3) uint8 RGB
        depth_colored:  (H, W, 3) uint8 RGB (from colorize_depth)
    
    Returns:
        (H+label_h, W*2+gap, 3) uint8 RGB
    """
    h, w = rgb.shape[:2]
    dh, dw = depth_colored.shape[:2]
    
    # Match sizes
    if (h, w) != (dh, dw):
        depth_colored = cv2.resize(depth_colored, (w, h))
    
    gap = 4
    label_h = 30 if labels else 0
    
    canvas = np.ones((h + label_h, w * 2 + gap, 3), dtype=np.uint8) * 40
    canvas[label_h:, :w] = rgb
    canvas[label_h:, w + gap:] = depth_colored[:, :w]
    
    if labels:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, "RGB Input", (w // 2 - 45, 20),
                    font, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Predicted Depth (meters)", (w + gap + w // 2 - 100, 20),
                    font, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    
    return canvas


def create_comparison_grid(rgb, pred_depth, gt_depth=None):
    """
    Create a multi-panel comparison:
        [RGB | Prediction | Ground Truth | Error Map]
    
    Used during validation to visually inspect model quality.
    """
    h, w = rgb.shape[:2] if isinstance(rgb, np.ndarray) else (480, 640)
    
    # Handle torch tensors
    if hasattr(rgb, 'permute'):
        # Denormalize ImageNet
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        rgb_np = rgb.permute(1, 2, 0).numpy()
        rgb_np = ((rgb_np * std + mean) * 255).clip(0, 255).astype(np.uint8)
    else:
        rgb_np = rgb
    
    if hasattr(pred_depth, 'squeeze'):
        pred_np = pred_depth.squeeze().numpy() if hasattr(pred_depth, 'numpy') else pred_depth.squeeze()
    else:
        pred_np = pred_depth
    
    h, w = rgb_np.shape[:2]
    pred_col = colorize_depth(pred_np)
    pred_col = cv2.resize(pred_col, (w, h))
    
    panels = [rgb_np, pred_col]
    titles = ["RGB", "Prediction"]
    
    if gt_depth is not None:
        if hasattr(gt_depth, 'squeeze'):
            gt_np = gt_depth.squeeze().numpy() if hasattr(gt_depth, 'numpy') else gt_depth.squeeze()
        else:
            gt_np = gt_depth
        
        gt_col = colorize_depth(gt_np)
        gt_col = cv2.resize(gt_col, (w, h))
        panels.append(gt_col)
        titles.append("Ground Truth")
        
        # Error map
        error = np.abs(pred_np - gt_np)
        err_col = colorize_depth(error, min_depth=0, max_depth=2.0,
                                  colormap=cv2.COLORMAP_HOT)
        err_col = cv2.resize(err_col, (w, h))
        panels.append(err_col)
        titles.append("Abs Error")
    
    # Build grid
    n = len(panels)
    gap = 4
    label_h = 28
    canvas = np.ones((h + label_h, w * n + gap * (n - 1), 3), dtype=np.uint8) * 30
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, (panel, title) in enumerate(zip(panels, titles)):
        x_off = i * (w + gap)
        canvas[label_h:, x_off:x_off + w] = panel
        tw = cv2.getTextSize(title, font, 0.5, 1)[0][0]
        cv2.putText(canvas, title, (x_off + w // 2 - tw // 2, 18),
                    font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    
    return canvas


def save_comparison(rgb_tensor, pred_tensor, gt_tensor, filepath):
    """Save a comparison grid image to disk. Called during training validation."""
    grid = create_comparison_grid(rgb_tensor, pred_tensor, gt_tensor)
    grid_bgr = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filepath, grid_bgr)


def depth_histogram(depth_map, bins=50, title="Depth Distribution"):
    """
    Create a histogram image of depth values.
    Useful for understanding the depth distribution of predictions.
    """
    valid = depth_map[depth_map > 0].flatten()
    if len(valid) == 0:
        return np.zeros((300, 400, 3), dtype=np.uint8)
    
    # Compute histogram
    hist, edges = np.histogram(valid, bins=bins)
    hist_norm = hist / hist.max()
    
    # Draw
    canvas_h, canvas_w = 300, 400
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
    
    bar_w = (canvas_w - 60) // bins
    max_bar_h = canvas_h - 60
    
    for i in range(bins):
        bar_h = int(hist_norm[i] * max_bar_h)
        x = 40 + i * bar_w
        y_top = canvas_h - 40 - bar_h
        
        # Color based on depth value
        depth_val = (edges[i] + edges[i + 1]) / 2
        norm_val = (depth_val - edges[0]) / (edges[-1] - edges[0] + 1e-8)
        color_u8 = int(norm_val * 255)
        bar_color = cv2.applyColorMap(
            np.array([[color_u8]], dtype=np.uint8), cv2.COLORMAP_MAGMA)[0, 0]
        bar_color = tuple(int(c) for c in bar_color)
        
        cv2.rectangle(canvas, (x, y_top), (x + bar_w - 1, canvas_h - 40),
                      bar_color, -1)
    
    # Axes labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, f"{edges[0]:.1f}m", (35, canvas_h - 15),
                font, 0.4, (0, 0, 0), 1)
    cv2.putText(canvas, f"{edges[-1]:.1f}m", (canvas_w - 55, canvas_h - 15),
                font, 0.4, (0, 0, 0), 1)
    cv2.putText(canvas, title, (canvas_w // 2 - 60, 20),
                font, 0.5, (0, 0, 0), 1)
    
    return canvas


def overlay_depth_on_rgb(rgb, depth_map, alpha=0.5, colormap=cv2.COLORMAP_MAGMA):
    """
    Blend colorized depth map onto RGB image as a transparent overlay.
    Great for visualizing which parts of the image are near vs far.
    """
    depth_col = colorize_depth(depth_map, colormap=colormap)
    
    if rgb.shape[:2] != depth_col.shape[:2]:
        depth_col = cv2.resize(depth_col, (rgb.shape[1], rgb.shape[0]))
    
    blended = cv2.addWeighted(rgb, 1 - alpha, depth_col, alpha, 0)
    return blended
