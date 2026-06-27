"""
STEP 1A: Generate Checkerboard Pattern for Camera Calibration
==============================================================
WHY: To calibrate a camera, we need a pattern with known geometry.
     A checkerboard has precise corners that OpenCV can detect automatically.

RUN:  python step1_camera_calibration/generate_checkerboard.py

THEN: Print the output image on A4 paper at 100% scale.
      Tape it to a flat rigid surface like cardboard.
"""

import numpy as np
import os

def generate_checkerboard(rows=9, cols=6, square_size_mm=25, dpi=300):
    """
    Generate a checkerboard pattern image.
    
    Parameters:
        rows: Inner corner count vertically (actual squares = rows+1)
        cols: Inner corner count horizontally (actual squares = cols+1)
        square_size_mm: Physical size of each square in millimeters
        dpi: Print resolution
    """
    try:
        import cv2
    except ImportError:
        print("ERROR: OpenCV not installed. Run: pip install opencv-python")
        return
    
    mm_per_inch = 25.4
    pixels_per_mm = dpi / mm_per_inch
    
    total_rows = rows + 1  # 10 squares tall
    total_cols = cols + 1  # 7 squares wide
    square_px = int(square_size_mm * pixels_per_mm)
    border_px = int(20 * pixels_per_mm)  # 20mm border
    
    img_h = total_rows * square_px + 2 * border_px
    img_w = total_cols * square_px + 2 * border_px
    
    # Start with white image
    img = np.ones((img_h, img_w), dtype=np.uint8) * 255
    
    # Draw black squares
    for r in range(total_rows):
        for c in range(total_cols):
            if (r + c) % 2 == 0:
                y = border_px + r * square_px
                x = border_px + c * square_px
                img[y:y+square_px, x:x+square_px] = 0
    
    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "checkerboard.png")
    cv2.imwrite(path, img)
    
    print("=" * 55)
    print(" CHECKERBOARD GENERATED")
    print("=" * 55)
    print(f" File: {path}")
    print(f" Pattern: {cols}x{rows} inner corners")
    print(f" Square size: {square_size_mm}mm")
    print()
    print(" NEXT STEPS:")
    print("  1. Print this image on A4 paper (100% scale)")
    print("  2. Tape to flat cardboard")
    print("  3. Run: python step1_camera_calibration/capture_checkerboard.py")
    print("=" * 55)

if __name__ == "__main__":
    generate_checkerboard()
