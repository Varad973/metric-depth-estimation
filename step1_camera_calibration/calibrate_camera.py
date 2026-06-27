"""
STEP 1C: Camera Calibration — Find Intrinsic Parameters
========================================================
Computes fx, fy, cx, cy from checkerboard photos OR uses a built-in database.

RUN:
    # From checkerboard photos:
    python step1_camera_calibration/calibrate_camera.py --image_dir data/calibration_images

    # From built-in database (easiest):
    python step1_camera_calibration/calibrate_camera.py --use_database --camera iphone_13

    # List all cameras:
    python step1_camera_calibration/calibrate_camera.py --list_cameras

OUTPUT: camera_params.json (used by all later steps)

WHAT ARE INTRINSICS?
    fx, fy = focal length in pixels (how "zoomed in" the camera is)
    cx, cy = principal point (where the optical axis hits the sensor, usually near center)
    Together they form the camera matrix K:
        K = | fx  0  cx |
            | 0  fy  cy |
            | 0   0   1 |
"""

import cv2
import numpy as np
import json
import os
import sys
import glob
import argparse

# ============================================================
# BUILT-IN DATABASE OF PHONE CAMERA INTRINSICS
# These are approximate. For best results, calibrate yourself.
# ============================================================
CAMERA_DATABASE = {
    "iphone_12": {
        "name": "Apple iPhone 12 (Wide)", "image_width": 4032, "image_height": 3024,
        "fx": 3200.0, "fy": 3200.0, "cx": 2016.0, "cy": 1512.0,
        "focal_length_mm": 4.2, "sensor_width_mm": 5.565,
        "distortion": [0.0, 0.0, 0.0, 0.0, 0.0]
    },
    "iphone_13": {
        "name": "Apple iPhone 13 (Wide)", "image_width": 4032, "image_height": 3024,
        "fx": 3274.0, "fy": 3274.0, "cx": 2016.0, "cy": 1512.0,
        "focal_length_mm": 5.1, "sensor_width_mm": 5.7,
        "distortion": [0.0, 0.0, 0.0, 0.0, 0.0]
    },
    "iphone_14": {
        "name": "Apple iPhone 14 (Wide)", "image_width": 4032, "image_height": 3024,
        "fx": 3274.0, "fy": 3274.0, "cx": 2016.0, "cy": 1512.0,
        "focal_length_mm": 5.1, "sensor_width_mm": 5.7,
        "distortion": [0.0, 0.0, 0.0, 0.0, 0.0]
    },
    "iphone_15": {
        "name": "Apple iPhone 15 (Wide)", "image_width": 4032, "image_height": 3024,
        "fx": 3390.0, "fy": 3390.0, "cx": 2016.0, "cy": 1512.0,
        "focal_length_mm": 5.3, "sensor_width_mm": 5.7,
        "distortion": [0.0, 0.0, 0.0, 0.0, 0.0]
    },
    "samsung_s22": {
        "name": "Samsung Galaxy S22 (Wide)", "image_width": 4000, "image_height": 3000,
        "fx": 3580.0, "fy": 3580.0, "cx": 2000.0, "cy": 1500.0,
        "focal_length_mm": 5.4, "sensor_width_mm": 6.0,
        "distortion": [0.0, 0.0, 0.0, 0.0, 0.0]
    },
    "samsung_s23": {
        "name": "Samsung Galaxy S23 (Wide)", "image_width": 4000, "image_height": 3000,
        "fx": 3700.0, "fy": 3700.0, "cx": 2000.0, "cy": 1500.0,
        "focal_length_mm": 5.9, "sensor_width_mm": 6.4,
        "distortion": [0.0, 0.0, 0.0, 0.0, 0.0]
    },
    "samsung_s24": {
        "name": "Samsung Galaxy S24 (Wide)", "image_width": 4000, "image_height": 3000,
        "fx": 3700.0, "fy": 3700.0, "cx": 2000.0, "cy": 1500.0,
        "focal_length_mm": 5.9, "sensor_width_mm": 6.4,
        "distortion": [0.0, 0.0, 0.0, 0.0, 0.0]
    },
    "pixel_7": {
        "name": "Google Pixel 7 (Wide)", "image_width": 4080, "image_height": 3072,
        "fx": 3430.0, "fy": 3430.0, "cx": 2040.0, "cy": 1536.0,
        "focal_length_mm": 5.4, "sensor_width_mm": 6.4,
        "distortion": [0.0, 0.0, 0.0, 0.0, 0.0]
    },
    "pixel_8": {
        "name": "Google Pixel 8 (Wide)", "image_width": 4080, "image_height": 3072,
        "fx": 3450.0, "fy": 3450.0, "cx": 2040.0, "cy": 1536.0,
        "focal_length_mm": 5.5, "sensor_width_mm": 6.4,
        "distortion": [0.0, 0.0, 0.0, 0.0, 0.0]
    },
    "nyu_kinect": {
        "name": "Microsoft Kinect (NYU Depth V2 Dataset)", "image_width": 640, "image_height": 480,
        "fx": 518.8579, "fy": 519.4696, "cx": 325.5824, "cy": 253.7362,
        "focal_length_mm": 5.4, "sensor_width_mm": 6.66,
        "distortion": [0.0, 0.0, 0.0, 0.0, 0.0]
    },
    "webcam_720p": {
        "name": "Generic 720p Webcam", "image_width": 1280, "image_height": 720,
        "fx": 920.0, "fy": 920.0, "cx": 640.0, "cy": 360.0,
        "focal_length_mm": 2.6, "sensor_width_mm": 3.6,
        "distortion": [0.0, 0.0, 0.0, 0.0, 0.0]
    },
    "webcam_1080p": {
        "name": "Generic 1080p Webcam", "image_width": 1920, "image_height": 1080,
        "fx": 1380.0, "fy": 1380.0, "cx": 960.0, "cy": 540.0,
        "focal_length_mm": 2.6, "sensor_width_mm": 3.6,
        "distortion": [0.0, 0.0, 0.0, 0.0, 0.0]
    },
    "oneplus_11": {
        "name": "OnePlus 11 (Wide)", "image_width": 4000, "image_height": 3000,
        "fx": 3500.0, "fy": 3500.0, "cx": 2000.0, "cy": 1500.0,
        "focal_length_mm": 5.6, "sensor_width_mm": 6.4,
        "distortion": [0.0, 0.0, 0.0, 0.0, 0.0]
    },
    "redmi_note_12": {
        "name": "Redmi Note 12 Pro (Wide)", "image_width": 4000, "image_height": 3000,
        "fx": 3400.0, "fy": 3400.0, "cx": 2000.0, "cy": 1500.0,
        "focal_length_mm": 5.4, "sensor_width_mm": 6.4,
        "distortion": [0.0, 0.0, 0.0, 0.0, 0.0]
    }
}


def calibrate_from_images(image_dir, pattern_size=(6, 9), square_size_mm=25.0):
    """
    Compute camera intrinsics from checkerboard images using OpenCV.
    
    HOW IT WORKS:
    1. Detect checkerboard corners in each image (2D pixel coordinates)
    2. We know the real-world 3D positions (grid of points on a flat board)
    3. OpenCV solves for the camera matrix K that maps 3D → 2D
    """
    # Build 3D coordinates of checkerboard corners (in mm)
    # These are the KNOWN real-world positions
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size_mm
    
    obj_points = []  # 3D points (same for every image — the board doesn't change)
    img_points = []  # 2D points (different for each image — camera moves)
    
    # Find all images
    files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        files.extend(glob.glob(os.path.join(image_dir, ext)))
    files = sorted(files)
    
    if not files:
        print(f"ERROR: No images found in {image_dir}")
        sys.exit(1)
    
    print(f"Processing {len(files)} images...")
    
    img_size = None
    valid = 0
    
    for f in files:
        img = cv2.imread(f)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if img_size is None:
            img_size = (gray.shape[1], gray.shape[0])
        
        found, corners = cv2.findChessboardCorners(
            gray, pattern_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        if found:
            # Refine corner positions to sub-pixel accuracy
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_points.append(objp)
            img_points.append(corners)
            valid += 1
            print(f"  [OK] {os.path.basename(f)}")
        else:
            print(f"  [SKIP] {os.path.basename(f)}")
    
    if valid < 5:
        print(f"\nOnly {valid} valid images. Need at least 5.")
        sys.exit(1)
    
    print(f"\nCalibrating with {valid} images...")
    
    # THE CORE CALIBRATION CALL
    # This finds K (camera matrix) and distortion coefficients
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, img_size, None, None
    )
    
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    # Compute reprojection error (how accurate is our calibration)
    total_err = 0
    for i in range(len(obj_points)):
        proj, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], K, dist)
        total_err += cv2.norm(img_points[i], proj, cv2.NORM_L2) / len(proj)
    reproj_err = total_err / len(obj_points)
    
    return {
        "source": "calibration",
        "image_width": img_size[0],
        "image_height": img_size[1],
        "fx": float(fx), "fy": float(fy),
        "cx": float(cx), "cy": float(cy),
        "distortion": dist.flatten().tolist(),
        "reprojection_error_pixels": float(reproj_err),
        "num_images_used": valid
    }


def get_from_database(camera_name):
    """Look up camera intrinsics from the built-in database."""
    if camera_name not in CAMERA_DATABASE:
        print(f"Camera '{camera_name}' not found. Available:")
        for k in sorted(CAMERA_DATABASE.keys()):
            print(f"  {k}")
        sys.exit(1)
    
    entry = CAMERA_DATABASE[camera_name]
    return {
        "source": f"database ({entry['name']})",
        "image_width": entry["image_width"],
        "image_height": entry["image_height"],
        "fx": entry["fx"], "fy": entry["fy"],
        "cx": entry["cx"], "cy": entry["cy"],
        "distortion": entry["distortion"],
        "focal_length_mm": entry["focal_length_mm"],
        "sensor_width_mm": entry["sensor_width_mm"]
    }


def main():
    parser = argparse.ArgumentParser(description="Camera Calibration")
    parser.add_argument("--image_dir", type=str, default="data/calibration_images")
    parser.add_argument("--use_database", action="store_true")
    parser.add_argument("--camera", type=str, default="nyu_kinect",
                       help="Camera name from database")
    parser.add_argument("--cols", type=int, default=6)
    parser.add_argument("--rows", type=int, default=9)
    parser.add_argument("--square_size", type=float, default=25.0, help="mm")
    parser.add_argument("--output", type=str, default="camera_params.json")
    parser.add_argument("--list_cameras", action="store_true")
    args = parser.parse_args()
    
    if args.list_cameras:
        print("\nAvailable cameras:")
        print("-" * 65)
        for key, cam in sorted(CAMERA_DATABASE.items()):
            print(f"  {key:20s}  {cam['name']}")
            print(f"  {'':20s}  {cam['image_width']}x{cam['image_height']}  "
                  f"fx={cam['fx']:.0f}  f={cam['focal_length_mm']}mm")
            print()
        return
    
    if args.use_database:
        params = get_from_database(args.camera)
    else:
        params = calibrate_from_images(
            args.image_dir,
            pattern_size=(args.cols, args.rows),
            square_size_mm=args.square_size
        )
    
    # Save
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = os.path.join(project_root, args.output)
    with open(out_path, 'w') as f:
        json.dump(params, f, indent=2)
    
    # Display
    print("\n" + "=" * 55)
    print(" CAMERA INTRINSIC PARAMETERS")
    print("=" * 55)
    print(f" Source: {params['source']}")
    print(f" Resolution: {params['image_width']}x{params['image_height']}")
    print(f" fx = {params['fx']:.2f} pixels")
    print(f" fy = {params['fy']:.2f} pixels")
    print(f" cx = {params['cx']:.2f} pixels")
    print(f" cy = {params['cy']:.2f} pixels")
    if 'reprojection_error_pixels' in params:
        err = params['reprojection_error_pixels']
        quality = "Excellent" if err < 0.3 else "Good" if err < 0.5 else "OK" if err < 1.0 else "Poor"
        print(f" Reprojection error: {err:.4f} px ({quality})")
    print(f"\n Intrinsic matrix K:")
    print(f"   | {params['fx']:8.2f}    0.00  {params['cx']:8.2f} |")
    print(f"   |     0.00  {params['fy']:8.2f}  {params['cy']:8.2f} |")
    print(f"   |     0.00    0.00      1.00 |")
    print(f"\n Saved to: {out_path}")
    print("=" * 55)

if __name__ == "__main__":
    main()
