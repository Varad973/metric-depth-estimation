"""
STEP 1B: Capture Checkerboard Photos for Calibration
"""

import cv2
import os
import sys
import glob
import argparse


def capture_from_webcam(output_dir, pattern_size=(6, 9)):
    """Interactive webcam capture of checkerboard images."""
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam!")
        print("If using phone photos, use: --input_dir path/to/photos")
        sys.exit(1)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print("=" * 55)
    print(f" Camera: {w}x{h}")
    print(f" Pattern: {pattern_size[0]}x{pattern_size[1]} inner corners")
    print(f" SPACE = capture | Q = quit")
    print("=" * 55)
    
    os.makedirs(output_dir, exist_ok=True)
    count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        found, corners = cv2.findChessboardCorners(
            gray, pattern_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK
        )
        
        if found:
            cv2.drawChessboardCorners(display, pattern_size, corners, found)
            cv2.putText(display, "PATTERN FOUND - SPACE to capture",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(display, "Looking for checkerboard...",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        cv2.putText(display, f"Captured: {count}/20",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.imshow("Checkerboard Capture", display)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' ') and found:
            path = os.path.join(output_dir, f"calib_{count:03d}.png")
            cv2.imwrite(path, frame)
            count += 1
            print(f"  Captured #{count}: {path}")
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nTotal: {count} images saved to {output_dir}")
    if count >= 15:
        print("Now run: python step1_camera_calibration/calibrate_camera.py")
    else:
        print(f"WARNING: Only {count} images. Try to get at least 15.")


def verify_existing_images(input_dir, pattern_size=(6, 9)):
    """Check which existing photos have detectable checkerboard patterns."""
    
    files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    if not files:
        print(f"No images found in {input_dir}")
        sys.exit(1)
    
    valid = 0
    for f in sorted(files):
        img = cv2.imread(f)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, _ = cv2.findChessboardCorners(gray, pattern_size,
                    cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        status = "OK" if found else "SKIP"
        print(f"  [{status}] {os.path.basename(f)}")
        if found:
            valid += 1
    
    print(f"\nValid: {valid}/{len(files)}")
    if valid >= 10:
        print(f"Run: python step1_camera_calibration/calibrate_camera.py --image_dir {input_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=None,
                       help="Directory with existing checkerboard photos")
    parser.add_argument("--output_dir", type=str, default="data/calibration_images")
    parser.add_argument("--cols", type=int, default=6)
    parser.add_argument("--rows", type=int, default=9)
    args = parser.parse_args()
    
    pattern = (args.cols, args.rows)
    
    if args.input_dir:
        verify_existing_images(args.input_dir, pattern)
    else:
        capture_from_webcam(args.output_dir, pattern)

if __name__ == "__main__":
    main()
