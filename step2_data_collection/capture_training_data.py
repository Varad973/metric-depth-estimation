"""
STEP 2B: Capture Your Own Training Data at Measured Distances
=============================================================
Take photos of objects at known distances to create custom training data.
This lets you train the model on NEW objects not in any existing dataset.

RUN:
    # Get instructions for phone-based capture:
    python step2_data_collection/capture_training_data.py --mode manual

    # Process photos you've already taken:
    python step2_data_collection/capture_training_data.py --mode process --input_dir my_photos/

    # Live webcam capture:
    python step2_data_collection/capture_training_data.py --mode webcam

HOW TO NAME YOUR PHOTOS:
    Include the measured distance in the filename:
    photo_2.0m_001.jpg   → object at 2.0 meters
    photo_3.5m_002.jpg   → object at 3.5 meters
    chair_1.5m.png       → chair at 1.5 meters
"""

import os
import sys
import re
import json
import glob
import argparse
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def show_manual_instructions(output_dir):
    """Print step-by-step instructions for phone-based data capture."""
    
    rgb_dir = os.path.join(output_dir, "rgb")
    os.makedirs(rgb_dir, exist_ok=True)
    
    # Create template CSV
    csv_path = os.path.join(output_dir, "depth_labels.csv")
    with open(csv_path, 'w') as f:
        f.write("filename,depth_meters,object_type\n")
        f.write("# Fill in your photos below:\n")
        f.write("# photo_001.jpg,2.0,chair\n")
        f.write("# photo_002.jpg,3.5,person\n")
    
    print("=" * 60)
    print(" CUSTOM DATA CAPTURE — INSTRUCTIONS")
    print("=" * 60)
    print()
    print(" WHAT YOU NEED:")
    print("   - Phone camera (or any camera)")
    print("   - Measuring tape or ruler")
    print("   - Objects to photograph (chairs, cups, books, etc.)")
    print()
    print(" PROCEDURE:")
    print("   1. Set phone on tripod or stable surface")
    print("   2. Place object at MEASURED distance:")
    print("      → 1.0m, 1.5m, 2.0m, 2.5m, 3.0m, 4.0m, 5.0m")
    print("   3. Take photo — object should fill most of frame")
    print("   4. NAME the file with the distance:")
    print("      → chair_2.0m_001.jpg")
    print("      → bottle_1.5m.png")
    print("   5. Repeat for many objects at many distances")
    print("   6. Aim for 50-200+ images")
    print()
    print(f" COPY photos to: {rgb_dir}/")
    print()
    print(" THEN RUN:")
    print(f"   python step2_data_collection/capture_training_data.py "
          f"--mode process --input_dir {rgb_dir}")
    print()
    print(f" ALTERNATIVE: Fill in {csv_path}")
    print("=" * 60)


def process_photos(input_dir, output_dir):
    """
    Process photos with depth info in filenames.
    Creates proper dataset structure for training.
    """
    import cv2
    
    rgb_dir = os.path.join(output_dir, "rgb")
    depth_dir = os.path.join(output_dir, "depth")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    
    # Find images
    files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    if not files:
        print(f"No images found in {input_dir}")
        return
    
    # Also check for CSV labels
    csv_path = os.path.join(output_dir, "depth_labels.csv")
    csv_labels = {}
    if os.path.exists(csv_path):
        import csv
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row['filename'].startswith('#'):
                    csv_labels[row['filename']] = float(row['depth_meters'])
    
    entries = []
    count = 0
    
    for fpath in sorted(files):
        fname = os.path.basename(fpath)
        
        # Try CSV first, then filename parsing
        depth = csv_labels.get(fname)
        if depth is None:
            match = re.search(r'(\d+\.?\d*)m', fname)
            if match:
                depth = float(match.group(1))
        
        if depth is None:
            print(f"  [SKIP] {fname} — no depth info (add Xm to filename)")
            continue
        
        # Read image
        img = cv2.imread(fpath)
        if img is None:
            continue
        h, w = img.shape[:2]
        
        # Save as standardized format
        name = f"custom_{count:05d}"
        cv2.imwrite(os.path.join(rgb_dir, f"{name}.png"), img)
        
        # Create depth map — uniform depth for the whole image
        # (this is approximate: assumes object fills most of the frame)
        depth_map = np.full((h, w), depth, dtype=np.float32)
        np.save(os.path.join(depth_dir, f"{name}.npy"), depth_map)
        
        entries.append({"name": name, "depth": depth, "original": fname})
        count += 1
        print(f"  [OK] {fname} → {depth}m")
    
    if count == 0:
        print("No valid images processed!")
        return
    
    # Create train/val split
    names = [e["name"] for e in entries]
    np.random.seed(42)
    np.random.shuffle(names)
    split = int(len(names) * 0.8)
    
    with open(os.path.join(output_dir, "train.txt"), 'w') as f:
        f.write('\n'.join(sorted(names[:split])))
    with open(os.path.join(output_dir, "val.txt"), 'w') as f:
        f.write('\n'.join(sorted(names[split:])))
    
    # Save info
    with open(os.path.join(output_dir, "dataset_info.json"), 'w') as f:
        json.dump({
            "num_samples": count,
            "entries": entries,
            "train": split,
            "val": count - split
        }, f, indent=2)
    
    print(f"\nProcessed {count} images → train: {split}, val: {count - split}")
    print("Next: python step3_training/train.py --config configs/custom.yaml")


def webcam_capture(output_dir):
    """Interactive webcam-based capture with depth annotation."""
    import cv2
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam. Use --mode manual instead.")
        return
    
    rgb_dir = os.path.join(output_dir, "rgb")
    depth_dir = os.path.join(output_dir, "depth")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    
    current_depth = 2.0
    count = 0
    
    print("WEBCAM CAPTURE — SPACE=capture, UP/DOWN=adjust depth, Q=quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        display = frame.copy()
        cv2.putText(display, f"Depth: {current_depth:.1f}m | Captured: {count}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Capture", display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            name = f"custom_{count:05d}"
            cv2.imwrite(os.path.join(rgb_dir, f"{name}.png"), frame)
            h, w = frame.shape[:2]
            np.save(os.path.join(depth_dir, f"{name}.npy"),
                   np.full((h, w), current_depth, dtype=np.float32))
            count += 1
            print(f"  #{count}: depth={current_depth}m")
        elif key in [82, 0]:  # Up
            current_depth = min(current_depth + 0.5, 20.0)
        elif key in [84, 1]:  # Down
            current_depth = max(current_depth - 0.5, 0.3)
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {count} images to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["manual", "process", "webcam"], default="manual")
    parser.add_argument("--output_dir", default="data/custom")
    parser.add_argument("--input_dir", default=None)
    args = parser.parse_args()
    
    out = os.path.join(PROJECT_ROOT, args.output_dir)
    
    if args.mode == "manual":
        show_manual_instructions(out)
    elif args.mode == "process":
        inp = args.input_dir or os.path.join(out, "rgb")
        process_photos(inp, out)
    elif args.mode == "webcam":
        webcam_capture(out)

if __name__ == "__main__":
    main()
