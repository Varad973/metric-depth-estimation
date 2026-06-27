"""
STEP 2C: Add New Custom Objects to Training
=============================================
Train the model to recognize NEW objects that aren't in any existing dataset.

EXAMPLE: You have a custom 3D-printed part, a specific piece of furniture,
         or any object the model hasn't seen before.

PROCEDURE:
    1. Place your custom object at multiple known distances
    2. Take photos at each distance (minimum 10-20 per object)
    3. Run this script to add them to your training set
    4. Re-train the model

RUN:
    python step2_data_collection/add_custom_objects.py \
        --object_name "robot_arm" \
        --photos_dir my_robot_photos/ \
        --output_dir data/custom
"""

import os
import sys
import re
import json
import glob
import argparse
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def add_custom_object(object_name, photos_dir, output_dir, 
                       object_real_width_m=None, object_real_height_m=None):
    """
    Add a new custom object to the training dataset.
    
    Photos should be named with distance:
        robot_arm_1.0m_001.jpg
        robot_arm_2.0m_001.jpg
        robot_arm_3.5m_front.jpg
    
    OR provide a CSV file with labels.
    """
    import cv2
    
    rgb_dir = os.path.join(output_dir, "rgb")
    depth_dir = os.path.join(output_dir, "depth")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    
    # Find existing files to continue numbering
    existing = glob.glob(os.path.join(rgb_dir, "custom_*.png"))
    start_idx = len(existing)
    
    # Find photos
    files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        files.extend(glob.glob(os.path.join(photos_dir, ext)))
    
    if not files:
        print(f"No images found in {photos_dir}")
        return
    
    print(f"Adding '{object_name}' — {len(files)} photos")
    
    entries = []
    count = 0
    
    for fpath in sorted(files):
        fname = os.path.basename(fpath)
        
        # Parse depth from filename
        match = re.search(r'(\d+\.?\d*)m', fname)
        if not match:
            print(f"  [SKIP] {fname} — add depth to filename (e.g., {object_name}_2.0m.jpg)")
            continue
        
        depth = float(match.group(1))
        
        img = cv2.imread(fpath)
        if img is None:
            continue
        h, w = img.shape[:2]
        
        name = f"custom_{start_idx + count:05d}"
        cv2.imwrite(os.path.join(rgb_dir, f"{name}.png"), img)
        
        # Create depth map
        # For a single object at known distance, the object region has that depth
        # Background gets a larger depth value
        depth_map = np.full((h, w), depth, dtype=np.float32)
        np.save(os.path.join(depth_dir, f"{name}.npy"), depth_map)
        
        entries.append({
            "name": name,
            "object": object_name,
            "depth_m": depth,
            "original_file": fname
        })
        count += 1
        print(f"  [OK] {fname} → {depth}m (as {name})")
    
    # Update metadata
    meta_path = os.path.join(output_dir, "custom_objects.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    else:
        meta = {"objects": {}}
    
    meta["objects"][object_name] = {
        "num_images": count,
        "entries": entries,
        "real_width_m": object_real_width_m,
        "real_height_m": object_real_height_m,
    }
    
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    
    # Rebuild train/val split including all data
    all_files = sorted([
        f.replace('.png', '') for f in os.listdir(rgb_dir) if f.endswith('.png')
    ])
    np.random.seed(42)
    np.random.shuffle(all_files)
    split = int(len(all_files) * 0.8)
    
    with open(os.path.join(output_dir, "train.txt"), 'w') as f:
        f.write('\n'.join(sorted(all_files[:split])))
    with open(os.path.join(output_dir, "val.txt"), 'w') as f:
        f.write('\n'.join(sorted(all_files[split:])))
    
    print(f"\nAdded {count} images of '{object_name}'")
    print(f"Total dataset: {len(all_files)} images (train: {split}, val: {len(all_files)-split})")
    print(f"\nTo train: python step3_training/train.py --config configs/custom.yaml")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--object_name", required=True, help="Name of the new object")
    parser.add_argument("--photos_dir", required=True, help="Directory with photos")
    parser.add_argument("--output_dir", default="data/custom")
    parser.add_argument("--object_width", type=float, default=None, help="Real width in meters")
    parser.add_argument("--object_height", type=float, default=None, help="Real height in meters")
    args = parser.parse_args()
    
    out = args.output_dir
    if not os.path.isabs(out):
        out = os.path.join(PROJECT_ROOT, out)
    
    add_custom_object(args.object_name, args.photos_dir, out,
                      args.object_width, args.object_height)

if __name__ == "__main__":
    main()
