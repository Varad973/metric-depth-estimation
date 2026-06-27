"""
STEP 2A: Download NYU Depth V2 Dataset
=======================================
This is the most popular indoor depth dataset with REAL metric depth values.

Contains:
    - 1449 RGB images + matching depth maps
    - Depth in meters (0.5m to ~10m)
    - Known Kinect camera intrinsics
    - Indoor scenes: bedrooms, kitchens, offices, etc.

RUN:  python step2_data_collection/download_nyu_dataset.py

SIZE: ~2.8 GB download, ~4 GB after extraction
"""

import os
import sys
import numpy as np
import json
from tqdm import tqdm

# Project root for relative paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def download_nyu(output_dir):
    """Download the NYU Depth V2 labeled .mat file."""
    try:
        import gdown
    except ImportError:
        os.system(f"{sys.executable} -m pip install gdown")
        import gdown
    
    os.makedirs(output_dir, exist_ok=True)
    mat_path = os.path.join(output_dir, "nyu_depth_v2_labeled.mat")
    
    if os.path.exists(mat_path):
        print(f"Already downloaded: {mat_path}")
        return mat_path
    
    print("Downloading NYU Depth V2 (~2.8 GB)...")
    print("This may take 5-15 minutes depending on your internet.")
    
    url = "https://drive.google.com/uc?id=1WoOZOBpOWfmwe7bknWS5PMUCLBPFKTOw"
    try:
        gdown.download(url, mat_path, quiet=False)
        return mat_path
    except Exception as e:
        print(f"\nDownload failed: {e}")
        print("\nMANUAL DOWNLOAD:")
        print("1. Visit: https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html")
        print("2. Download 'Labeled dataset (~2.8 GB)'")
        print(f"3. Save as: {mat_path}")
        print("4. Re-run this script with --skip_download")
        return None


def extract_dataset(mat_path, output_dir, max_samples=0):
    """
    Extract RGB-depth pairs from the NYU Depth V2 zip archive.

    The zip contains:
        data/nyu2_train/<scene>/<id>.jpg  (RGB)
        data/nyu2_train/<scene>/<id>.png  (depth as 16-bit PNG, millimeters)
        data/nyu2_train.csv               (rgb_path,depth_path pairs)
        data/nyu2_test/<id>_colors.png    (RGB)
        data/nyu2_test/<id>_depth.png     (depth)
        data/nyu2_test.csv
    """
    import zipfile
    import io
    from PIL import Image
    import cv2

    print(f"Extracting dataset from {mat_path}...")

    rgb_dir = os.path.join(output_dir, "rgb")
    depth_dir = os.path.join(output_dir, "depth")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    z = zipfile.ZipFile(mat_path, 'r')

    # Read the train CSV to get paired RGB-depth paths
    train_csv = z.read('data/nyu2_train.csv').decode().strip().split('\n')
    test_csv = z.read('data/nyu2_test.csv').decode().strip().split('\n')

    # Combine train + test pairs
    pairs = []
    for line in train_csv:
        parts = line.strip().split(',')
        if len(parts) == 2:
            pairs.append((parts[0].strip(), parts[1].strip()))

    n_train = len(pairs)

    for line in test_csv:
        parts = line.strip().split(',')
        if len(parts) == 2:
            pairs.append((parts[0].strip(), parts[1].strip()))

    N = len(pairs)
    if max_samples > 0 and max_samples < N:
        # Shuffle deterministically then take a subset
        np.random.seed(42)
        indices = np.random.permutation(N).tolist()
        pairs = [pairs[i] for i in indices[:max_samples]]
        N = len(pairs)
        n_train = min(n_train, N)
    print(f"Extracting {N} RGB-Depth pairs")

    min_d, max_d, sum_d = float('inf'), 0, 0

    for i, (rgb_zip_path, depth_zip_path) in enumerate(tqdm(pairs, desc="Extracting")):
        # Read RGB from zip
        try:
            rgb_data = z.read(rgb_zip_path)
            rgb_img = Image.open(io.BytesIO(rgb_data)).convert('RGB')
            rgb = np.array(rgb_img)
        except Exception as e:
            print(f"  Skipping {rgb_zip_path}: {e}")
            continue

        # Read depth PNG from zip
        try:
            depth_data = z.read(depth_zip_path)
            depth_img = Image.open(io.BytesIO(depth_data))
            depth = np.array(depth_img).astype(np.float32)
            # Depth PNGs are typically 16-bit with values in millimeters
            # or 8-bit with scaled values. Convert to meters.
            if depth.max() > 100:
                depth = depth / 1000.0  # millimeters to meters
            elif depth.max() > 10:
                depth = depth / 25.5  # 8-bit scaled (0-255 -> ~0-10m)
            # else already in meters or relative scale
        except Exception as e:
            print(f"  Skipping {depth_zip_path}: {e}")
            continue

        # Track statistics
        valid = depth[depth > 0]
        if len(valid) > 0:
            min_d = min(min_d, float(valid.min()))
            max_d = max(max_d, float(valid.max()))
            sum_d += float(valid.mean())

        # Save as individual files
        rgb_img.save(os.path.join(rgb_dir, f"{i:05d}.png"))
        np.save(os.path.join(depth_dir, f"{i:05d}.npy"), depth)

    z.close()

    # Create train/val split (use original train as train, test as val)
    train_ids = list(range(n_train))
    val_ids = list(range(n_train, N))

    # If test set is too small, split train instead
    if len(val_ids) < 100:
        np.random.seed(42)
        all_ids = list(range(N))
        np.random.shuffle(all_ids)
        split = int(N * 0.85)
        train_ids = sorted(all_ids[:split])
        val_ids = sorted(all_ids[split:])

    with open(os.path.join(output_dir, "train.txt"), 'w') as fp:
        fp.write('\n'.join(f"{i:05d}" for i in train_ids))
    with open(os.path.join(output_dir, "val.txt"), 'w') as fp:
        fp.write('\n'.join(f"{i:05d}" for i in val_ids))

    # Save dataset info
    info = {
        "num_samples": N,
        "train_samples": len(train_ids),
        "val_samples": len(val_ids),
        "depth_min_meters": min_d,
        "depth_max_meters": max_d,
        "depth_mean_meters": sum_d / N if N > 0 else 0,
        "depth_unit": "meters",
        "camera": {
            "name": "Microsoft Kinect",
            "fx": 518.8579, "fy": 519.4696,
            "cx": 325.5824, "cy": 253.7362,
            "image_width": 640, "image_height": 480
        }
    }

    with open(os.path.join(output_dir, "dataset_info.json"), 'w') as fp:
        json.dump(info, fp, indent=2)

    print(f"\nDone!")
    print(f"  RGB images: {rgb_dir}/")
    print(f"  Depth maps: {depth_dir}/")
    print(f"  Train: {len(train_ids)}, Val: {len(val_ids)}")
    print(f"  Depth range: {min_d:.2f}m to {max_d:.2f}m")

    return info


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/nyu_depth_v2")
    parser.add_argument("--skip_download", action="store_true")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Limit extraction to N samples (0 = all)")
    args = parser.parse_args()
    
    out = os.path.join(PROJECT_ROOT, args.output_dir)
    
    if not args.skip_download:
        mat_path = download_nyu(out)
        if mat_path is None:
            return
    else:
        mat_path = os.path.join(out, "nyu_depth_v2_labeled.mat")
    
    if os.path.exists(mat_path):
        extract_dataset(mat_path, out, max_samples=args.max_samples)
        print("\nNext: python step3_training/train.py")

if __name__ == "__main__":
    main()
